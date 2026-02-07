#include <cmath>
#include <complex>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

// DMCWithSign is defined in DMC.cpp
std::pair<double, double> DMCWithSign(
    double tau,
    const double k,
    const double t,
    const double mu,
    const std::function<double(double, double, int, int, int)>& gAbsFunc,
    const std::function<std::complex<double>(double, double, int, int, int)>& gPhaseFunc,
    const std::function<double(int)>& wFunc,
    const std::function<double(double, int)>& eFunc,
    int electronBand,
    int phononMode,
    int numElectronBands,
    int numPhononModes,
    int numMomentumPoints,
    const double up_lim,
    const int N,
    const int bins,
    std::vector<double>& histogram,
    std::vector<int>& orderDist,
    std::vector<double>& exact_estimator,
    std::vector<double>& exact_estimator_signed,
    std::vector<double>& exact_estimator_err,
    std::vector<double>& exact_estimator_signed_err,
    double& avgSign,
    double& avgPhaseMag,
    bool tauChange);

namespace {
bool parseIntArg(const std::string& arg, const std::string& key, int& value) {
    const std::string prefix = key + "=";
    if (arg.rfind(prefix, 0) != 0) {
        return false;
    }
    value = std::stoi(arg.substr(prefix.size()));
    return true;
}

bool parseDoubleArg(const std::string& arg, const std::string& key, double& value) {
    const std::string prefix = key + "=";
    if (arg.rfind(prefix, 0) != 0) {
        return false;
    }
    value = std::stod(arg.substr(prefix.size()));
    return true;
}

bool parseStringArg(const std::string& arg, const std::string& key, std::string& value) {
    const std::string prefix = key + "=";
    if (arg.rfind(prefix, 0) != 0) {
        return false;
    }
    value = arg.substr(prefix.size());
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    int N = 1e6;
    int bins = 80;
    int qPoints = 256;
    int bands = 2;
    double tHop = 1.0;
    double omegaOverT = 0.5;
    double lambda = 1.0;
    double mu = -4.7;
    double k = 0.0;
    double tau0 = 6.0;
    double upLim = 9.0;
    double phaseAmp = 0.3;
    bool intrabandOnly = false;
    bool bandDispersion = false;
    double bandOffset = 0.1;

    std::string variant = "abs";
    std::string outPath;
    std::string outGreenPath;
    bool outProvided = false;
    bool outGreenProvided = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--intraband-only") {
            intrabandOnly = true;
            continue;
        }
        if (arg == "--band-dispersion") {
            bandDispersion = true;
            continue;
        }
        if (parseIntArg(arg, "--N", N) ||
            parseIntArg(arg, "--bins", bins) ||
            parseIntArg(arg, "--q-points", qPoints) ||
            parseIntArg(arg, "--bands", bands) ||
            parseDoubleArg(arg, "--t", tHop) ||
            parseDoubleArg(arg, "--omega-over-t", omegaOverT) ||
            parseDoubleArg(arg, "--lambda", lambda) ||
            parseDoubleArg(arg, "--mu", mu) ||
            parseDoubleArg(arg, "--k", k) ||
            parseDoubleArg(arg, "--tau0", tau0) ||
            parseDoubleArg(arg, "--up-lim", upLim) ||
            parseDoubleArg(arg, "--phase", phaseAmp) ||
            parseDoubleArg(arg, "--band-offset", bandOffset) ||
            parseStringArg(arg, "--variant", variant) ||
            parseStringArg(arg, "--out", outPath) ||
            parseStringArg(arg, "--out-green", outGreenPath)) {
            if (arg.rfind("--out=", 0) == 0) {
                outProvided = true;
            }
            if (arg.rfind("--out-green=", 0) == 0) {
                outGreenProvided = true;
            }
            continue;
        }
        std::cerr << "Unknown argument: " << arg << "\n";
        return 1;
    }

    if (bands < 1) {
        std::cerr << "Invalid --bands (must be >= 1)\n";
        return 1;
    }

    const int electronBand = 0;
    const int phononMode = 0;
    const int numElectronBands = bands;
    const int numPhononModes = 1;

    const double omega = omegaOverT * tHop;
    const double g = std::sqrt(std::max(0.0, lambda * omega * tHop));

    auto eFunc = [mu, tHop, bandDispersion, bandOffset](double kElectron, int band) {
        if (bandDispersion) {
            return -2.0 * tHop * std::cos(static_cast<double>(band) * kElectron)
                   - bandOffset * static_cast<double>(band)
                   - mu;
        }
        return -2.0 * tHop * std::cos(kElectron) - mu;
    };
    auto wFunc = [omega](int /*mode*/) { return omega; };

    auto gReal = [g, intrabandOnly](double /*kElectron*/, double qPhonon,
                         int bandIn, int bandOut, int /*mode*/) {
        if (intrabandOnly && bandIn != bandOut) {
            return 0.0;
        }
        return 2.0 * g * std::sin(0.5 * qPhonon);
    };
    auto gAbs = [gReal](double kElectron, double qPhonon,
                        int bandIn, int bandOut, int mode) {
        return std::abs(gReal(kElectron, qPhonon, bandIn, bandOut, mode));
    };
    auto gPhase = [g, phaseAmp, intrabandOnly](double /*kElectron*/, double qPhonon,
                                    int bandIn, int bandOut, int /*mode*/) {
        if (intrabandOnly && bandIn != bandOut) {
            return std::complex<double>(0.0, 0.0);
        }
        const double amp = 2.0 * g * std::sin(0.5 * qPhonon);
        const double phi = phaseAmp * qPhonon;
        const std::complex<double> phase = std::exp(std::complex<double>(0.0, phi));
        return amp * phase;
    };
    auto gPhaseUnity = [](double /*kElectron*/, double /*qPhonon*/,
                          int /*bandIn*/, int /*bandOut*/, int /*mode*/) {
        return std::complex<double>(1.0, 0.0);
    };

    if (variant != "abs" && variant != "signed") {
        std::cerr << "Invalid --variant (use abs or signed)\n";
        return 1;
    }

    if (!outProvided) {
        outPath = (variant == "signed")
            ? "sign_oscillation/results/sign_oscillation_signed.csv"
            : "sign_oscillation/results/sign_oscillation_abs.csv";
    }
    if (!outGreenProvided) {
        outGreenPath = (variant == "signed")
            ? "sign_oscillation/results/sign_oscillation_signed_green.csv"
            : "sign_oscillation/results/sign_oscillation_abs_green.csv";
    }

    std::ofstream out(outPath);
    if (!out.is_open()) {
        std::cerr << "Cannot open output file: " << outPath << "\n";
        return 1;
    }
    std::ofstream outGreen(outGreenPath);
    if (!outGreen.is_open()) {
        std::cerr << "Cannot open output file: " << outGreenPath << "\n";
        return 1;
    }

    out << std::setprecision(12);
    outGreen << std::setprecision(12);
    out << "variant,lambda,g,t,omega,omega_over_t,mu,bands,avg_sign,avg_phase_mag,avg_order,energy_estimator\n";
    outGreen << "tau,green_estimator,green_error\n";

    std::vector<double> histogram;
    std::vector<int> orderDist;
    std::vector<double> exactAbs;
    std::vector<double> exactSigned;
    std::vector<double> exactAbsErr;
    std::vector<double> exactSignedErr;
    double avgSign = 0.0;
    double avgPhaseMag = 0.0;

    std::cout << "Running sign-oscillation demo (" << variant << ")\n";
    std::cout << "lambda=" << lambda
              << " g=" << g
              << " mu=" << mu
              << " bands=" << bands
              << " N=" << N << "\n";

    std::function<std::complex<double>(double, double, int, int, int)> phaseFunc;
    if (variant == "signed") {
        phaseFunc = gPhase;
    } else {
        phaseFunc = gPhaseUnity;
    }

    auto [avgOrder, energyEstimator] = DMCWithSign(
        tau0,
        k,
        tHop,
        mu,
        gAbs,
        phaseFunc,
        wFunc,
        eFunc,
        electronBand,
        phononMode,
        numElectronBands,
        numPhononModes,
        qPoints,
        upLim,
        N,
        bins,
        histogram,
        orderDist,
        exactAbs,
        exactSigned,
        exactAbsErr,
        exactSignedErr,
        avgSign,
        avgPhaseMag,
        true);

    out << variant << ","
        << lambda << ","
        << g << ","
        << tHop << ","
        << omega << ","
        << omegaOverT << ","
        << mu << ","
        << bands << ","
        << avgSign << ","
        << avgPhaseMag << ","
        << avgOrder << ","
        << energyEstimator << "\n";

    const double binWidth = upLim / bins;
    const std::vector<double>& selected =
        (variant == "signed") ? exactSigned : exactAbs;
    const std::vector<double>& selectedErr =
        (variant == "signed") ? exactSignedErr : exactAbsErr;
    for (int b = 0; b < bins; ++b) {
        const double tauCenter = (b + 0.5) * binWidth;
        outGreen << tauCenter << ","
                 << selected[b] << ","
                 << selectedErr[b] << "\n";
    }

    out.close();
    outGreen.close();
    std::cout << "avg sign (Re)=" << avgSign
              << " avg |phase|=" << avgPhaseMag << "\n";
    std::cout << "Saved: " << outPath << "\n";
    std::cout << "Saved: " << outGreenPath << "\n";
    return 0;
}
