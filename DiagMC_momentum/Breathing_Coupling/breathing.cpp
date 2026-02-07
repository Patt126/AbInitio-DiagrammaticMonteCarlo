#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// DMC is defined in DMC.cpp
std::pair<double, double> DMC(
    double tau,
    const double k,
    const double t,
    const double mu,
    const std::function<double(double, double, int, int, int)>& gFunc,
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
    // Paper setup for Fig. 2(a): t=1, Omega/t=0.5, k=0, lambda in [0,2]
    int N = 1e6;
    int bins = 100;
    int qPoints = 256;
    double tHop = 1.0;
    double omegaOverT = 0.5;
    double lambdaMin = 0.0;
    double lambdaMax = 2.0;
    double lambdaStep = 0.1;
    double k = 0.0;
    double tau0 = 10.0;
    double upLim = 50.0;
    double muOffset = 0.32;
    double muAlpha = 1.0;
    double muMomentum = 0.0;
    bool useFixedMu = false;
    double muFixed = -2.4;
    std::string outPath = "Breathing_Coupling/results/breathing_momentum.csv";
    std::string outGreenPath = "Breathing_Coupling/results/breathing_green_momentum.csv";

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (parseIntArg(arg, "--N", N) ||
            parseIntArg(arg, "--bins", bins) ||
            parseIntArg(arg, "--q-points", qPoints) ||
            parseDoubleArg(arg, "--t", tHop) ||
            parseDoubleArg(arg, "--omega-over-t", omegaOverT) ||
            parseDoubleArg(arg, "--lambda-min", lambdaMin) ||
            parseDoubleArg(arg, "--lambda-max", lambdaMax) ||
            parseDoubleArg(arg, "--lambda-step", lambdaStep) ||
            parseDoubleArg(arg, "--k", k) ||
            parseDoubleArg(arg, "--tau0", tau0) ||
            parseDoubleArg(arg, "--up-lim", upLim) ||
            parseDoubleArg(arg, "--mu-offset", muOffset) ||
            parseDoubleArg(arg, "--mu-alpha", muAlpha) ||
            parseDoubleArg(arg, "--mu-momentum", muMomentum) ||
            parseDoubleArg(arg, "--mu", muFixed) ||
            parseStringArg(arg, "--out", outPath) ||
            parseStringArg(arg, "--out-green", outGreenPath)) {
            if (arg.rfind("--mu=", 0) == 0) {
                useFixedMu = true;
            }
            continue;
        }
        std::cerr << "Unknown argument: " << arg << "\n";
        return 1;
    }

    if (lambdaStep <= 0.0 || lambdaMax < lambdaMin) {
        std::cerr << "Invalid lambda range/step\n";
        return 1;
    }

    const double omega = omegaOverT * tHop;
    const int electronBand = 0;
    const int phononMode = 0;
    const int numElectronBands = 1;
    const int numPhononModes = 1;

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

    out << "lambda,g,t,omega,omega_over_t,k,mu_used,avg_order,energy,energy_over_t\n";
    outGreen << "lambda,g,tau,histogram_data,green_estimator\n";
    out << std::setprecision(12);
    outGreen << std::setprecision(12);

    std::vector<double> histogram;
    std::vector<int> orderDist;
    std::vector<double> exactEstimator;

    // Keep mu below the current energy by a configurable offset so G(tau) decays.
    double muCurrent = -2.3 * tHop;
    double muPrev = muCurrent;

    std::cout << "Running Fig.2(a)-style breathing-mode benchmark\n";
    std::cout << "t=" << tHop
              << " Omega/t=" << omegaOverT
              << " k=" << k
              << " N=" << N
              << " lambda in [" << lambdaMin << ", " << lambdaMax << "]\n";

    for (double lambda = lambdaMin; lambda <= lambdaMax + 1e-12; lambda += lambdaStep) {
        const double g = std::sqrt(std::max(0.0, lambda * omega * tHop));  // lambda = g^2/(Omega t)
        const double muUsed = useFixedMu ? muFixed : muCurrent;

        auto eFunc = [muUsed, tHop](double kElectron, int /*band*/) {
            return -2.0 * tHop * std::cos(kElectron) - muUsed;
        };
        auto wFunc = [omega](int /*mode*/) { return omega; };

        // Paper breathing mode: g_q = -2 i g sin(q/2).
        // Here we use the real positive magnitude |g_q| = 2 g |sin(q/2)| in this sign-free implementation.
        auto gFunc = [g](double /*kElectron*/, double qPhonon,
                         int /*previousBand*/, int /*newBand*/, int /*mode*/) {
            return 2.0 * g * std::abs(std::sin(0.5 * qPhonon));
        };

        N = (lambda>1) ? 1e7 : 1e6;
        lambdaStep = (lambda>1.2)? 0.05 : 0.1 ;
        
        auto [avgOrder, energyEstimator] = DMC(
            tau0,
            k,
            tHop,
            muUsed,
            gFunc,
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
            exactEstimator,
            true);

        out << lambda << ","
            << g << ","
            << tHop << ","
            << omega << ","
            << omegaOverT << ","
            << k << ","
            << muUsed << ","
            << avgOrder << ","
            << energyEstimator << ","
            << (energyEstimator / tHop) << "\n";
        const double binWidth = upLim / bins;
        for (int b = 0; b < bins; ++b) {
            const double tauCenter = (b + 0.5) * binWidth;
            outGreen << lambda << ","
                     << g << ","
                     << tauCenter << ","
                     << histogram[b] << ","
                     << exactEstimator[b] << "\n";
        }

        std::cout << "lambda=" << lambda
                  << " g=" << g
                  << " mu=" << muUsed
                  << " <order>=" << avgOrder
                  << " E/t=" << (energyEstimator / tHop) << "\n";

        // Keep a small offset from the current estimate to avoid singular normalization
        // in the auxiliary histogram formula when e(k=0)-muUsed is exactly zero.
        if (!useFixedMu) {
            const double targetMu = energyEstimator - muOffset * tHop;
            const double delta = muCurrent - muPrev;
            const double muNext =
                muCurrent + muAlpha * (targetMu - muCurrent) + muMomentum * delta;
            muPrev = muCurrent;
            muCurrent = muNext;
        }
    }
    out.close();
    outGreen.close();
    std::cout << "Saved CSV: " << outPath << "\n";
    std::cout << "Saved CSV: " << outGreenPath << "\n";
    return 0;
}
