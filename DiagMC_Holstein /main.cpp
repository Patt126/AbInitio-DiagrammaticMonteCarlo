#include <iostream>
#include <vector>
#include <chrono>
#include "feynmanDiagram.h"
#include "saveSimulationToFile.h"

// DMC is defined in DMC.cpp
// Return value: pair<avgOrder, energyEstimator>
std::pair<double,double> DMC(
    double tau,
    const double k,
    const double t,
    const double mu,
    const double g,
    const double w0,
    const double up_lim,
    const int N,
    const int bins,
    std::vector<double>& histogram,
    std::vector<int>& orderDist,
    std::vector<double>& Green_estimator,
    bool tauChange /* default true in your current usage */
);

int main() {
    using std::cout;
    using std::endl;

    // Simulation parameters
    const int    N       = 1e7;     // Monte Carlo steps
    const int    bins    = 100;     // tau-bins for G(tau)
    const double g       = 0.5;     // e-ph coupling
    const double mu      = -2.2;    // chemical potential
    const double w0      = 0.5;     // phonon frequency
    const double t_hop   = 1.0;     // electron hopping t
    const double up_lim  = 50.0;    // max tau allowed
    const double k       = 0.0;     // electron momentum
    const double tau0    = 10.0;    // initial tau for the walk

    //  Storage for observables 
    std::vector<double> histogram;         // G(tau) histogram / visit counts
    std::vector<int>    orderDist;         // distribution of diagram order
    std::vector<double> Green_estimator;   // improved estimator per tau-bin

    cout << "\nDiagrammatic Monte Carlo for Holstein polaron\n";
    cout << "Continuous-time sampling with diagram updates\n\n";

    cout << "Parameters:\n";
    cout << "  N steps        = " << N      << "\n";
    cout << "  bins           = " << bins   << "\n";
    cout << "  g              = " << g      << "\n";
    cout << "  mu             = " << mu     << "\n";
    cout << "  w0             = " << w0     << "\n";
    cout << "  t (hopping)    = " << t_hop  << "\n";
    cout << "  k              = " << k      << "\n";
    cout << "  tau0           = " << tau0   << "\n";
    cout << "  tau upper lim  = " << up_lim << "\n\n";

    // Timestamp start
    auto t_start = std::chrono::high_resolution_clock::now();

    //  Run MC 
    // tauChange=true means allow global tau-change updates 
    auto [avgOrder, energyEstimator] = DMC(
        tau0,
        k,
        t_hop,
        mu,
        g,
        w0,
        up_lim,
        N,
        bins,
        histogram,
        orderDist,
        Green_estimator,
        true
    );

    // Timestamp end
    auto t_end   = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;

    cout << "Run complete.\n";
    cout << "  <diagram order>        = " << avgOrder         << "\n";
    cout << "  energy estimator (E-μ) = " << energyEstimator  << "\n";
    cout << "  elapsed time [s]       = " << elapsed.count()  << "\n\n";

    // ---- Save results ----
    // saveSimulationToFile(name, ...)
    // 
    saveSimulationToFile(
        "myResult",
        N,
        bins,
        g,
        mu,
        w0,
        k,
        t_hop,
        up_lim,
        energyEstimator,
        histogram,
        orderDist,
        Green_estimator
    );

    cout << "Results written to ./results/\n";

    // Optional sweep over tau for stability check.
    // Disabled by default because it re-runs DMC many times and costs runtime.
    /*
    for (double tauProbe = 1.0; tauProbe < 100.0; tauProbe += 5.0) {
        std::vector<double> h2;
        std::vector<int>    o2;
        std::vector<double> G2;

        auto [avgOrderProbe, estProbe] = DMC(
            tauProbe,
            k,
            t_hop,
            mu,
            g,
            w0,
            tauProbe,   // use tauProbe also as up_lim here (your commented code)
            N,
            bins,
            h2,
            o2,
            G2,
            false       // tauChange = false to freeze tau during probe
        );

        cout << tauProbe
             << "  energy=" << estProbe
             << "  <order>=" << avgOrderProbe
             << endl;
    }
    */

    return 0;
}
