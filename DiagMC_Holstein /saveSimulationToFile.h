#ifndef SAVESIMULATIONTOFILE_H
#define SAVESIMULATIONTOFILE_H

#include <vector>
#include <string>

// declaration of saveSimulationToFile() defined in saveSimulationToFile.cpp
// writes all simulation parameters and outputs (histogram, order distribution, estimator)
// to a .csv file inside ./results/
template <typename T>
void saveSimulationToFile(
    const std::string& baseFilename,      // base name for output file
    int N,                                // number of Monte Carlo iterations
    int bins,                             // number of histogram bins
    double g,                             // coupling constant
    double mu,                            // chemical potential
    double w0,                            // phonon frequency
    double k,                             // electron momentum
    double t,                             // hopping amplitude
    double up_lim,                        // upper cutoff for tau
    double energyEstimator,               // estimated energy at the end of run
    const std::vector<T>& histogramData,  // sampled G(tau) histogram
    const std::vector<int>& orderDist,    // diagram order distribution
    const std::vector<double>& GreenEstimator // improved estimator for G(tau)
);

#endif
