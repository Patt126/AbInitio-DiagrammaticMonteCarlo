#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h> // for stat(), used to check file existence

// check if a file already exists on disk
inline bool fileExists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

// save results of a single DMC run to a .csv file
// baseFilename       = name of the output file (without path or extension)
// N, bins, ...       = simulation parameters
// energyEstimator    = final energy estimator (E - mu) + mu
// histogramData      = estimated G(tau) from simple bin counting
// orderDist          = histogram of diagram orders
// GreenEstimator     = improved / normalized G(tau) estimator
//
// Output is stored in ./results/ with filename baseFilename.csv
// If the file already exists, it appends (1), (2), ... to the name.
template<typename T>
void saveSimulationToFile(
    const std::string& baseFilename,
    int N,
    int bins,
    double g,
    double mu,
    double w0,
    double k,
    double t,
    double up_lim,
    double energyEstimator,
    const std::vector<T>& histogramData,
    const std::vector<int>& orderDist,
    const std::vector<double>& GreenEstimator)
{
    std::string finalFilename;
    std::string path = "./results/";
    std::string extension = ".csv";
    int counter = 1;

    // generate unique filename
    if (!fileExists(path + baseFilename + extension)) {
        finalFilename = path + baseFilename + extension;
    } else {
        while (true) {
            std::stringstream ss;
            ss << path << baseFilename << "(" << counter << ")" << extension;
            if (!fileExists(ss.str())) {
                finalFilename = ss.str();
                break;
            }
            counter++;
        }
    }

    // open file for writing
    std::ofstream outfile(finalFilename);
    if (!outfile.is_open()) {
        std::cerr << "Error: cannot open file for writing: "
                  << finalFilename << std::endl;
        return;
    }

    // write simulation parameters
    outfile << "=== Simulation Parameters ===\n";
    outfile << "N = " << N << "\n";
    outfile << "bins = " << bins << "\n";
    outfile << "g = " << g << "\n";
    outfile << "mu = " << mu << "\n";
    outfile << "w0 = " << w0 << "\n";
    outfile << "k = " << k << "\n";
    outfile << "t = " << t << "\n";
    outfile << "up_lim = " << up_lim << "\n";
    outfile << "energyEstimator = " << energyEstimator << "\n";

    outfile << "\n=== Results ===\n";

    // histogram of G(tau)
    outfile << "histogram_data";
    for (const auto& value : histogramData) {
        outfile << "," << value;
    }
    outfile << "\n";

    // distribution of diagram orders
    outfile << "order_data";
    for (const auto& value : orderDist) {
        outfile << "," << value;
    }
    outfile << "\n";

    // improved / normalized Green’s function estimator
    outfile << "Green_estimator";
    for (const auto& value : GreenEstimator) {
        outfile << "," << value;
    }
    outfile << "\n";

    outfile.close();
}

// explicit template instantiation for the version used in the code (T = double)
template void saveSimulationToFile(
    const std::string& baseFilename,
    int N,
    int bins,
    double g,
    double mu,
    double w0,
    double k,
    double t,
    double up_lim,
    double energyEstimator,
    const std::vector<double>& histogramData,
    const std::vector<int>& orderDist,
    const std::vector<double>& GreenEstimator);
