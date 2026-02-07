#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <random>
#include <chrono>
#include <functional>
#include <algorithm>
#include <complex>
#include "./saveSimulationToFile.h"
#include "./feynmanDiagram.h"
using namespace std;

// rng setup. mt19937 seeded with current time
unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937 rnd (seed);
std::uniform_real_distribution<double> uniform(0.0,1.0);

// Diagrammatic Monte Carlo sampler for the Holstein polaron.
// tau         = initial total propagation time of the diagram
// k, t, mu... = physical parameters
// up_lim      = max allowed propagation time (tau upper cutoff)
// N           = number of Monte Carlo steps
// bins        = number of bins in tau histogram
// histogram   = output estimator for G(tau) based on simple bin-counting
// orderDist   = histogram of number of phonon lines in the diagram
// exact_estimator = reweighted estimator for G(tau) with reduced bin-size bias
// tauChange   = if true, allow global tau updates (stretch / change final tau)
// return      = <avg expansion order>, energy estimator
pair<double, double> DMC(
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
    vector<double>& histogram,
    vector<int>& orderDist,
    vector<double>& exact_estimator,
    bool tauChange /* = true */
)
{
    const int bands = std::max(1, numElectronBands);
    const int modes = std::max(1, numPhononModes);
    const int qStates = std::max(1, numMomentumPoints); // q is sampled on a normalized grid measure, so no extra N_q factor in M-H proposal ratio
    const double baseProposalMultiplicity =
        static_cast<double>(modes) * bands * bands;

    // when sampling the phonon line lenght we use information from their expoential trend to make more efficient proposal
    // The slope is considered based on the current phonon frequency, this however can easily vanish, creating numerical problems
    auto safeSlopeFromMode = [&wFunc](int mode){
        const double w = wFunc(mode);
        const double eps = 1e-12;
        return 1.0 / ((w > eps) ? w : eps); 
    };

    // reference mode used in global observables that still use a single scale
    const double w0 = wFunc(phononMode);

    // diagram object: stores the current configuration (linked list of vertices)
    // handles insert/remove phonon line, tau rescaling, etc.
    Diag diagram(tau, t, k, mu, gFunc, wFunc, eFunc, electronBand, phononMode);

    // clear + resize output containers
    histogram.clear();
    histogram.resize(bins);

    orderDist.clear();
    orderDist.resize(20); // will resize dynamically later if order > 19

    exact_estimator.clear();
    exact_estimator.resize(bins);

    // accumulators for energy estimator etc.
    double length = 0;
    double extimator = 0;
    double extimatorSquare = 0;
    int Nadd = 0;      // how many proposed "add phonon line" moves
    int N0 = 0;        // how many samples where diagram has order 0
    double avgOrder = 0;
    double bin_width = (double) up_lim / bins;

    // current total weight of the diagram (kept for debugging; not used now)
    double currentWeight = diagram.getFullWeight();

    // helper lambdas for proposal distributions and acceptance factors

    // choose which local update to attempt (uniform between allowed updates)
    auto p_uni = [](){return (double)1/5;}; // not used explicitly anymore

    // for sampling the end time of a phonon line:
    // we bias tauEnd - tauIn with an exponential distribution ~ exp(-(tEnd-tStart)/a)
    // a = 1/w0. exp_sample_norm() returns normalization constant for that truncated exponential
    auto exp_sample_norm = [](double tStart, double tau, double slope){
        return (double)1/ (slope*(1 - exp(-(tau-tStart)/slope)));
    };

    // invert the CDF of that truncated exponential to actually draw tauEnd
    auto exp_sample = [](double r,double tStart, double slope, double C){
        return -slope*log(1 - r / (slope * C)) + tStart;
    };

    // probability density of proposing tauEnd given tauIn under that exponential
    auto p_exp = [](double tStart,double tEnd, double slope, double C){
        return exp(-(tEnd - tStart)/slope)*C;
    };

    // I0 accumulates total time spent in order-0 diagrams
    // used later for normalization of G(tau) (see chapter 3 thesis sez Green Function Histogram)
    double I0 = 0;

    // normGreenEst[b] is the measure for the improved estimator at bin b
    // intervalEstimator is the "window" half-width around each bin center where we accept contributions
    vector<double> normGreenEst;
    double intervalEstimator = 4*bin_width;
    normGreenEst.resize(bins);

    // precompute normalization factor for each bin's exact_estimator entry
    // handle bins that are near tau=0 or near tau=up_lim (truncated window)
    double center = bin_width/2;
    for (int b = 0; b < bins ; b++){    
        if(center - intervalEstimator < 0 ){
            normGreenEst[b] = center + intervalEstimator;
        }
        else if(center + intervalEstimator > up_lim){
            normGreenEst[b] = up_lim - center + intervalEstimator;
        }
        else{
            normGreenEst[b] = 2*intervalEstimator;
        }
        center += bin_width;
    }

    // global variables updated during the Markov chain
    double tauCurrent = tau;        // current diagram total time
    double tauIn , tauEnd;          // proposed phonon line start/end
    double sumInterval = diagram.getEnergy(diagram.getK()) * diagram.getTau();
    int NextEn = 0;                 // number of samples contributing to energy estimator
    const int updates = tauChange ? 4 : 2; // if tauChange=false, only use add/remove moves

    // blocking analysis data to get an error bar on energy estimator
    // blockMeans[r] will store the average energy estimator for block size 10^(r+1)
    vector<vector<double>> blockMeans = {};
    vector<double> currentBlockAccumulation = {};
    vector<int> blockSizes = {};

    for (int i = 1; i < log10(N); i++) {
        blockMeans.push_back({}); 
        currentBlockAccumulation.push_back(0);
        blockSizes.push_back(0);
    }

    // main Monte Carlo loop
    for (int i = 0; i < N; i++){

        // choose which update to attempt:
        // 1: add phonon line
        // 2: remove phonon line
        // 3: stretch whole diagram to a new tau (rescale internal times)
        // 4: only change final tau (shift tail) without rescaling
        std::uniform_int_distribution<int> distribution(1,updates);
        int choice = distribution(rnd);

        switch (choice)
        { 
        // add phonon line
        case 1: {

            Nadd++;

            // choose start time tauIn uniformly in [0, tauCurrent]
            std::uniform_real_distribution<double> uniform_1(0,tauCurrent);
            tauIn = uniform_1(rnd); 

            // choose end time tauEnd > tauIn with biased exponential proposal
            double r = uniform(rnd);
            // prevIn / nextEnd tell us where to insert the two new vertices
            Vertex* prevIn = nullptr;
            Vertex* nextEnd = nullptr;

            std::uniform_int_distribution<int> bandDistribution(0, bands - 1);
            std::uniform_int_distribution<int> modeDistribution(0, modes - 1);
            std::uniform_int_distribution<int> qDistribution(0, qStates - 1);
            int inBandOut = bandDistribution(rnd);
            int modeForLine = modeDistribution(rnd);
            int qIndex = qDistribution(rnd);
            double q = -M_PI + (2.0 * M_PI) * (qIndex + 0.5) / qStates;
            double a = safeSlopeFromMode(modeForLine);
            double C = exp_sample_norm(tauIn, tauCurrent, a); 
            tauEnd = exp_sample(r, tauIn, a, C);
            const bool endAtTail = (tauEnd > diagram.getTailTime());
            int endBandOut = endAtTail ? diagram.getTailBand() : bandDistribution(rnd);
            const double proposalMultiplicityThis =
                baseProposalMultiplicity / (endAtTail ? bands : 1.0);

            if(tauEnd<tauIn) {
                // sanity check, shouldn't trigger
                cout<<tauIn<<" "<<tauEnd<<endl;
            }

            length += (tauEnd - tauIn);

            // getAddWeight returns only W(new)/W(old), built constructively
            // from propagators and couplings on the affected interval
            double diagW = diagram.getAddWeight(
                tauIn, tauEnd, q, prevIn, nextEnd, inBandOut, endBandOut, modeForLine);

            // acceptance prob A = min(1, ...)
            // factors:
            // - reverse remove choice: 1/(n+1)
            // - forward add samples: mode, in-band, q-grid point, tau_i, tau_f
            //   and end-band only if the annihilation is not at tail
            double A = min(1.0,
                diagW/(diagram.getPhLineNum()+1)
                * proposalMultiplicityThis
                * tauCurrent / p_exp(tauIn,tauEnd,a,C)
            );

            int inBandIn = prevIn ? prevIn->bandOut : diagram.getElectronBand();
            if(uniform(rnd) < A){
                // actually insert the line in the linked list object
                diagram.insertPhononLine(
                    prevIn, nextEnd, tauIn, tauEnd, q,
                    inBandIn, inBandOut, -1, endBandOut, modeForLine);

                // for energy estimator: we accumulate log of acceptance weights
                // note: sumInterval used later to build energy estimator
                if(!tauChange || tauChange){
                    double kInAtEmission = prevIn ? prevIn->kOut : diagram.getK();
                    bool noVerticesBetween = nextEnd ? (nextEnd->prev == prevIn) : true;
                    int endBandIn = noVerticesBetween
                        ? inBandOut
                        : nextEnd->prev->bandOut;
                    double kBeforeAnnihilation = nextEnd ? (nextEnd->kIn + q) : (diagram.getK() + q);
                    double gCreate = diagram.getCoupling(
                        kInAtEmission, q, inBandIn, inBandOut, modeForLine);
                    double gAnnih = diagram.getCoupling(
                        kBeforeAnnihilation, -q, endBandIn, endBandOut, modeForLine);
                    double gLine = gCreate * gAnnih;
                    if(gLine != 0.0){
                        sumInterval += -log(diagW/gLine);
                    }
                }
            }
            break;
        }

        // remove phonon line
        case 2: {
            if(diagram.getPhLineNum() > 0 ){

                // choose a random phonon line to remove
                std::uniform_int_distribution<int> LineDist(1,diagram.getPhLineNum());
                int position = LineDist(rnd);

                Vertex* inVertex = nullptr;

                // getRemoveWeight returns ratio W(new)/W(old) for removing that line
                double diagWeight= diagram.getRemoveWeight(position, inVertex);
                if(!inVertex){
                    // if we failed to identify the line for some reason, skip
                    break;
                }

                tauIn  = inVertex -> time;
                tauEnd = inVertex -> link -> time;

                double a = safeSlopeFromMode(inVertex->mode);
                double C = exp_sample_norm(tauIn, tauCurrent, a);
                const bool removedEndsAtTail = (inVertex->link->next == nullptr);
                const double reverseAddProposalMultiplicity =
                    baseProposalMultiplicity / (removedEndsAtTail ? bands : 1.0);

                // reverse of the add-phonon acceptance formula
                double A = min(1.0,
                    diagram.getPhLineNum()/(reverseAddProposalMultiplicity * tauCurrent)
                    * p_exp(tauIn,tauEnd,a,C)*diagWeight
                );

                if(uniform(rnd) < A && inVertex ){
                    double gCreate = diagram.getCoupling(
                        inVertex->kIn, inVertex->q,
                        inVertex->bandIn, inVertex->bandOut, inVertex->mode);
                    double gAnnih = diagram.getCoupling(
                        inVertex->link->kIn, inVertex->link->q,
                        inVertex->link->bandIn, inVertex->link->bandOut, inVertex->link->mode);
                    double gLine = gCreate * gAnnih;
                    diagram.removePhononLine(inVertex);

                    if(!tauChange || tauChange){   
                        if(gLine != 0.0){
                            sumInterval += -log(diagWeight*gLine);
                        }
                    }
                }
            }
            break;
        }

        // stretch diagram
        case 3:{
            // propose a completely new tauNew uniformly in [0, up_lim]
            // then rescale all internal times so that everything fits in [0,tauNew]
            // this is a global move that changes the diagram's time extent
            std::uniform_real_distribution<double> uniformTau(0,up_lim);
            double tauNew = uniformTau(rnd);

            // getStretchWeight gives ratio W(new)/W(old) * proposal factors for rescaling
            double diagWeight = diagram.getStretchWeight(tauNew);

            // acceptance includes factor (tauNew/tauCurrent)^(order)
            // coming from the uniform reparametrization of vertex times
            double A = min(1.0,
                diagWeight * pow(tauNew/tauCurrent,(double) diagram.getOrder())
            );

            if (uniform(rnd) <= A){
                diagram.stretchDiagram(tauNew);

                tauCurrent = tauNew;
                sumInterval += -log(diagWeight);
            }
            break;
        }

        // change final tau
        case 4 :{
            // here we only change the final propagation time (diagram tail)
            // internal vertex times remain the same
            std::uniform_real_distribution<double> uniformTau(diagram.getTailTime(),up_lim);
            double tauNew = uniformTau(rnd);

            double diagWeight = diagram.getTauWeight(tauNew);

            double A = min(1.0, diagWeight);

            if (uniform(rnd) <= A){
                diagram.setTau(tauNew);

                tauCurrent = tauNew;
                sumInterval += -log(diagWeight);
            }
            break;
        }

        } // end switch on update type
        
        // basic consistency check: electron momentum conservation along the line
        if(!diagram.testConservation()){
            diagram.printDiagram();
        }

        // record time spent in diagrams of order 0
        // this is used for normalization of G(tau) later
        if(diagram.getPhLineNum() == 0){
            I0 += tauCurrent;
            N0++;
        }

        // accumulate estimators only if tauChange==true
        // (if tauChange=false we are doing a fixed-tau probe run)
        if(tauChange){
            // naive histogram estimator for G(tau):
            // we just count how often we visit a configuration with total tau in each bin
            int bin = trunc( tauCurrent/ bin_width);
            histogram[bin] = histogram[bin] + 1;

            // improved estimator:
            // instead of only counting at tauCurrent, we also contribute to any bin whose
            // center is close to the current tau (within intervalEstimator)
            // weight involves getStretchWeight(center) etc.
            double center = bin_width/2;
            for (int b = 0; b < bins ; b++){
                if( abs(diagram.getTau() - center)<= intervalEstimator ){
                    exact_estimator[b] += diagram.getStretchWeight(center)
                                        * pow(center/diagram.getTau(), (double) diagram.getOrder())
                                        / (normGreenEst[b]);
                }
                center += bin_width;
            }
        }
        
        // make sure orderDist is large enough
        if(orderDist.size() <= diagram.getPhLineNum()){
            orderDist.resize(diagram.getPhLineNum() + 1);
        }
        orderDist[diagram.getPhLineNum()] ++;

        // accumulate average expansion order
        avgOrder += diagram.getPhLineNum();

        // energy estimator:
        // we look at (sumInterval - order) / tau.
        // for large tau this should approach (E - mu)
        if(diagram.getTau() > 2/w0){
            double currentFluctuation =
                (sumInterval - diagram.getOrder()) / diagram.getTau();

            extimator       += currentFluctuation;
            extimatorSquare += pow(currentFluctuation, 2);
            NextEn++;

            // blocking analysis:
            // accumulate block averages at different block sizes (10, 100, 1000, ...)
            for (int r = 0; r < blockMeans.size(); r++) {
                currentBlockAccumulation[r] += currentFluctuation;
                blockSizes[r]++;

                int targetBlockSize = pow(10, r + 1);
                if (blockSizes[r] == targetBlockSize) {

                    // push back block mean shifted by +mu (so it's closer to physical energy)
                    blockMeans[r].push_back(
                        currentBlockAccumulation[r] / blockSizes[r] + mu
                    );

                    // reset for next block
                    currentBlockAccumulation[r] = 0;
                    blockSizes[r] = 0;
                }
            }
        }

    } // end Monte Carlo loop


    if(tauChange){
        // normalization for G(tau)
        // we estimate an absolute prefactor c
        // c ~ (1 - exp(-E(k)*up_lim)) / (N0 * E(k)) * N
        // then divide histogram counts by bin_width and total steps
        double e0 = diagram.getEnergy(diagram.getK());
        double c = (double)(1 - exp(-e0*up_lim))
                    /(N0*e0)*N; 
        cout<<"Normalization= "<<c<<endl;

        for(int bin = 0; bin < bins; bin++ ){
            histogram[bin]      = (double) histogram[bin]*c/(bin_width*N);
            exact_estimator[bin]= c * exact_estimator[bin]/N;
        }

        cout<<"avg proposed length: "<< (double) length / Nadd <<endl;
        cout<<"avarage order: "<<(double) avgOrder / N<<endl;
    }

    // now build the final energy estimator and its naive error bar
    // extimator ~ < (sumInterval - order)/tau > + mu
    extimator = (double) extimator / NextEn;

    cout<< "naive STD extimator = "
        << sqrt((extimatorSquare - NextEn*pow(extimator,2))/(NextEn - 1))
           / sqrt(NextEn)
        <<endl;

    extimator += mu;

    // blocking analysis:
    // for each block size 10^(r+1), compute std / sqrt(nBlocks)
    // also print Rx = variance * blockSize , a rough autocorrelation measure
    cout << "--- Blocking Analysis ---" << endl;
    for(int r = 0; r < blockMeans.size(); r++){
        cout<<"blockSize = "<< pow(10,r + 1);

        double variance = 0;
        for(int c = 0; c < blockMeans[r].size(); c++){
            variance += pow(blockMeans[r][c] - extimator,2);  
        }

        variance = variance/(blockMeans[r].size() - 1);

        cout<< " std = "<< sqrt(variance)/sqrt(blockMeans[r].size())
            << " Rx = "<< variance * pow(10,r+1)
            << endl;
    }

    // return average order and final energy estimator
    return {(double) avgOrder / N, extimator};
}

// DMCWithSign:
// sample diagrams with |g| but reweight Green-function estimators with sign(g)
pair<double, double> DMCWithSign(
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
    vector<double>& histogram,
    vector<int>& orderDist,
    vector<double>& exact_estimator,
    vector<double>& exact_estimator_signed,
    vector<double>& exact_estimator_err,
    vector<double>& exact_estimator_signed_err,
    double& avgSign,
    double& avgPhaseMag,
    bool tauChange /* = true */
)
{
    const int bands = std::max(1, numElectronBands);
    const int modes = std::max(1, numPhononModes);
    const int qStates = std::max(1, numMomentumPoints);
    const double baseProposalMultiplicity =
        static_cast<double>(modes) * bands * bands;

    auto safeSlopeFromMode = [&wFunc](int mode){
        const double w = wFunc(mode);
        const double eps = 1e-12;
        return 1.0 / ((w > eps) ? w : eps); 
    };

    const double w0 = wFunc(phononMode);

    Diag diagram(tau, t, k, mu, gAbsFunc, wFunc, eFunc, electronBand, phononMode);

    histogram.clear();
    histogram.resize(bins);

    orderDist.clear();
    orderDist.resize(20);

    exact_estimator.clear();
    exact_estimator.resize(bins);
    exact_estimator_signed.clear();
    exact_estimator_signed.resize(bins);
    exact_estimator_err.clear();
    exact_estimator_err.resize(bins);
    exact_estimator_signed_err.clear();
    exact_estimator_signed_err.resize(bins);
    vector<std::complex<double>> exact_phase;
    exact_phase.resize(bins);
    vector<double> exact_estimator_sq;
    exact_estimator_sq.resize(bins);
    vector<double> exact_phase_real_sum;
    exact_phase_real_sum.resize(bins);
    vector<double> exact_phase_real_sq;
    exact_phase_real_sq.resize(bins);

    double length = 0;
    double extimator = 0;
    double extimatorSquare = 0;
    int Nadd = 0;
    int N0 = 0;
    double avgOrder = 0;
    double bin_width = (double) up_lim / bins;

    double currentWeight = diagram.getFullWeight();

    auto exp_sample_norm = [](double tStart, double tau, double slope){
        return (double)1/ (slope*(1 - exp(-(tau-tStart)/slope)));
    };

    auto exp_sample = [](double r,double tStart, double slope, double C){
        return -slope*log(1 - r / (slope * C)) + tStart;
    };

    auto p_exp = [](double tStart,double tEnd, double slope, double C){
        return exp(-(tEnd - tStart)/slope)*C;
    };

    double I0 = 0;

    vector<double> normGreenEst;
    double intervalEstimator = 4*bin_width;
    normGreenEst.resize(bins);

    double center = bin_width/2;
    for (int b = 0; b < bins ; b++){    
        if(center - intervalEstimator < 0 ){
            normGreenEst[b] = center + intervalEstimator;
        }
        else if(center + intervalEstimator > up_lim){
            normGreenEst[b] = up_lim - center + intervalEstimator;
        }
        else{
            normGreenEst[b] = 2*intervalEstimator;
        }
        center += bin_width;
    }

    double tauCurrent = tau;
    double tauIn , tauEnd;
    double sumInterval = diagram.getEnergy(diagram.getK()) * diagram.getTau();
    int NextEn = 0;
    const int updates = tauChange ? 4 : 2;

    vector<vector<double>> blockMeans = {};
    vector<double> currentBlockAccumulation = {};
    vector<int> blockSizes = {};

    for (int i = 1; i < log10(N); i++) {
        blockMeans.push_back({}); 
        currentBlockAccumulation.push_back(0);
        blockSizes.push_back(0);
    }

    std::complex<double> phaseSum(0.0, 0.0);

    for (int i = 0; i < N; i++){

        std::uniform_int_distribution<int> distribution(1,updates);
        int choice = distribution(rnd);

        switch (choice)
        { 
        case 1: {

            Nadd++;

            std::uniform_real_distribution<double> uniform_1(0,tauCurrent);
            tauIn = uniform_1(rnd); 

            double r = uniform(rnd);
            Vertex* prevIn = nullptr;
            Vertex* nextEnd = nullptr;

            std::uniform_int_distribution<int> bandDistribution(0, bands - 1);
            std::uniform_int_distribution<int> modeDistribution(0, modes - 1);
            std::uniform_int_distribution<int> qDistribution(0, qStates - 1);
            int inBandOut = bandDistribution(rnd);
            int modeForLine = modeDistribution(rnd);
            int qIndex = qDistribution(rnd);
            double q = -M_PI + (2.0 * M_PI) * (qIndex + 0.5) / qStates;
            double a = safeSlopeFromMode(modeForLine);
            double C = exp_sample_norm(tauIn, tauCurrent, a); 
            tauEnd = exp_sample(r, tauIn, a, C);
            const bool endAtTail = (tauEnd > diagram.getTailTime());
            int endBandOut = endAtTail ? diagram.getTailBand() : bandDistribution(rnd);
            const double proposalMultiplicityThis =
                baseProposalMultiplicity / (endAtTail ? bands : 1.0);

            if(tauEnd<tauIn) {
                cout<<tauIn<<" "<<tauEnd<<endl;
            }

            length += (tauEnd - tauIn);

            double diagW = diagram.getAddWeight(
                tauIn, tauEnd, q, prevIn, nextEnd, inBandOut, endBandOut, modeForLine);

            double A = min(1.0,
                diagW/(diagram.getPhLineNum()+1)
                * proposalMultiplicityThis
                * tauCurrent / p_exp(tauIn,tauEnd,a,C)
            );

            int inBandIn = prevIn ? prevIn->bandOut : diagram.getElectronBand();
            if(uniform(rnd) < A){
                diagram.insertPhononLine(
                    prevIn, nextEnd, tauIn, tauEnd, q,
                    inBandIn, inBandOut, -1, endBandOut, modeForLine);

                if(!tauChange || tauChange){
                    double kInAtEmission = prevIn ? prevIn->kOut : diagram.getK();
                    bool noVerticesBetween = nextEnd ? (nextEnd->prev == prevIn) : true;
                    int endBandIn = noVerticesBetween
                        ? inBandOut
                        : nextEnd->prev->bandOut;
                    double kBeforeAnnihilation = nextEnd ? (nextEnd->kIn + q) : (diagram.getK() + q);
                    double gCreate = diagram.getCoupling(
                        kInAtEmission, q, inBandIn, inBandOut, modeForLine);
                    double gAnnih = diagram.getCoupling(
                        kBeforeAnnihilation, -q, endBandIn, endBandOut, modeForLine);
                    double gLine = gCreate * gAnnih;
                    if(gLine != 0.0){
                        sumInterval += -log(diagW/gLine);
                    }
                }
            }
            break;
        }

        case 2: {
            if(diagram.getPhLineNum() > 0 ){

                std::uniform_int_distribution<int> LineDist(1,diagram.getPhLineNum());
                int position = LineDist(rnd);

                Vertex* inVertex = nullptr;

                double diagWeight= diagram.getRemoveWeight(position, inVertex);
                if(!inVertex){
                    break;
                }

                tauIn  = inVertex -> time;
                tauEnd = inVertex -> link -> time;

                double a = safeSlopeFromMode(inVertex->mode);
                double C = exp_sample_norm(tauIn, tauCurrent, a);
                const bool removedEndsAtTail = (inVertex->link->next == nullptr);
                const double reverseAddProposalMultiplicity =
                    baseProposalMultiplicity / (removedEndsAtTail ? bands : 1.0);

                double A = min(1.0,
                    diagram.getPhLineNum()/(reverseAddProposalMultiplicity * tauCurrent)
                    * p_exp(tauIn,tauEnd,a,C)*diagWeight
                );

                if(uniform(rnd) < A && inVertex ){
                    double gCreate = diagram.getCoupling(
                        inVertex->kIn, inVertex->q,
                        inVertex->bandIn, inVertex->bandOut, inVertex->mode);
                    double gAnnih = diagram.getCoupling(
                        inVertex->link->kIn, inVertex->link->q,
                        inVertex->link->bandIn, inVertex->link->bandOut, inVertex->link->mode);
                    double gLine = gCreate * gAnnih;
                    diagram.removePhononLine(inVertex);

                    if(!tauChange || tauChange){   
                        if(gLine != 0.0){
                            sumInterval += -log(diagWeight*gLine);
                        }
                    }
                }
            }
            break;
        }

        case 3:{
            std::uniform_real_distribution<double> uniformTau(0,up_lim);
            double tauNew = uniformTau(rnd);

            double diagWeight = diagram.getStretchWeight(tauNew);

            double A = min(1.0,
                diagWeight * pow(tauNew/tauCurrent,(double) diagram.getOrder())
            );

            if (uniform(rnd) <= A){
                diagram.stretchDiagram(tauNew);

                tauCurrent = tauNew;
                sumInterval += -log(diagWeight);
            }
            break;
        }

        case 4 :{
            std::uniform_real_distribution<double> uniformTau(diagram.getTailTime(),up_lim);
            double tauNew = uniformTau(rnd);

            double diagWeight = diagram.getTauWeight(tauNew);

            double A = min(1.0, diagWeight);

            if (uniform(rnd) <= A){
                diagram.setTau(tauNew);

                tauCurrent = tauNew;
                sumInterval += -log(diagWeight);
            }
            break;
        }

        } 
        
        if(!diagram.testConservation()){
            diagram.printDiagram();
        }

        if(diagram.getPhLineNum() == 0){
            I0 += tauCurrent;
            N0++;
        }

        std::complex<double> phaseFactor(1.0, 0.0);
        if(tauChange){
            phaseFactor = diagram.getCouplingPhase(gPhaseFunc);
            phaseSum += phaseFactor;

            int bin = trunc( tauCurrent/ bin_width);
            histogram[bin] = histogram[bin] + 1;

            double center = bin_width/2;
            for (int b = 0; b < bins ; b++){
                if( abs(diagram.getTau() - center)<= intervalEstimator ){
                    const double contrib = diagram.getStretchWeight(center)
                                         * pow(center/diagram.getTau(), (double) diagram.getOrder())
                                         / (normGreenEst[b]);
                    exact_estimator[b] += contrib;
                    exact_estimator_sq[b] += contrib * contrib;
                    exact_phase[b] += phaseFactor * contrib;
                    const double realContrib = std::real(phaseFactor) * contrib;
                    exact_phase_real_sum[b] += realContrib;
                    exact_phase_real_sq[b] += realContrib * realContrib;
                }
                center += bin_width;
            }
        }
        
        if(orderDist.size() <= diagram.getPhLineNum()){
            orderDist.resize(diagram.getPhLineNum() + 1);
        }
        orderDist[diagram.getPhLineNum()] ++;

        avgOrder += diagram.getPhLineNum();

        if(diagram.getTau() > 2/w0){
            double currentFluctuation =
                (sumInterval - diagram.getOrder()) / diagram.getTau();

            extimator       += currentFluctuation;
            extimatorSquare += pow(currentFluctuation, 2);
            NextEn++;

            for (int r = 0; r < blockMeans.size(); r++) {
                currentBlockAccumulation[r] += currentFluctuation;
                blockSizes[r]++;

                int targetBlockSize = pow(10, r + 1);
                if (blockSizes[r] == targetBlockSize) {

                    blockMeans[r].push_back(
                        currentBlockAccumulation[r] / blockSizes[r] + mu
                    );

                    currentBlockAccumulation[r] = 0;
                    blockSizes[r] = 0;
                }
            }
        }

    }

    if(tauChange){
        double e0 = diagram.getEnergy(diagram.getK());
        double c = (double)(1 - exp(-e0*up_lim))
                    /(N0*e0)*N; 
        cout<<"Normalization= "<<c<<endl;

        const double invN = (N > 0) ? (1.0 / N) : 0.0;
        const double phaseMeanMag = std::abs(phaseSum) * invN;
        const double phaseNorm = (phaseMeanMag > 1e-12) ? phaseMeanMag : 1e-12;
        for(int bin = 0; bin < bins; bin++ ){
            histogram[bin]      = (double) histogram[bin]*c/(bin_width*N);
            const double meanAbs = exact_estimator[bin] * invN;
            exact_estimator[bin]= c * meanAbs;
            const double varAbs = std::max(0.0, exact_estimator_sq[bin] * invN - meanAbs * meanAbs);
            exact_estimator_err[bin] = c * std::sqrt(varAbs * invN);

            if(std::abs(phaseSum) > 0.0){
                exact_estimator_signed[bin]= (c * exact_phase[bin] / phaseSum).real();
                const double meanReal = exact_phase_real_sum[bin] * invN;
                const double varReal = std::max(0.0, exact_phase_real_sq[bin] * invN - meanReal * meanReal);
                exact_estimator_signed_err[bin] = c * std::sqrt(varReal * invN) / phaseNorm;
            } else {
                exact_estimator_signed[bin]= 0.0;
                exact_estimator_signed_err[bin]= 0.0;
            }
        }

        cout<<"avg proposed length: "<< (double) length / Nadd <<endl;
        cout<<"avarage order: "<<(double) avgOrder / N<<endl;
    }

    avgSign = (tauChange && N > 0) ? (phaseSum.real() / N) : 0.0;
    avgPhaseMag = (tauChange && N > 0) ? (std::abs(phaseSum) / N) : 0.0;

    extimator = (double) extimator / NextEn;

    cout<< "naive STD extimator = "
        << sqrt((extimatorSquare - NextEn*pow(extimator,2))/(NextEn - 1))
           / sqrt(NextEn)
        <<endl;

    extimator += mu;

    cout << "--- Blocking Analysis ---" << endl;
    for(int r = 0; r < blockMeans.size(); r++){
        cout<<"blockSize = "<< pow(10,r + 1);

        double variance = 0;
        for(int c = 0; c < blockMeans[r].size(); c++){
            variance += pow(blockMeans[r][c] - extimator,2);  
        }

        variance = variance/(blockMeans[r].size() - 1);

        cout<< " std = "<< sqrt(variance)/sqrt(blockMeans[r].size())
            << " Rx = "<< variance * pow(10,r+1)
            << endl;
    }

    return {(double) avgOrder / N, extimator};
}

// Backward-compatible overload:
// scalar-band energy epsilon(k), single phonon mode frequency w0,
// scalar-band coupling g(k, q). Internally routed to the indexed API.
pair<double, double> DMC(
    double tau,
    const double k,
    const double t,
    const double mu,
    const std::function<double(double, double)>& gFunc,
    const double w0,
    const double up_lim,
    const int N,
    const int bins,
    vector<double>& histogram,
    vector<int>& orderDist,
    vector<double>& exact_estimator,
    bool tauChange /* = true */
)
{
    auto eIndexed = [mu, t](double kElectron, int /*band*/) {
        return -2 * t * cos(kElectron) - mu;
    };
    auto wIndexed = [w0](int /*mode*/) {
        return w0;
    };
    auto gIndexed = [gFunc](double kElectron, double qPhonon,
                            int /*previousBand*/, int /*newBand*/, int /*mode*/) {
        return gFunc(kElectron, qPhonon);
    };

    return DMC(
        tau,
        k,
        t,
        mu,
        gIndexed,
        wIndexed,
        eIndexed,
        0,
        0,
        1,
        1,
        1,
        up_lim,
        N,
        bins,
        histogram,
        orderDist,
        exact_estimator,
        tauChange
    );
}
