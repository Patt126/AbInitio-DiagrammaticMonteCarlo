#ifndef Diag_H
#define Diag_H
#include <functional>
#include <complex>

// vertex = node in the linked list representing one interaction point
// each phonon line corresponds to two vertices connected through 'link'
class Vertex {
public:
    double time;     // imaginary time of this vertex
    int sign;        // +1 emission, -1 absorption
    Vertex* next;    // next vertex in time order
    Vertex* prev;    // previous vertex in time order
    Vertex* link;    // partner vertex (start ↔ end of same phonon line)
    double kIn;      // electron momentum before vertex
    double kOut;     // electron momentum after vertex
    double q;        // phonon momentum (same magnitude, opposite sign at linked vertex)
    int bandIn;      // electron band before vertex
    int bandOut;     // electron band after vertex
    int mode;        // phonon mode carried by the linked phonon line

    Vertex(double time, bool create, double q,
           int mode = 0,
           int bandIn = 0,
           int bandOut = 0,
           double kIn = 0,
           double kOut = 0);
};

// continuous-time Feynman diagram for a single electron coupled to phonons
// stored as a doubly linked list of vertices ordered by time
// methods compute weights and perform local Monte Carlo updates
// all internal methods work with times scaled by total tau
class Diag {
private:
    Vertex* head;      // first vertex (emission)
    Vertex* tail;      // last vertex (absorption or end of electron line)
    double tau;        // total propagation time
    void updateK(bool add, double q, Vertex* inVertex); // update k along electron line after insert/remove

    using EnergyFunc = std::function<double(double, int)>;               // epsilon(k, band)
    using PhononFreqFunc = std::function<double(int)>;                   // w(mode)
    using CouplingFunc = std::function<double(double, double, int, int, int)>; // g(k, q, band_prev, band_new, mode)
    using ComplexCouplingFunc = std::function<std::complex<double>(double, double, int, int, int)>; // complex g

    const double k;    // total initial electron momentum
    const double mu;   // chemical potential
    const int electronBand; // active electron band index
    const int phononMode;   // active phonon mode index
    const PhononFreqFunc wFunc; // phonon frequency w(mode)
    const CouplingFunc gFunc; // momentum and index dependent coupling g(...)
    const double t;    // hopping amplitude
    const EnergyFunc E; // dispersion epsilon(k, band)

public:
    // full constructor with optional band/mode indices
    Diag(double T,
         double t,
         double initial_k,
         double mu,
         CouplingFunc couplingFunc,
         PhononFreqFunc phononFreqFunc,
         EnergyFunc energyFunc,
         int electronBand = 0,
         int phononMode = 0);

    // backward-compatible constructor: constant w0, scalar-band functions
    Diag(double T,
         double t,
         double initial_k,
         double mu,
         double w0,
         std::function<double(double, double)> couplingFunc,
         std::function<double(double)> energyFunc,
         int electronBand = 0,
         int phononMode = 0);
    ~Diag();

    // add a new phonon line between inTime and endTime with momentum q
    // prevIn and nextEnd are the neighboring vertices in the list
    void insertPhononLine(Vertex* prevIn, Vertex* nextEnd,
                          double inTime, double endTime, double q,
                          int inBandIn = -1,
                          int inBandOut = -1,
                          int endBandIn = -1,
                          int endBandOut = -1,
                          int mode = -1);

    // remove a phonon line given one of its vertices (inVertex)
    void removePhononLine(Vertex* inVertex);

    // compute Metropolis ratios for proposed moves
    double getAddWeight(double inTime, double endTime, double q,
                        Vertex*& prevIn, Vertex*& nextEnd,
                        int inBandOut = -1,
                        int endBandOut = -1,
                        int mode = -1);
    double getRemoveWeight(int lineNum, Vertex*& inVertex);
    double getTauWeight(double tauNew);      // for pure tau change
    double getStretchWeight(double tauNew);  // for rescaling all vertex times

    // apply accepted updates
    void stretchDiagram(double tauNew);
    void setTau(double tau);

    // diagnostics and accessors
    void printDiagram() const;
    double getTau() const;
    double getK() const;
    double getEnergy(double kElectron) const;
    double getEnergy(double kElectron, int band) const;
    double getPhononFrequency() const;
    double getPhononFrequency(int mode) const;
    double getCoupling(double kElectron, double qPhonon,
                       int previousBand, int newBand, int mode) const;
    double getCoupling(double kElectron, double qPhonon) const;
    double getCouplingSquared(double kElectron, double qPhonon,
                              int previousBand, int newBand, int mode) const;
    double getCouplingSquared(double kElectron, double qPhonon) const;
    double getCouplingSign(const CouplingFunc& signFunc) const;
    std::complex<double> getCouplingPhase(const ComplexCouplingFunc& phaseFunc) const;
    int getElectronBand() const;
    int getPhononMode() const;
    double getTailTime() const;
    int getTailBand() const;
    int getOrder() const;        // number of phonon lines *2 = number of vertices
    int getPhLineNum() const;    // number of phonon lines
    bool testConservation() const; // momentum conservation check
    double getFullWeight() const;  // total diagram weight (product of all propagators)

private:
    void deleteAll();  // internal cleanup
    void syncBands();  // enforce band continuity along electron line
    int vrtxCount;     // number of vertices currently in the list
};

#endif // Diag_H
