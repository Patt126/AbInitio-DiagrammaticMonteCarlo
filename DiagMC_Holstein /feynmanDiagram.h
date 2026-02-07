#ifndef Diag_H
#define Diag_H
#include <functional>

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

    Vertex(double time, bool create, double q, double kIn = 0, double kOut = 0);
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

    const double k;    // total initial electron momentum
    const double mu;   // chemical potential
    const double w0;   // phonon frequency
    const double g;    // coupling constant
    const double t;    // hopping amplitude
    const std::function<double(double)> E; // dispersion ε(k)

public:
    Diag(double T,
         double t,
         double initial_k,
         double mu,
         double w0,
         double g,
         std::function<double(double)> energyFunc);
    ~Diag();

    // add a new phonon line between inTime and endTime with momentum q
    // prevIn and nextEnd are the neighboring vertices in the list
    void insertPhononLine(Vertex* prevIn, Vertex* nextEnd,
                          double inTime, double endTime, double q);

    // remove a phonon line given one of its vertices (inVertex)
    void removePhononLine(Vertex* inVertex);

    // compute Metropolis ratios for proposed moves
    double getAddWeight(double inTime, double endTime, double q,
                        Vertex*& prevIn, Vertex*& nextEnd);
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
    double getTailTime() const;
    int getOrder() const;        // number of phonon lines *2 = number of vertices
    int getPhLineNum() const;    // number of phonon lines
    bool testConservation() const; // momentum conservation check
    double getFullWeight() const;  // total diagram weight (product of all propagators)

private:
    void deleteAll();  // internal cleanup
    int vrtxCount;     // number of vertices currently in the list
};

#endif // Diag_H
