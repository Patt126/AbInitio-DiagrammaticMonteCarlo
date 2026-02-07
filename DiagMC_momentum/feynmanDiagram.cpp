#include "feynmanDiagram.h"
#include <iostream>
#include <math.h>
#include <functional>
#include <complex>
using namespace std;

// Vertex constructor
// time   = imaginary time of this vertex
// create = true  -> emission vertex  (sign = +1)
//          false -> absorption vertex (sign = -1)
// q, mode        = phonon momentum/mode carried by this phonon line
// kIn, bandIn    = electron momentum/band just before the vertex
// kOut, bandOut  = electron momentum just after the vertex
Vertex::Vertex(double time, bool create, double q, int mode,
               int bandIn, int bandOut, double kIn, double kOut ) {
    this->time = time;
    if(create) this->sign = +1;
    else this->sign = -1;
    this->kIn  = kIn;
    this->kOut = kOut;
    this->q    = q;
    this->bandIn = bandIn;
    this->bandOut = bandOut;
    this->mode = mode;
    next = nullptr;
    prev = nullptr;
    link = nullptr; // will be set to the partner vertex (start<->end of the same phonon line)
}

// Diagram constructor (initial/final band/momentum, hopping parameter and phonon mode if fixed)
// energyFunc: epsilon(k, band)
// phononFreqFunc: w(q, mode)
// couplingFunc: g(k, q, previousBand, newBand, mode)
Diag::Diag(double T, double t, double initial_k, double mu,
           Diag::CouplingFunc couplingFunc,
           Diag::PhononFreqFunc phononFreqFunc,
           Diag::EnergyFunc energyFunc,
           int electronBand,
           int phononMode)
   : k(initial_k),
     mu(mu),
     electronBand(electronBand),
     phononMode(phononMode),
     wFunc(phononFreqFunc),
     gFunc(couplingFunc),
     t(t),
     E(energyFunc)
{
    head = nullptr;
    tail = nullptr;
    tau = T;
    vrtxCount = 0;
}

// Backward-compatible constructor:
// scalar epsilon(k), scalar w0, scalar-band g(k, q)
Diag::Diag(double T, double t, double initial_k, double mu, double w0,
           std::function<double(double, double)> couplingFunc,
           std::function<double(double)> energyFunc,
           int electronBand,
           int phononMode)
    : Diag(
        T,
        t,
        initial_k,
        mu,
        [couplingFunc](double kElectron, double qPhonon,
                       int /*previousBand*/, int /*newBand*/, int /*mode*/) {
            return couplingFunc(kElectron, qPhonon);
        },
        [w0](int /*mode*/) { return w0; },
        [energyFunc](double kElectron, int /*band*/) { return energyFunc(kElectron); },
        electronBand,
        phononMode)
{}

// destructor -> free all allocated Vertex nodes
Diag::~Diag() {
    deleteAll();
}

// total number of vertices (each phonon line contributes 2 vertices)
int Diag::getOrder() const{
    return vrtxCount;
}

// number of phonon lines
int Diag::getPhLineNum() const{
    return vrtxCount/2;
}

// time of last vertex (0 if no vertices)
double Diag::getTailTime() const{
    if(tail){
        return (tail->time);
    } else {
        return 0;
    }
}

int Diag::getTailBand() const{
    return tail ? tail->bandOut : electronBand;
}

// set total propagation time tau
// requirement: tau must be strictly after the last vertex time, or physics is inconsistent
void Diag::setTau(double tau){
    if(tail != nullptr  && tau <= (tail->time)){
        cout<<"the end cannot be before/coincide the last vertex\n";
        cout<< tau<<" "<<tail->time<<endl;
    } else {
        this->tau = tau;
    }
}

double Diag::getTau() const{
    return tau;
}

double Diag::getK() const{
    return this->k;
}

double Diag::getEnergy(double kElectron) const{
    return E(kElectron, electronBand);
}

double Diag::getEnergy(double kElectron, int band) const{
    return E(kElectron, band);
}

double Diag::getPhononFrequency() const{
    return wFunc(phononMode);
}

double Diag::getPhononFrequency(int mode) const{
    return wFunc(mode);
}

double Diag::getCoupling(double kElectron, double qPhonon) const{
    return getCoupling(kElectron, qPhonon, electronBand, electronBand, phononMode); 
}

double Diag::getCoupling(double kElectron, double qPhonon,
                         int previousBand, int newBand, int mode) const{
    return gFunc(kElectron, qPhonon, previousBand, newBand, mode);
}

double Diag::getCouplingSquared(double kElectron, double qPhonon) const{
    return getCouplingSquared(kElectron, qPhonon, electronBand, electronBand, phononMode);
}

double Diag::getCouplingSquared(double kElectron, double qPhonon,
                                int previousBand, int newBand, int mode) const{
    double gLocal = getCoupling(kElectron, qPhonon, previousBand, newBand, mode);
    return gLocal * gLocal;
}

double Diag::getCouplingSign(const CouplingFunc& signFunc) const{
    if(!head){
        return 1.0;
    }
    double sign = 1.0;
    Vertex* temp = head;
    while(temp){
        const double gLocal = signFunc(
            temp->kIn, temp->q, temp->bandIn, temp->bandOut, temp->mode);
        if(gLocal == 0.0){
            return 0.0;
        }
        if(gLocal < 0.0){
            sign = -sign;
        }
        temp = temp->next;
    }
    return sign;
}

std::complex<double> Diag::getCouplingPhase(const ComplexCouplingFunc& phaseFunc) const{
    if(!head){
        return {1.0, 0.0};
    }
    std::complex<double> phase(1.0, 0.0);
    Vertex* temp = head;
    while(temp){
        const std::complex<double> gLocal =
            phaseFunc(temp->kIn, temp->q, temp->bandIn, temp->bandOut, temp->mode);
        const double mag = std::abs(gLocal);
        if(mag == 0.0){
            return {0.0, 0.0};
        }
        phase *= (gLocal / mag);
        temp = temp->next;
    }
    return phase;
}

int Diag::getElectronBand() const{
    return electronBand;
}

int Diag::getPhononMode() const{
    return phononMode;
}

// updateK propagates electron momentum changes along the electron line
// this must be called after adding a phonon line (add=true) or before removing it (add=false)
// idea:
//  - inserting a phonon line changes electron momentum by +q at emission,
//    and gives it back (-q) at absorption
//  - all intermediate electron segments between the two vertices shift by +q
//  - removing a line is the inverse operation (-q on each affected segment)
// this keeps kIn/kOut consistent along the chain
void Diag::updateK(bool add, double q, Vertex* inVertex){
    double s = -1; // default remove (we subtract q back)
    if(add){
        s = +1; // when adding: we add q along the segment until the partner vertex

        // set kIn/kOut for the first vertex of the new phonon line
        if(inVertex == head){
            inVertex->kIn  = this->k;
            inVertex->kOut = this->k + q;
        } else {
            inVertex->kIn  = inVertex->prev->kOut;
            inVertex->kOut = inVertex->kIn + q;
        }
    }

    // walk forward from the emission vertex up to (but not including) the absorption vertex
    // shift kIn and kOut by s*q
    Vertex* current = inVertex->next;
    while(current != inVertex->link){
        current->kIn  += s*q;
        current->kOut += s*q;
        current = current->next;
    }

    //set kIn/kOut at the absorption vertex
    if(add){
        inVertex->link->kIn  = inVertex->link->prev->kOut;
        inVertex->link->kOut = inVertex->link->kIn - q;
        // debug:
        // if(inVertex->link->next && inVertex->link->kOut != inVertex->link->next->kIn) { ... }
    }
}

// insert a phonon line made of two vertices:
//  newInVertex  at inTime  (emission, sign=+1, momentum q)
//  newEndVertex at endTime (absorption, sign=-1, momentum -q)
// prevIn  = vertex that comes right before inTime (or nullptr if at head)
// nextEnd = vertex that comes right after endTime (or nullptr if at tail)
// this function updates the linked list pointers in all corner cases:
//  - empty diagram
//  - both vertices before current head
//  - both after current tail
//  - only first before head
//  - only second after tail
//  - general middle insertion
void Diag::insertPhononLine(Vertex* prevIn, Vertex* nextEnd,
                            double inTime, double endTime, double q,
                            int inBandIn, int inBandOut,
                            int endBandIn, int endBandOut,
                            int mode) {
    // resolved... variable implement the logic to correctly identify 
    // the initial and final band and phonon mode of each vertex     
    int resolvedMode = (mode >= 0) ? mode : phononMode; 
    int resolvedInBandIn = (inBandIn >= 0) // ingoing band (previous node or initial one)
        ? inBandIn
        : (prevIn ? prevIn->bandOut : electronBand);
    int resolvedInBandOut = (inBandOut >= 0) ? inBandOut : resolvedInBandIn; // default assignment for safety

    bool noVerticesBetween = false;
    if(vrtxCount == 0){ // empty
        noVerticesBetween = true;
    } else if(nextEnd){ // consecuitive
        Vertex* firstAfterIn = prevIn ? prevIn->next : head;
        noVerticesBetween = (firstAfterIn == nextEnd);
    } else { // line at the end
        noVerticesBetween = (prevIn == tail);
    }

    int defaultEndBandIn = resolvedInBandOut;
    if(!noVerticesBetween){
        defaultEndBandIn = nextEnd ? nextEnd->prev->bandOut : tail->bandOut; // if there is nothing next and is the last
    }

    int resolvedEndBandIn = (endBandIn >= 0) ? endBandIn : defaultEndBandIn; 
    int resolvedEndBandOut = (endBandOut >= 0) ? endBandOut : resolvedEndBandIn; 
    if(nextEnd == nullptr){
        // If annihilation is inserted as the last vertex, keep the outgoing
        // band fixed to the tail propagator band (or external band if empty).
        resolvedEndBandOut = tail ? tail->bandOut : electronBand;
    }

    Vertex* newInVertex  = new Vertex(inTime,  true,  q,
                                      resolvedMode, resolvedInBandIn, resolvedInBandOut);
    Vertex* newEndVertex = new Vertex(endTime, false, -q,
                                      resolvedMode, resolvedEndBandIn, resolvedEndBandOut);
    
    // case: diagram was empty
    if (vrtxCount == 0) {
        head = newInVertex;
        tail = newEndVertex;

        head->next = tail;
        tail->prev = head;

        head->link = tail;
        tail->link = head;

        updateK(true, q, newInVertex);
        syncBands();

        vrtxCount += 2;
        return;
    }

    // case: both vertices go before current head (endTime < head->time)
    if(nextEnd == head){
        newInVertex->prev = nullptr;
        newInVertex->next = newEndVertex;

        newEndVertex->prev = newInVertex;
        newEndVertex->next = head;

        head->prev = newEndVertex;
        head = newInVertex;

        newInVertex->link  = newEndVertex;
        newEndVertex->link = newInVertex;

        updateK(true, q, newInVertex);
        syncBands();
        vrtxCount += 2;
        return;
    }

    // case: both vertices go after current tail (inTime > tail->time)
    else if(prevIn == tail){
        newInVertex->prev = tail;
        newInVertex->next = newEndVertex;

        newEndVertex->prev = newInVertex;
        newEndVertex->next = nullptr;

        tail->next = newInVertex;
        tail = newEndVertex;

        newInVertex->link  = newEndVertex;
        newEndVertex->link = newInVertex;

        updateK(true, q, newInVertex);
        syncBands();
        vrtxCount += 2; 
        return;
    }

    // case: only the first vertex is before head (inTime < head->time), but end is internal / tail
    else if (prevIn == nullptr){

        // insert newInVertex at head
        newInVertex->prev = nullptr;
        newInVertex->next = head;
        head->prev = newInVertex;
        head = newInVertex;

        // place newEndVertex either before nextEnd or at tail
        if(!nextEnd){
            // end is after current tail
            newEndVertex->next = nullptr;
            newEndVertex->prev = tail;
            tail->next = newEndVertex;
            tail = newEndVertex;
        }
        else{
            newEndVertex->next = nextEnd;
            newEndVertex->prev = nextEnd->prev;
            nextEnd->prev->next = newEndVertex;
            nextEnd->prev = newEndVertex;       
        }

        vrtxCount += 2;

        newInVertex->link  = newEndVertex;
        newEndVertex->link = newInVertex;

        updateK(true, q, newInVertex);
        syncBands();
        return;
    }

    // case: only the second vertex is after tail (endTime > tail->time)
    else if (nextEnd == nullptr){
        
        // append newEndVertex at tail
        newEndVertex->next = nullptr;
        newEndVertex->prev = tail;
        tail->next = newEndVertex;
        tail = newEndVertex;
       
        // insert newInVertex right after prevIn
        newInVertex->next = prevIn->next;
        newInVertex->prev = prevIn;
        prevIn->next->prev = newInVertex;
        prevIn->next = newInVertex;

        vrtxCount += 2;

        newInVertex->link  = newEndVertex;
        newEndVertex->link = newInVertex;

        updateK(true, q, newInVertex);
        syncBands();
        return;
    }

    // general middle insertion:
    // we insert the emission vertex after prevIn,
    // and the absorption vertex before nextEnd
    // note: there is a possible adjacent-case optimization but not critical
    newEndVertex->link = newInVertex;
    newInVertex->link  = newEndVertex;

    newInVertex->next = prevIn->next;
    newInVertex->prev = prevIn;
    prevIn->next->prev = newInVertex;
    prevIn->next = newInVertex;
    vrtxCount++;

    newEndVertex->next = nextEnd;
    newEndVertex->prev = nextEnd->prev;
    nextEnd->prev->next = newEndVertex;
    nextEnd->prev = newEndVertex;    
    vrtxCount++;

    updateK(true, q, newInVertex);
    syncBands();
}

// remove a phonon line given its starting vertex (emission vertex)
// inVertex must have sign>0. We identify the partner (link), then splice them out.
// multiple structural cases:
// - diagram goes to empty
// - removing first two vertices at the head
// - removing a line where the start is at head but the end is not tail
// - removing a line that's local (no crossings)
// - removing a line that crosses other phonons (general case)
// updateK(false, ...) is called when the momentum jump has to be undone
void Diag::removePhononLine(Vertex* inVertex){
    if(inVertex->sign < 0){
        cout<<"pls specify a starting vertex\n";
        return;
    }

    // case: removing the only phonon line
    if(vrtxCount==2){
        vrtxCount = 0;
        delete head;
        head = nullptr;
        delete tail;
        tail = nullptr;
        return;
    }

    Vertex* endVertex = inVertex->link;

    // case: they are the first two vertices (inVertex == head and immediately followed by its partner)
    // here k bookkeeping doesn't change because there's no segment before head
    if(inVertex == head && inVertex->next == endVertex){
        endVertex->next->prev = nullptr;
        delete head;
        head = endVertex->next;
        delete endVertex;
        vrtxCount -= 2;
        syncBands();
        return;
    }

    // case: start is head but endVertex is somewhere else
    // here we need to restore momenta along the affected segment
    if(inVertex == head){
        updateK(false, inVertex->q, inVertex);

        Vertex* newHead = head->next;
        newHead->prev = nullptr;
        delete head;
        head = newHead;
        
        // if endVertex was tail, update tail
        if(!(endVertex->next)){
            tail = endVertex->prev;
            tail->next = nullptr;
            delete endVertex;
        }
        else{
            endVertex->prev->next = endVertex->next;
            endVertex->next->prev = endVertex->prev;
            delete endVertex;
        }

        vrtxCount -= 2;
        syncBands();
        return;
    }

    // case: no crossing (the endVertex is exactly the next of inVertex)
    // just splice them both out. updateK is not needed because no internal segment changes k
    if(endVertex->prev == inVertex){
        // subcase: both are at the tail
        if(endVertex == tail){
            tail = inVertex->prev;
            tail->next = nullptr;
            delete inVertex;
            delete endVertex;
            vrtxCount -= 2;
            syncBands();
            return;
        }
        
        endVertex->next->prev = inVertex->prev;
        inVertex->prev->next  = endVertex->next;
        delete inVertex;
        delete endVertex;
        vrtxCount -= 2;
        syncBands();
        return;
    }

    // general crossing case:
    // the phonon line spans across other phonon lines
    // we need to restore electron momentum along that interval
    updateK(false, inVertex->q, inVertex);

    // remove endVertex
    if(endVertex == tail){
        tail = endVertex->prev;
        tail->next = nullptr;
    } else {
        endVertex->next->prev = endVertex->prev;
        endVertex->prev->next = endVertex->next;
    }

    // remove inVertex
    inVertex->next->prev = inVertex->prev;
    inVertex->prev->next = inVertex->next;

    delete inVertex;
    delete endVertex;
    vrtxCount -= 2;
    syncBands();
    return;
}

// rescale the entire diagram in time so that total tau becomes tauNew
// multiply every vertex time by ratio = tauNew / oldTau
void Diag::stretchDiagram(double tauNew){
    double ratio = tauNew/(this->tau);
    if(vrtxCount != 0){
        Vertex* temp = head;
        while(temp){
            temp->time *= ratio;
            temp = temp->next;
        }
    }
    this->tau = tauNew;
}

// getAddWeight:
// constructive evaluation of W_add/W following the F1*F2*F3 logic in appendix B:
//  - creation vertex at inTime with k -> k+q, band: inBandIn -> inBandOut
//  - momentum shift +q propagated up to endTime
//  - annihilation vertex at endTime with q -> -q, band: endBandIn -> endBandOut
//  - post-end boundary correction (segment + next vertex coupling if present)
// note: this function does not modify the diagram, it only computes the ratio
double Diag::getAddWeight(double inTime, double endTime, double q,
                          Vertex*& prevIn, Vertex*& nextEnd,
                          int inBandOut, int endBandOut, int mode){
    if(endTime <= inTime || inTime < 0.0 || endTime > this->tau){
        return 0.0;
    }

    int resolvedMode = (mode >= 0) ? mode : phononMode;

    // first vertex at/after inTime -> used to identify the creation position
    Vertex* firstAfterIn = head;
    while(firstAfterIn && firstAfterIn->time < inTime){
        firstAfterIn = firstAfterIn->next;
    }
    // strict ordering: reject coincident insertion times
    if(firstAfterIn && firstAfterIn->time == inTime){
        cout<<"this time arleady correspond to a phonon\n";
        return 0.0;
    }
    // first vertex strictly before tau_I (i in the appendix), nullptr -> at head
    prevIn = firstAfterIn ? firstAfterIn->prev : tail;  

    int inBandIn = prevIn ? prevIn->bandOut : electronBand; 
    int resolvedInBandOut = (inBandOut >= 0) ? inBandOut : inBandIn;

    // first vertex strictly after endTime (j in the appendix) -> identifies the annihilation position
    Vertex* firstAfterEnd = firstAfterIn;
    while(firstAfterEnd && firstAfterEnd->time <= endTime){
        if(firstAfterEnd->time == endTime){
            cout<<"this time arleady correspond to a phonon\n";
            return 0.0;
        }
        firstAfterEnd = firstAfterEnd->next;
    }
    nextEnd = firstAfterEnd;

    double ratio = 1.0;
    double kInAtEmission = prevIn ? prevIn->kOut : this->k;

    // F1 + F2 (constructive traversal): electron propagator changes and
    // coupling reweighting for all existing vertices in (inTime, endTime).
    double tLeft = inTime;
    int oldBandSegment = inBandIn;
    int newBandSegment = resolvedInBandOut;
    int resolvedEndBandIn = resolvedInBandOut;

    Vertex* shifted = firstAfterIn;
    while(shifted && shifted->time < endTime){ // iterate till vertex j-1 before current line end
        const double dt = shifted->time - tLeft;
        const double kOldSeg = shifted->kIn;
        ratio *= exp(
            -(getEnergy(kOldSeg + q, newBandSegment) - getEnergy(kOldSeg, oldBandSegment)) * dt
        ); 

        const double old_g = getCoupling(shifted->kIn, shifted->q,
                                        shifted->bandIn, shifted->bandOut, shifted->mode);
        if(old_g == 0.0){
            return 0.0;
        }
        // this differ just for the ingoing band & q increase
        const double new_g = getCoupling(shifted->kIn + q, shifted->q,
                                        newBandSegment, shifted->bandOut, shifted->mode);
        ratio *= new_g / old_g;

        tLeft = shifted->time;
        oldBandSegment = shifted->bandOut;
        newBandSegment = shifted->bandOut; // for the followin there is no more band alteration (just +q in momentum)
        resolvedEndBandIn = shifted->bandOut; // iterate to find the correct band ingoing the destructor we are inserting
        shifted = shifted->next;
    }

    // last shifted electron segment up to the annihilation time 
    const double kOldFinalSeg = nextEnd ? nextEnd->kIn : this->k; // condition to handle final vertex
    ratio *= exp(
        -(getEnergy(kOldFinalSeg + q, newBandSegment) - getEnergy(kOldFinalSeg, oldBandSegment))
        * (endTime - tLeft)
    );

    int resolvedEndBandOut = (endBandOut >= 0) ? endBandOut : resolvedEndBandIn;

    // F3 boundary contribution after annihilation (segment and next vertex/ handle if last)
    int oldBandAfterEnd = nextEnd ? nextEnd->bandIn : (tail ? tail->bandOut : electronBand);
    if(nextEnd == nullptr){
        // if annihilation is the last vertex, keep outgoing band fixed to tail propagator band
        resolvedEndBandOut = oldBandAfterEnd;
    }
    double kAfterEnd = nextEnd ? nextEnd->kIn : this->k;
    double dtAfterEnd = nextEnd ? (nextEnd->time - endTime) : (this->tau - endTime);
    ratio *= exp(
        -(getEnergy(kAfterEnd, resolvedEndBandOut) - getEnergy(kAfterEnd, oldBandAfterEnd))
        * dtAfterEnd
    );

    if(nextEnd && resolvedEndBandOut != oldBandAfterEnd){
        const double old_g = getCoupling(nextEnd->kIn, nextEnd->q,
                                        oldBandAfterEnd, nextEnd->bandOut, nextEnd->mode);
        if(old_g == 0.0){
            return 0.0;
        }
        const double new_g = getCoupling(nextEnd->kIn, nextEnd->q,
                                        resolvedEndBandOut, nextEnd->bandOut, nextEnd->mode);
        ratio *= new_g / old_g;
    }
    // else: is the last, we will force end band out to the same of the final propagator

    // new phonon line factors: D0 * g_create * g_annih
    double gCreate = getCoupling(kInAtEmission, q, inBandIn, resolvedInBandOut, resolvedMode);
    double gAnnih = getCoupling(kAfterEnd + q, -q,
                                resolvedEndBandIn, resolvedEndBandOut, resolvedMode);
    ratio *= gCreate * gAnnih;
    ratio *= exp(-getPhononFrequency(resolvedMode) * (endTime - inTime));
    return ratio;
}

// getRemoveWeight:
// choose a phonon line by its index (1..#lines) and compute Metropolis ratio
// this returns W(new)/W(old), basically inverse of getAddWeight
// also sets inVertex to the emission vertex of that line
double Diag::getRemoveWeight(int lineNum, Vertex*& inVertex){
    if(lineNum*2 > vrtxCount || lineNum < 1){
        cout<<"not many phonon lines\n";
        return 0;
    }

    // select the emission vertex of the requested phonon line index
    int num = 0;
    inVertex = head;
    while(inVertex){
        if(inVertex->sign > 0){
            num++;
            if(num == lineNum){
                break;
            }
        }
        inVertex = inVertex->next;
    }
    if(!inVertex || inVertex->sign <= 0){
        cout<<"line selection failed\n";
        return 0.0;
    }

    // compute ratio from electron propagators between inVertex and its partner
    double diagWeight = 1.0;
    double q = inVertex->q;
    Vertex* temp = inVertex; // start from emission vertex
    if(temp->next){
        diagWeight *= exp(-(getEnergy(temp->kOut - q, temp->bandIn) //Note now the final Green function as momentum k-q
                          - getEnergy(temp->kOut, temp->bandOut))
                          *(temp->next->time - temp->time));
        temp = temp->next;
    }
    while(temp != inVertex->link){
        int segmentBand = temp->bandOut;
        diagWeight *= exp(-(getEnergy(temp->kOut - q, segmentBand)
                          - getEnergy(temp->kOut, segmentBand))
                          *(temp->next->time - temp->time));
        temp = temp->next;
    }

    // removing this line shifts all vertices in (t_in, t_end) by -q.
    // all vertex couplings in that interval are reweighted.
    double couplingRatio = 1.0;
    Vertex* shifted = inVertex->next;
    int newBandCursor = inVertex->bandIn; // in the first iteration consider the fact that removing the vertex following vertex change in band
    while(shifted != inVertex->link){
        double old_g = getCoupling(shifted->kIn, shifted->q,
                                  shifted->bandIn, shifted->bandOut, shifted->mode);
        if(old_g == 0.0){
            return 0.0;
        }
        double new_g = getCoupling(shifted->kIn - q, shifted->q,
                                  newBandCursor, shifted->bandOut, shifted->mode);
        couplingRatio *= new_g / old_g;
        newBandCursor = shifted->bandOut;  // Following iteratio should be the same (current vertex band out= following band in)
        shifted = shifted->next;
    }

    Vertex* nextAfterEnd = inVertex->link->next; // vertex after anhilation we are removing
    int oldBandAfterEnd = inVertex->link->bandOut;
    int newBandAfterEnd = (inVertex->link->prev == inVertex) // deal with case there are no other line crossing
        ? inVertex->bandIn
        : inVertex->link->prev->bandOut;
    // check if current vertex is not tail for momentum and time
    double kAfterEnd = nextAfterEnd ? nextAfterEnd->kIn : this->k; 
    double dtAfterEnd = nextAfterEnd
        ? (nextAfterEnd->time - inVertex->link->time)
        : (this->tau - inVertex->link->time);
    
    double postEndBandRatio = exp(
        -(getEnergy(kAfterEnd, newBandAfterEnd) - getEnergy(kAfterEnd, oldBandAfterEnd))
        * dtAfterEnd);
    // correct band in in following vertex if some
    if(nextAfterEnd && newBandAfterEnd != oldBandAfterEnd){
        double old_g = getCoupling(nextAfterEnd->kIn, nextAfterEnd->q,
                                  oldBandAfterEnd, nextAfterEnd->bandOut, nextAfterEnd->mode);
        if(old_g == 0.0){
            return 0.0;
        }
        double new_g = getCoupling(nextAfterEnd->kIn, nextAfterEnd->q,
                                  newBandAfterEnd, nextAfterEnd->bandOut, nextAfterEnd->mode);
        couplingRatio *= new_g / old_g;
    }

    double gCreate = getCoupling(inVertex->kIn, inVertex->q,
                                 inVertex->bandIn, inVertex->bandOut, inVertex->mode);
    double gAnnih = getCoupling(inVertex->link->kIn, inVertex->link->q,
                                inVertex->link->bandIn, inVertex->link->bandOut, inVertex->link->mode);
    if(gCreate == 0.0 || gAnnih == 0.0){
        return 0.0;
    }
    return diagWeight
         * exp(getPhononFrequency(inVertex->mode)*(inVertex->link->time - inVertex->time)) // 1/D^0 just + in exp
         * postEndBandRatio
         * couplingRatio
         / (gCreate * gAnnih);
}

// getTauWeight:
// Metropolis factor for changing only the final tau
// electron just propagates freely for the additional tail
double Diag::getTauWeight(double tauNew){
    int tailBand = tail ? tail->bandOut : electronBand;
    return exp(-getEnergy(this->k, tailBand)*(tauNew - this->tau));
}

// getStretchWeight:
// Metropolis factor for rescaling the whole diagram from tau to tauNew
// this multiplies every time interval by (1+strain), we accumulate
// contributions from electron and phonon propagators accordingly
double Diag::getStretchWeight(double tauNew){
    double strain = tauNew/(this->tau) - 1;

    if(vrtxCount == 0){
        return exp(-getEnergy(this->k, electronBand)*(this->tau)*strain);
    } 

    // first vertex contribution:
    //   electron from 0->head->time
    //   + phonon line starting at head if head is an emission
    double weight = exp((-getEnergy(this->k, head->bandIn)*(head->time)
                          - getPhononFrequency(head->mode)*(head->link->time - head->time))*strain);

    Vertex* temp = head->next;
    while(temp){
        // electron free propagator between previous and current vertex
        weight *= exp(-getEnergy(temp->kIn, temp->bandIn)
                      *(temp->time - temp->prev->time )*strain);

        // if this vertex is an emission (sign>0) start of a phonon line,
        // include its phonon propagator weight
        if(temp->sign > 0){
            weight *= exp(-getPhononFrequency(temp->mode)*(temp->link->time - temp->time)*strain);
        }
        temp = temp->next;
    }

    // final tail from last vertex to tau
    weight *= exp(-getEnergy(this->k, tail->bandOut)*(this->tau - tail->time)*strain);
    return weight;
}

// printDiagram:
// human-readable dump of the current diagram
// prints each vertex with its time, sign, q, and local electron momentum
// also checks that the final k matches the initial k
void Diag::printDiagram() const{
    cout << "\n=============== DIAGRAMMA DETTAGLIATO ===============\n";
    if (!head) {
        cout << "Diagramma vuoto." << endl;
        cout << "e----[ k=" << this->k << ", b=" << electronBand
             << " ]----(t=" << this->tau << ")--->" << endl;
        cout << "=====================================================\n" << endl;
        return;
    }

    cout << "Ordine: " << getOrder()
         << ", #Linee: " << getPhLineNum()
         << ", k_initial: " << this->k
         << ", tau: " << this->tau << endl;
    
    // segmento iniziale prima del primo vertice
    printf("e----[k=%.1f,b=%d]----", this->k, head->bandIn);

    Vertex* temp = head;
    while (temp) {
        // stampa il vertice
        printf("●(t=%.1f, s=%+d, q=%.1f, b:%d->%d, nu=%d)",
               temp->time, temp->sign, temp->q, temp->bandIn, temp->bandOut, temp->mode);
        
        // stampa il segmento dopo il vertice
        printf("----[k=%.1f,b=%d]----", temp->kOut, temp->bandOut);
        
        temp = temp->next;
    }
    cout << "e_final" << endl;

    // check coerenza momento finale
    if (abs(tail->kOut - this->k) > 1e-9) {
        cout << "!!! ATTENZIONE: Momento finale (k=" << tail->kOut
             << ") NON coerente con quello iniziale (k=" << this->k << ") !!!" << endl;
    } else {
        cout << ">>> OK: Momento finale coerente (k_final = " << tail->kOut << ")" << endl;
    }
    cout << "=====================================================\n" << endl;
}

// testConservation:
// sanity check for momentum flow along the diagram
// at each vertex: kOut - kIn should match q (sign included)
// and along electron segments: kOut(previous) == kIn(next)
bool Diag::testConservation() const{
    Vertex* temp = head;
    double tol = 1e-9;
    bool state = true;
    while(temp){
        double err = abs(temp->kOut - temp->kIn - temp->q);
        if(err > tol){
            cout<<"problemone, non viene rispettata la conservazione nel vertice al tempo "<<temp->time<<endl;
            cout<<temp->sign<<" Kin= "<<temp->kIn<<" Kout= "<<temp->kOut<<" q= "<<temp->q<<endl;
            state = false;
        }
        if(temp->next && abs(temp->kOut - temp->next->kIn) > tol){
            cout<<"problemone, la linea fononica al tempo "<<temp->time<<" inizia e finisce con momento diverso"<<endl;
            cout<<temp->sign<<" start Kout= "<<temp->kOut<<" end Kin= "<<temp->next->kIn<<endl;
            state = false;
        }
        if(temp->next && temp->bandOut != temp->next->bandIn){
            cout<<"problemone, band mismatch tra vertici consecutivi al tempo "<<temp->time<<endl;
            cout<<"bandOut="<<temp->bandOut<<" next.bandIn="<<temp->next->bandIn<<endl;
            state = false;
        }
        if(temp->link && temp->mode != temp->link->mode){
            cout<<"problemone, mode mismatch sulla stessa linea fononica al tempo "<<temp->time<<endl;
            cout<<"mode(start)="<<temp->mode<<" mode(end)="<<temp->link->mode<<endl;
            state = false;
        }
        temp = temp->next;
    }
    return state;
}

// getFullWeight:
// usefull for debugging of weight ratio calculation
// compute the full diagram weight W for the current configuration
// product of:
//   electron propagators between vertices
//   phonon propagators for each emitted phonon line
//   one electron-phonon coupling factor per vertex
double Diag::getFullWeight() const{
    if(vrtxCount==0){
        return exp(-getEnergy(this->k, electronBand)*(this->tau));
    }

    // first segment: electron from 0 to first vertex
    double diagWeight = exp(-getEnergy(this->k, head->bandIn)*(head->time));

    Vertex* temp = head;
    while(temp){
        // every interaction vertex contributes one coupling factor
        diagWeight *= getCoupling(temp->kIn, temp->q, temp->bandIn, temp->bandOut, temp->mode);

        // phonon propagator is associated once per phonon line (at emission vertex)
        if(temp->sign > 0){
            diagWeight *= exp(-getPhononFrequency(temp->mode)*(temp->link->time - temp->time ));
        }

        // electron propagator to the next vertex
        if(temp->next){
            diagWeight *= exp(-getEnergy(temp->next->kIn, temp->next->bandIn)
                               *(temp->next->time - temp->time));
        }
        temp = temp->next;
    }

    // final tail from last vertex to tau
    diagWeight *= exp(-getEnergy(this->k, tail->bandOut)*(this->tau - tail->time)) ;
 
    return diagWeight;
}

void Diag::syncBands() {
    if(!head){
        return;
    }

    head->bandIn = electronBand;
    Vertex* temp = head;
    while(temp->next){
        temp->next->bandIn = temp->bandOut;
        temp = temp->next;
    }
}

// delete all vertices (used by destructor)
void Diag::deleteAll() {
    Vertex* temp = head;
    while (temp) {
        Vertex* nextVertex = temp->next;
        delete temp;
        temp = nextVertex;
    }
    head = nullptr;
}
