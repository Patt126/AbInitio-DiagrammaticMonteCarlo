#include "feynmanDiagram.h"
#include <iostream>
#include <math.h>
#include <functional>
using namespace std;

// Vertex constructor
// time   = imaginary time of this vertex
// create = true  -> emission vertex  (sign = +1)
//          false -> absorption vertex (sign = -1)
// q      = phonon momentum carried by this phonon line
// kIn    = electron momentum just before the vertex
// kOut   = electron momentum just after the vertex
Vertex::Vertex(double time, bool create, double q, double kIn, double kOut ) {
    this->time = time;
    if(create) this->sign = +1;
    else this->sign = -1;
    this->kIn  = kIn;
    this->kOut = kOut;
    this->q    = q;
    next = nullptr;
    prev = nullptr;
    link = nullptr; // will be set to the partner vertex (start<->end of the same phonon line)
}

// Diagram constructor
// tau = total propagation time
// t   = hopping
// initial_k = initial electron momentum
// mu, w0, g = physical parameters
// energyFunc = dispersion ε(k), injected from outside (lambda in DMC)
Diag::Diag(double T, double t, double initial_k, double mu, double w0, double g,
           std::function<double(double)> energyFunc) 
   : k(initial_k),
     t(t),
     mu(mu),
     w0(w0),
     g(g),
     E(energyFunc)
{
    head = nullptr;
    tail = nullptr;
    tau = T;
    vrtxCount = 0;
}

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

// updateK propagates electron momentum changes along the electron line
// this must be called after adding a phonon line (add=true) or before removing it (add=false)
// idea:
//  - inserting a phonon line changes electron momentum by -q at emission,
//    and gives it back (+q) at absorption
//  - all intermediate electron segments between the two vertices shift by -q
//  - removing a line is the inverse operation (+q back to each segment)
// this keeps kIn/kOut consistent along the chain
void Diag::updateK(bool add, double q, Vertex* inVertex){
    double s = +1; // default remove (we will add back q)
    if(add){
        s = -1; // when adding: we subtract q along the segment until the partner vertex

        // set kIn/kOut for the first vertex of the new phonon line
        if(inVertex == head){
            inVertex->kIn  = this->k;
            inVertex->kOut = this->k - q;
        } else {
            inVertex->kIn  = inVertex->prev->kOut;
            inVertex->kOut = inVertex->kIn - q;
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

    // now set kIn/kOut at the absorption vertex
    if(add){
        inVertex->link->kIn  = inVertex->link->prev->kOut;
        inVertex->link->kOut = inVertex->link->kIn + q; 
        // sanity check in debug:
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
void Diag::insertPhononLine(Vertex* prevIn, Vertex* nextEnd,double inTime, double endTime, double q) {
    
    Vertex* newInVertex  = new Vertex(inTime,  true,  q);
    Vertex* newEndVertex = new Vertex(endTime, false, -q);
    
    // case: diagram was empty
    if (vrtxCount == 0) {
        head = newInVertex;
        tail = newEndVertex;

        head->next = tail;
        tail->prev = head;

        head->link = tail;
        tail->link = head;

        updateK(true, q, newInVertex);

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
            return;
        }
        
        endVertex->next->prev = inVertex->prev;
        inVertex->prev->next  = endVertex->next;
        delete inVertex;
        delete endVertex;
        vrtxCount -= 2;
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
// compute Metropolis ratio for inserting a phonon line
// also identify where that insertion will go (prevIn / nextEnd)
// logic:
//  1. determine where inTime and endTime sit relative to existing vertices
//  2. compute product of electron propagators and phonon propagator over that interval
//  3. multiply by g^2
// note: does not actually modify the diagram
double Diag::getAddWeight(double inTime, double endTime, double q,
                          Vertex*& prevIn, Vertex*& nextEnd){
    
    // trivial: empty diagram
    if(vrtxCount == 0){
        return pow(g,2)
             * exp(-(E(this->k - q) - E(this->k))*(endTime - inTime))
             * exp(-w0*(endTime - inTime));
    }

    // case: start is after current tail
    if(inTime > tail->time){
        prevIn = tail;
        // end will also be after tail
        return pow(g,2)
             * exp(-(E(this->k - q) - E(this->k))*(endTime - inTime))
             * exp(-w0*(endTime - inTime));
    }

    // general case:
    // find the first existing vertex with time > inTime
    Vertex* inTemp = head;
    while(inTemp && inTemp->time <= inTime){
        if(inTemp->time == endTime) {
            cout<<"this time arleady correspond to a phonon\n";
            return 0.0;
        }
        inTemp = inTemp->next;
    }

    if(inTemp){
        prevIn = inTemp->prev;

        // if the next vertex in time is already after endTime, we can evaluate weight locally
        if(inTemp->time > endTime){
            nextEnd = inTemp;
            return pow(g,2)
                 * exp(-(E(inTemp->kIn - q) - E(inTemp->kIn))*(endTime - inTime))
                 * exp(-w0*(endTime - inTime));
        }
    }
    else{
        // inTime is after tail (should have been caught above)
        prevIn = tail;
    }

    // now we walk forward accumulating electron propagator factors between inTime and endTime
    nextEnd = inTemp;
    double DiagWeight =
        exp(-(E(nextEnd->kIn - q) - E(nextEnd->kIn))*(nextEnd->time - inTime));

    nextEnd = nextEnd->next;
    while(nextEnd && nextEnd->time <= endTime){
        if(nextEnd->time == endTime) {
            cout<<"this time arleady correspond to a phonon\n";
            return 0.0;
        }
        // multiply by propagator for each full segment we cross
        DiagWeight *= exp(-(E(nextEnd->kIn - q) - E(nextEnd->kIn))
                          *(nextEnd->time - nextEnd->prev->time));
        nextEnd = nextEnd->next;
    }
    
    if(nextEnd){
        // final partial segment before endTime
        DiagWeight *= exp(-(E(nextEnd->kIn - q) - E(nextEnd->kIn))
                          *(endTime - nextEnd->prev->time));
    }
    else{
        // endTime lies after current tail
        DiagWeight *= exp(-(E(this->k - q) - E(this->k))
                          *(endTime - tail->time));
    }

    return pow(g,2)*DiagWeight*exp(-w0*(endTime - inTime));
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

    // select the emission vertex of that phonon line
    // if it's the last phonon line (highest index), we can jump directly via tail->link
    if(lineNum == vrtxCount/2){
        inVertex = tail->link;
    } else {
        int num = 1;
        inVertex = head;
        while(num < lineNum){
            inVertex = inVertex->next;
            if(inVertex->sign > 0){
                num++;
            }
        }
    }

    // compute ratio from electron propagators between inVertex and its partner
    double diagWeight = 1.0;
    double q = inVertex->q;
    Vertex* temp = inVertex; // start from emission vertex
    while(temp != inVertex->link){
        diagWeight *= exp(-(E(temp->kOut + q) - E(temp->kOut))
                          *(temp->next->time - temp->time));
        temp = temp->next;
    }

    return diagWeight
         * exp(w0*(inVertex->link->time - inVertex->time))
         / (pow(g,2));
}

// getTauWeight:
// Metropolis factor for changing only the final tau
// electron just propagates freely for the additional tail
double Diag::getTauWeight(double tauNew){
    return exp(-E(this->k)*(tauNew - this->tau));
}

// getStretchWeight:
// Metropolis factor for rescaling the whole diagram from tau to tauNew
// this multiplies every time interval by (1+strain), we accumulate
// contributions from electron and phonon propagators accordingly
double Diag::getStretchWeight(double tauNew){
    double strain = tauNew/(this->tau) - 1;

    if(vrtxCount == 0){
        return exp(-E(this->k)*(this->tau)*strain);
    } 

    // first vertex contribution:
    //   electron from 0->head->time
    //   + phonon line starting at head if head is an emission
    double weight = exp((-E(this->k)*(head->time)
                          - w0*(head->link->time - head->time))*strain);

    Vertex* temp = head->next;
    while(temp){
        // electron free propagator between previous and current vertex
        weight *= exp(-E(temp->kIn)
                      *(temp->time - temp->prev->time )*strain);

        // if this vertex is an emission (sign>0) start of a phonon line,
        // include its phonon propagator weight
        if(temp->sign > 0){
            weight *= exp(-w0*(temp->link->time - temp->time)*strain);
        }
        temp = temp->next;
    }

    // final tail from last vertex to tau
    weight *= exp(-E(this->k)*(this->tau - tail->time)*strain);
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
        cout << "e----[ k=" << this->k << " ]----(t=" << this->tau << ")--->" << endl;
        cout << "=====================================================\n" << endl;
        return;
    }

    cout << "Ordine: " << getOrder()
         << ", #Linee: " << getPhLineNum()
         << ", k_initial: " << this->k
         << ", tau: " << this->tau << endl;
    
    // segmento iniziale prima del primo vertice
    printf("e----[k=%.1f]----", this->k);

    Vertex* temp = head;
    while (temp) {
        // stampa il vertice
        printf("●(t=%.1f, s=%+d, q=%.1f)", temp->time, temp->sign, temp->q);
        
        // stampa il segmento dopo il vertice
        printf("----[k=%.1f]----", temp->kOut);
        
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
// at each vertex: kIn - kOut should match q (sign included)
// and along electron segments: kOut(previous) == kIn(next)
bool Diag::testConservation() const{
    Vertex* temp = head;
    double tol = 1e-9;
    bool state = true;
    while(temp){
        double err = abs(temp->kIn - temp->kOut - temp->q);
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
        temp = temp->next;
    }
    return state;
}

// getFullWeight:
// compute the full diagram weight W for the current configuration
// product of:
//   electron propagators between vertices
//   phonon propagators for each emitted phonon line
//   coupling constants g^2 for each emission
double Diag::getFullWeight() const{
    if(vrtxCount==0){
        return exp(-E(this->k)*(this->tau));
    }

    // first segment: electron from 0 to head->time
    // then the first phonon line (head is assumed emission)
    double diagWeight =
        exp(-E(this->k)*(head->time))
        * pow(g,2)
        * exp(-w0*(head->link->time - head->time ));

    // now iterate over remaining vertices
    // for each interval, multiply electron propagator
    // if the vertex is an emission (sign>0), multiply g^2 * phonon propagator
    Vertex* temp = head->next;
    while(temp){
        diagWeight *= exp(-E(temp->kIn)
                           *(temp->time - temp->prev->time));
        if(temp->sign > 0){
            diagWeight *= pow(g,2)
                        * exp(-w0*(temp->link->time - temp->time ));
        }
        temp = temp->next;
    }

    // final tail from last vertex to tau
    diagWeight *= exp(-E(this->k)*(this->tau - tail->time)) ;
 
    return diagWeight;
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
