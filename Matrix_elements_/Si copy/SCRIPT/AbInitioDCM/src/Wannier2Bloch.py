import numpy as np
from matplotlib import pyplot as plt
from SVD import getLowRank
RY_TO_EV    = 13.605698066
RY_TO_THZ = 3289.84498
# Silicon Mass in Ry units = AMU * (Mass_Electron_AU / 2) 
# 28.0855 * 911.444 = 25598.3
mass_factor = 25598.3

# HAMILTONIAN
def diagonalize_Hw(H_w):
    """
    H_w: (n_orb, n_orb) or (n_k, n_orb, n_orb) hermitian complex array
    Returns:
      evals: shape (n_orb,) or (n_k, n_orb) (real)
      U:     shape (n_orb, n_orb) or (n_k, n_orb, n_orb)  (columns are eigenvectors)
    """
    if H_w.ndim == 2:
        # single k
        evals, U_wan = np.linalg.eigh(H_w)
        return evals, U_wan
    elif H_w.ndim == 3:
        # batched: numpy supports stacked eigh
        evals, U_wan = np.linalg.eigh(H_w)   # evals: (n_k, n_orb), U: (n_k, n_orb, n_orb)
        return evals, U_wan

#RETURN in EV!!
def interpolate_H(H_wan, K ,R, plotBand = False):
    exp1 = np.exp(1j* K @ R.T)
    if(K.ndim==1): 
        n_kp = 1
        exp1 = exp1.reshape(1,-1)
        
    else:
        n_kp = K.shape[0]
    H_w = np.einsum("kR, Rij -> kij",exp1,H_wan,optimize=True)
    #make the matrix for each k hermitian:
    H_w = 0.5*(H_w + H_w.conj().transpose(0,2,1))
    evals,U_wan = diagonalize_Hw(H_w)
    return evals*RY_TO_EV,U_wan

def plotBands(x_axis,nbands,evals,xticks_positions,xticks_labels,outdir=""):
    fig = plt.figure()
    ax = fig.gca()

    for i in range(nbands):
        ax.plot(x_axis, evals[:, i],color="Blue")

    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(xticks_labels)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("Band Interpolation on a Line")
    plt.ylabel("E (eV)")
    plt.show()
    fig.savefig(outdir+"Bands.jpg")

#Dynamical Matrix

def get_frequencies(w_squared):
    sign = np.sign(w_squared)
    w_abs = np.abs(w_squared)
    w_ry = np.sqrt(w_abs) * sign
    freqs = w_ry * RY_TO_THZ
    return freqs

def diagonalize_Dw(D_w):
    if D_w.ndim == 2:
        w, e = np.linalg.eigh(D_w)
        return w, e
    elif D_w.ndim == 3:
        w, e = np.linalg.eigh(D_w)
        return w, e

def interpolate_D(D_wan, K ,R, plotBand = False):
    exp1 = np.exp(1j* K @ R.T)
    if(K.ndim==1): 
        n_kp = 1
        exp1 = exp1.reshape(1,-1)
    else:
        n_kp = K.shape[0]
        
    D_w = np.einsum("kR, Rij -> kij",exp1,D_wan,optimize=True)
    
    # Apply Mass Division (Force Constants -> Dynamical Matrix)
    D_w /= mass_factor
    
    D_w = 0.5*(D_w + D_w.conj().transpose(0,2,1))
    
    # Corrected function call name from Hw to Dw
    w2,e = diagonalize_Dw(D_w)
    w = get_frequencies(w2)
    return w,e

def plotModes(x_axis, nmodes, w, xticks_positions, xticks_labels,outdir=""):   
    fig=plt.figure()
    ax = fig.gca()

    for i in range(nmodes):
        ax.plot(x_axis, w[:, i], color="blue")

    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(xticks_labels)
    ax.set_ylabel("Frequency (THz)")
    ax.set_title("Modes interpolation")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    fig.savefig(outdir+"modes.jpg")

# Interpolate G

# Get the fourier transform of left and right SV at specific k and q points
#k and q can either be a single k-point or a list (n_kp,3)
def fourier_transform(U,Vt,q,k,R):
    exp_k = np.exp( 1j * k @ R.T)
    if(k.ndim==1): exp_k = exp_k.reshape(1,-1)

    exp_q = np.exp(1j* q @ R.T)
    if(q.ndim==1): exp_q = exp_q.reshape(1,-1)

    U_k = np.einsum("kR, Rrvij -> krvij", exp_k, U,optimize = True)
    #print("Left SV matrix shape:",U_k.shape)
    Vt_q = np.einsum("qR, rRvij -> rqvij", exp_q, Vt,optimize = True)
    #print("Right SV matrix shape:", Vt_q.shape)

    return U_k,Vt_q

def fourierANDlr(U,Vt,s,q,k,R):
    exp_k = np.exp( 1j * k @ R.T)
    if(k.ndim==1): exp_k = exp_k.reshape(1,-1)

    exp_q = np.exp(1j* q @ R.T)
    if(q.ndim==1): exp_q = exp_q.reshape(1,-1)

    return np.einsum("qR, kS, rRvij, Srvij, rvij -> kqvij", 
                     exp_q,
                     exp_k,
                    Vt,
                    U,
                    s,
                    optimize = True)

#From a Fourier rotated matrix makes the final interpolation
def Wannier2Bloch_rotate(g_FT,U_kq,U_kdag,e_q):
  return np.einsum('qmi,kqaji,qav,kjn->kqvnm', 
                  U_kq, 
                  g_FT,
                  e_q,
                  U_kdag, 
                    optimize=True)

#Main of the module calling all the relevant fucntions
def Wannier2Bloch(H_sort,D_sort,G_wan,U,Vt,s,R,q_points,k_points):
    Ek,U_k = interpolate_H(H_sort,k_points,R)
    U_kdag = np.conjugate(U_k).transpose(0,2,1)
    wq,e_q = interpolate_D(D_sort,q_points,R)
    if(k_points.ndim == 1):
        kPq_points = q_points + np.ones(q_points.shape)*k_points
        Ekq,U_kq = interpolate_H(H_sort,kPq_points,R)
    elif(k_points.shape != q_points): 
        print("Error, if you more than one kpoint is provide, q and k must have the same shape")
        return
    else: 
        kPq_points = q_points + k_points
        Ekq,Ukq = interpolate_H(H_sort,kPq_points,R)
    
    """U_k,Vt_q = fourier_transform(U,Vt,q_points,k_points,R)
    G_FT = getLowRank(U_k,s,Vt_q)"""
    G_FT = fourierANDlr(U,Vt,s,q_points,k_points,R)
    
    G_bloch = Wannier2Bloch_rotate(G_FT,U_kq,U_kdag,e_q)
    
    return G_bloch*RY_TO_EV,wq,Ek

def BENCHMARK(H_sort,D_sort,G_wan,R,q_points,k_points):
    Ek,U_k = interpolate_H(H_sort,k_points,R)
    U_kdag = np.conjugate(U_k).transpose(0,2,1)
    wq,e_q = interpolate_D(D_sort,q_points,R)
    if(k_points.ndim == 1):
        kPq_points = q_points + np.ones(q_points.shape)*k_points
        Ekq,U_kq = interpolate_H(H_sort,kPq_points,R)
    elif(k_points.shape != q_points): 
        print("Error, if you more than one kpoint is provide, q and k must have the same shape")
        return
    else: 
        kPq_points = q_points + k_points
        Ekq,Ukq = interpolate_H(H_sort,kPq_points,R)
    
    exp_k = np.exp( 1j * k_points @ R.T)
    if(k_points.ndim==1): exp_k = exp_k.reshape(1,-1)

    exp_q = np.exp(1j* q_points @ R.T)
    if(q_points.ndim==1): exp_q = exp_q.reshape(1,-1)
    # S is Re R is Rp
    G_FT = np.einsum("kS,qR, SRvij -> kqvij", 
                     exp_k, 
                     exp_q,
                     G_wan,
                     optimize = True)
    
    G_bloch = Wannier2Bloch_rotate(G_FT,U_kq,U_kdag,e_q)
    
    return G_bloch*RY_TO_EV ,wq,Ek



