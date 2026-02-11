import numpy as np
from matplotlib import pyplot as plt
from SVD import getLowRank
import py4vasp

RY_TO_EV    = 13.605698066
RY_TO_THZ = 3289.84498

# AMU to Ry mass units conversion: AMU * 911.444
M_Li = 6.941 * 911.444
M_F  = 18.998 * 911.444

# Create a mass array for the 6 degrees of freedom (Li_x, Li_y, Li_z, F_x, F_y, F_z)
masses = np.array([M_Li, M_Li, M_Li, M_F, M_F, M_F])
# Pre-calculate the square root mass matrix
mass_map = np.sqrt(np.outer(masses, masses))

#k_cart: (..., 3) Cartesian
#b_lat: (3,3) columns are reciprocal vectors b1,b2,b3
#returns: (..., 3) fractional such that k_cart = k_frac @ b_lat.T
def cart_to_frac(k_cart, b_lat):
    k_cart = np.asarray(k_cart, dtype=float)
    Binv = np.linalg.inv(b_lat)
    return k_cart @ Binv.T

def frac_to_cart(k_frac, b_lat):
    k_frac = np.asarray(k_frac, dtype=float)
    return k_frac @ b_lat.T

#Fold to [-0.5, 0.5) componentwise in 1bz
def fold_frac_pmhalf(k_frac):
    k_frac = np.asarray(k_frac, dtype=float)
    return k_frac - np.round(k_frac)

#Fold Cartesian
def fold_k_cart_primitive(k_cart, b_lat):
    k_frac = cart_to_frac(k_cart, b_lat)
    k_frac = fold_frac_pmhalf(k_frac)
    return frac_to_cart(k_frac, b_lat)

# HAMILTONIAN
#H_w: (n_orb, n_orb) or (n_k, n_orb, n_orb) hermitian complex array
#Returns: evals: shape (n_orb,) or (n_k, n_orb) (real)
#U:     shape (n_orb, n_orb) or (n_k, n_orb, n_orb)  (columns are eigenvectors)
def diagonalize_Hw(H_w):
    if H_w.ndim == 2:
        # single k
        evals, U_wan = np.linalg.eigh(H_w)
        return evals, U_wan
    elif H_w.ndim == 3:
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

# Plot with VASP output data
def plotBands_comparded(x_axis,nbands,evals,xticks_positions,xticks_labels,vaspdir="",outdir=""):
    if vaspdir!="":
        calc = py4vasp.Calculation.from_path(vaspdir)
        d = calc.band.to_dict()
        bands = np.array(d["bands"])          # (nk, nb) or (spin, nk, nb)
        occ   = np.array(d["occupations"])
        x     = np.array(d["kpoint_distances"])
        x *= (max(x_axis))/max(x) #rescale x
        #bands += (bands[0,0]) - (evals[0,0])
        # ---- gestisci spin automaticamente ----
        if bands.ndim == 3:   # spin-polarized
            bands = bands[0]  # prendi spin-up (per Si/LiF non cambia)
            occ   = occ[0]
        # ---- identifica bande di valenza ----
        # una banda è "valence" se è occupata (occ ~ 1) in ALMENO un k-point
        is_valence = np.any(occ > 0.5, axis=0)

        valence_band_indices = np.where(is_valence)[0]
        # ---- ordina le bande di valenza per energia massima (VBM) ----
        vbm_energy_per_band = bands[:, valence_band_indices].max(axis=0)
        order = np.argsort(vbm_energy_per_band)
        top_vb = valence_band_indices[order][-nbands:]
        idx = [0, 40, 80, 120, 160, 200, 239]
        xticks = [x[i] for i in idx]
        
    fig = plt.figure()
    ax = fig.gca()

    ax.plot(x_axis, evals[:, 0],color="Blue",lw=1.6,label="interpolated")
    for i in range(1,nbands):
        ax.plot(x_axis, evals[:, i],color="Blue",lw=1.6)

    if vaspdir!="":
        shift = np.min(evals[:,0]) - np.min(bands[:,top_vb[0]]) #shift to match energy def
        ax.plot(x, bands[:, top_vb[0]] + shift, color="red", lw=1.2, linestyle="--",label="DFT")
        for b in top_vb[1:]:
            ax.plot(x, bands[:, b] + shift, color="red", lw=1.2, linestyle="--")

    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(xticks_labels)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title("Band Interpolation on a Line")
    plt.ylabel("E (eV)")
    plt.legend()
    plt.ylim([-3,3])
    plt.show()
    fig.savefig(outdir+"Bands_compared.jpg")



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
    D_w /= mass_map
    
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

# Plot with VASP output data
def plotModes_compared(x_axis, nmodes, w, xticks_positions, xticks_labels,vasp_dir="",outdir=""):   
    if vasp_dir!="":
        calc = py4vasp.Calculation.from_path(vasp_dir)
        d = calc.phonon_band.to_dict()
        x = np.asarray(d["qpoint_distances"])
        x *= (max(x_axis))/max(x) #rescale x
        bands = np.asarray(d["bands"])
    
    fig=plt.figure()
    ax = fig.gca()
    ax.plot(x_axis, w[:, 0], color="blue",lw=1.6, label = "interpolated")
    for i in range(1,nmodes):
        ax.plot(x_axis, w[:, i], color="blue",lw=1.6)
    
    if vasp_dir!="":
        ax.plot(x, bands[:, 0], color="red", lw=1.2,linestyle="--", label = "DFT")
        for j in range(1,bands.shape[1]):
            ax.plot(x, bands[:, j], color="red", lw=1.2,linestyle="--")


    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(xticks_labels)
    ax.set_ylabel("Frequency (THz)")
    ax.set_title("Modes interpolation")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
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

#perform fourier transform and LR reconstruction togheter
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


#k_points, q_points are Cartesian (Angstrom^-1).
#If folding=True, b_lat must be (3,3) with columns b1,b2,b3 (Angstrom^-1)
#and folding maps to fractional [-0.5,0.5)^3 then back to Cartesian.
def Wannier2Bloch(H_sort, D_sort, G_wan, U, Vt, s, R,
                 q_points, k_points,
                 folding=False, b_lat=None):

    # Optional: fold inputs too 
    if folding:
        if b_lat is None:
            raise ValueError("folding=True requires b_lat (3x3 reciprocal lattice matrix).")
        k_points = fold_k_cart_primitive(k_points, b_lat)
        q_points = fold_k_cart_primitive(q_points, b_lat)

    Ek, U_k = interpolate_H(H_sort, k_points, R)
    U_kdag = np.conjugate(U_k).transpose(0, 2, 1)

    wq, e_q = interpolate_D(D_sort, q_points, R)

    # Build k+q
    if np.ndim(k_points) == 1:
        kPq_points = q_points + k_points  # broadcast
    else:
        if k_points.shape != q_points.shape:
            raise ValueError("If more than one k-point is provided, q_points and k_points must have the same shape.")
        kPq_points = q_points + k_points

    if folding:
        kPq_points = fold_k_cart_primitive(kPq_points, b_lat)

    Ekq, U_kq = interpolate_H(H_sort, kPq_points, R)

    G_FT = fourierANDlr(U, Vt, s, q_points, k_points, R)

    G_bloch = Wannier2Bloch_rotate(G_FT, U_kq, U_kdag, e_q)

    return G_bloch * RY_TO_EV, wq, Ek

# Perform wannier interpolation with full tensor
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
    # S is Re and R is Rp
    G_FT = np.einsum("kS,qR, SRvij -> kqvij", 
                     exp_k, 
                     exp_q,
                     G_wan,
                     optimize = True)
    
    G_bloch = Wannier2Bloch_rotate(G_FT,U_kq,U_kdag,e_q)
    
    return G_bloch*RY_TO_EV ,wq,Ek



