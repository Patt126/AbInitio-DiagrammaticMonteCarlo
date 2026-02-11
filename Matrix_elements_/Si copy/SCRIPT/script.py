import numpy as np
import re
import os
from matplotlib import pyplot as plt
from scipy.linalg import svdvals, svd

def read_epwdata_fmt(fmt_file):
    """
    Reads H (Hamiltonian), D (Dynamical Matrix), and G (El-Ph Matrix)
    """
    print(f"Reading Text Data: {fmt_file}")
    with open(fmt_file, 'r') as f:
        raw = f.read() #string containing all the file RN!

    # 1. Parse Dimensions from Header
    # Remove complex (x,y), care later
    header = re.sub(r'\([^)]+\)', '', raw).split()
    #print(header)
    
    # Map based on epw_write source code
    ef = float(header[0])
    nbnd   = int(header[1])
    nrr_k = int(header[2]) # Electronic grid size
    nmodes = int(header[3])
    nrr_q = int(header[4]) # Phonon grsorted_id size
    nrr_g = int(header[5]) # El-Ph grsorted_id size (usually same as nrws_q)

    print(f"Dims: Bands={nbnd}, Modes={nmodes}, R_k={nrr_k}, R_g={nrr_g}, R_q={nrr_q}")

    # 2. Parse H and D (Text)
    # Find all (Real, Imag) pairs
    # Matsches all sequence in the form ( + any space [numbers . -+ E any space] , [numbers . -+ E any space] any space)
    # NB * is important because says any number including 0!
    # + says that we want to find one of more character in the group [ ]
    # () arounf []+ capture the entire group, so that matches becomes a tuple of captured element
    # with no ([...]+)  and only [] i capture the whole string '(...,...)'
    matches = re.findall(r'\(\s*([0-9.\-\+E]+)\s*,\s*([0-9.\-\+E]+)\s*\)', raw)
    #print(matches)
    data_complex = np.array([complex(float(r), float(i)) for r, i in matches])
        
    # H = H[Re,i,j]
    d = 0 #d index
    H = np.zeros((nrr_k,nbnd,nbnd),dtype=complex)
    for i in range(nbnd):
        for j in range(nbnd):
            for R in range(nrr_k):
                H[R,i,j] = data_complex[d]
                d+=1
    
    # D = H[Rp,a,b]
    D = np.zeros((nrr_q,nmodes,nmodes),dtype=complex)
    for a in range(nmodes):
        for b in range(nmodes):
            for Rp in range(nrr_q):
                D[Rp,a,b] = data_complex[d]
                d+=1
    
    if(d == data_complex.shape[0]):
        print("!All the file has been red!")
    else:
        print("WARNING: some lines are missing", data_complex.shape[0],d)
    return H,D,ef ,nbnd, nrr_k, nmodes, nrr_q, nrr_g
H,D,ef ,nbnd, nrr_k, nmodes, nrr_q, nrr_g = read_epwdata_fmt('../DATA/epwdata.fmt', )

NK = [6,6,6]  
alat = 10.26    # Lattice param (Bohr) - from scf.in
# FCC Lattice vectors
at = (alat / 2.0) * np.array([[-1., 0., 1.], [0., 1., 1.], [-1., 1., 0.]]).T

# REPLICATE WIGNER.F90 LOOP ORDER
def generate_raw_scan_order(nk, at):
    print("Replicating wigner.f90 scan loops...")
    
    # Metric tensors for distance check
    T_metric = np.zeros((3,3))
    for i in range(3):
        T_metric[:, i] = at[:, i] * nk[i]

    # The list that will hold the vectors IN THE ORDER THEY ARE FOUND
    found_vectors = []
    
    # EPW Scan Range: -2*N to +2*N (Lines 218-220 in wigner.f90)
    # Crucial: The loops are n1 (outer), n2, n3 (inner)
    
    range_1 = range(-2 * nk[0], 2 * nk[0] + 1)
    range_2 = range(-2 * nk[1], 2 * nk[1] + 1)
    range_3 = range(-2 * nk[2], 2 * nk[2] + 1)

    for n1 in range_1:
        for n2 in range_2:
            for n3 in range_3:
                
                # 1. Calculate Distance of current point R
                R = n1*at[:,0] + n2*at[:,1] + n3*at[:,2]
                dist_R = np.linalg.norm(R)
                
                # 2. Wigner-Seitz Check (Is it the closest image?)
                # We check against 125 supercell neighbors
                is_in_ws = True
                
                # Optimization: Check origin image immediately to save time
                # If dist_R is huge, it's likely not in WS
                
                for i1 in range(-2, 3):
                    for i2 in range(-2, 3):
                        for i3 in range(-2, 3):
                            if i1==0 and i2==0 and i3==0: continue
                            
                            T = i1*T_metric[:,0] + i2*T_metric[:,1] + i3*T_metric[:,2]
                            dist_img = np.linalg.norm(R - T)
                            
                            if dist_img < dist_R - 1e-4:
                                is_in_ws = False
                                break # Not a WS point
                    if not is_in_ws: break
                
                # 3. If valsorted_id, APPEND to list (NO SORTING)
                if is_in_ws:
                    found_vectors.append( (n1, n2, n3) )

    return np.array(found_vectors)

R = generate_raw_scan_order(NK, at)

print(f"\nTotal vectors found: {len(R)} (Should be {nrr_k})")

dist = np.linalg.norm(R, axis=1)


sorted_id = np.argsort(dist)
R_sort = R[sorted_id]
dist = dist[sorted_id]
H_sort = H[sorted_id]
D_sort = D[sorted_id]
print(H.shape)

values_H = np.array([np.max(np.abs(H_sort[i,...]))for i in range(H_sort.shape[0])])
values_D = np.array([np.max(np.abs(D_sort[i,...]))for i in range(D_sort.shape[0])])

fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Electronic
ax[0].scatter(dist, values_H, color='royalblue', lw=1)
ax[0].set_title("Hamiltonian Decay")
ax[0].set_xlabel(r"Distance $|\mathbf{R}_e|$ ($\AA$)")
ax[0].set_ylabel("Magnitude (Ry)")
ax[0].grid(True, alpha=0.3)

# Phonetic
ax[1].scatter(dist, values_D, color='crimson', lw=1)
ax[1].set_title("Dynamical Matrix Decay")
ax[1].set_xlabel(r"Distance $|\mathbf{R}_p|$ ($\AA$)")
ax[1].grid(True, alpha=0.3)

plt.yscale("log")
plt.tight_layout()
plt.show()

# Constants
RY_TO_EV    = 13.605698066
BOHR_TO_ANG = 0.52917721
AMASS_AMU   = 28.0855 # Silicon Mass
MASS_AU     = AMASS_AMU * 1822.888 # Electron mass units

G = np.array([0,0,0]) #gamma 
L = np.array([0.5,0.5,0.5])
X = np.array([0.5, 0.0, 0.5])
K = np.array([0.375, 0.3750, 0.75])
U = np.array([0.625, 0.250, 0.625])
W = np.array([ 0.5000000000,0.2500000000,0.7500000000])
path = np.array([[K,G],[G,L],[L,W],[W,X]])
xticks_labels = ['K','G','L','W','X']


def get_reciprocal_lattice(A):
    """Computes b_vectors such that a_i . b_j = 2pi delta_ij"""
    # 'A' columns are a1, a2, a3. 
    # b_matrix columns will be b1, b2, b3
    # b_matrix = 2pi * (at^-1).T
    return 2 * np.pi * np.linalg.inv(A).T

import numpy as np

def build_path_and_ticks(b_lat, path, steps=30):
    """
    b_lat : (3,3) reciprocal-lattice matrix (columns = b1,b2,b3)
    path  : array-like of shape (n_seg,2,3) with fractional endpoints in crystal coords
    steps : points per segment (including both ends)
    Returns:
      q_cart   : (N,3) Cartesian q points (1/Bohr)
      x_axis   : (N,) accumulated Cartesian distance
      xticks   : list of positions along x_axis where segment endpoints lie (length = n_seg+1)
    """
    # Build full path in fractional coordinates, avoid duplicating segment endpoints
    full_frac = []
    seg_endpoint_indices = []
    idx = 0
    for iseg, (a_frac, b_frac) in enumerate(path):
        seg = np.linspace(a_frac, b_frac, steps)  # includes both endpoints
        if iseg == 0:
            full_frac.extend(seg.tolist())
            idx += seg.shape[0]
            seg_endpoint_indices.append(idx - 1)
        else:
            seg = seg[1:]               # drop duplicate start point
            full_frac.extend(seg.tolist())
            idx += seg.shape[0]
            seg_endpoint_indices.append(idx - 1)

    full_frac = np.array(full_frac)               # (N,3)
    q_cart = (b_lat @ full_frac.T).T              # (N,3) Cartesian q

    # accumulate Cartesian distances
    x_axis = np.zeros(q_cart.shape[0], dtype=float)
    for i in range(1, q_cart.shape[0]):
        x_axis[i] = x_axis[i-1] + np.linalg.norm(q_cart[i] - q_cart[i-1])

    # xticks positions: start (0.0) and each segment endpoint
    xticks = [0.0] + [float(x_axis[idx]) for idx in seg_endpoint_indices]

    return q_cart, x_axis, xticks

b_lat = get_reciprocal_lattice(at)
q_points, x_axis, xticks_positions = build_path_and_ticks(b_lat, path, steps=30)

fig, ax = plt.subplots()
ax.scatter(x_axis, np.ones_like(x_axis)*0.5, s=6)
ax.scatter(xticks_positions, np.ones(len(xticks_positions))*0.5, color='red')
ax.set_xticks(xticks_positions)
ax.set_xticklabels(xticks_labels)
ax.set_ylim(0,2)
plt.show()
# K can either be a single k point or a collection with shape (n_R , 3)
def check_Hermitian(A):
    A_H = np.conjugate(A).transpose(0,2,1) # each A[k] is a matrix to check actually
    diff = np.abs(A_H -A)
    print("Max: ",np.max(diff)/np.max(np.abs(A)))
    print("Mean: ",np.mean(diff)/np.max(np.abs(A)))
print(H.shape)

def diagonalize_Hw(H_w):
    """
    H_w: (n_orb, n_orb) or (n_k, n_orb, n_orb) hermitian complex array
    Returns:
      evals: shape (n_orb,) or (n_k, n_orb) (real)
      U:     shape (n_orb, n_orb) or (n_k, n_orb, n_orb)  (columns are eigenvectors)
    """
    if H_w.ndim == 2:
        # single k
        evals, U = np.linalg.eigh(H_w)
        return evals, U
    elif H_w.ndim == 3:
        # batched: numpy supports stacked eigh
        evals, U = np.linalg.eigh(H_w)   # evals: (n_k, n_orb), U: (n_k, n_orb, n_orb)
        return evals, U

def interpolate_H(H_wan, K ,R, plotBand = False):
    exp1 = np.exp(1j* K @ R.T)
    if(K.ndim==1): 
        n_kp = 1
        exp1 = exp1.reshape(1,-1)
        
    else:
        n_kp = K.shape[0]
    print(exp1.shape)
    H_w = np.einsum("kR, Rij -> kij",exp1,H_wan,optimize=True)
    #make the matrix for each k hermitian:
    H_w = 0.5*(H_w + H_w.conj().transpose(0,2,1))
    evals,U = diagonalize_Hw(H_w)
    return evals,U
    

evals, U = interpolate_H(H_sort, q_points, R_sort)



evals, U = interpolate_H(H_sort, q_points, R_sort)

plt.figure()
ax = plt.gca()

for i in range(4):
    ax.plot(x_axis, evals[:, i]*RY_TO_EV)

ax.set_xticks(xticks_positions)
ax.set_xticklabels(xticks_labels)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
