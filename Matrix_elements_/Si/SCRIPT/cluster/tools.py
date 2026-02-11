import numpy as np
from matplotlib import pyplot as plt
from Wannier2Bloch import Wannier2Bloch,BENCHMARK
import time

def generate_R(k_mesh,nrr_k,at):
    nk = k_mesh
    # REPLICATE WIGNER.F90 LOOP ORDER
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
                    found_vectors.append( (n1*at[:,0] + n2*at[:,1]+ n3*at[:,2]) )
    print(f"\nTotal vectors found: {len(found_vectors)} (Should be {nrr_k})")
    return np.array(found_vectors) 


def get_reciprocal_lattice(at):
    """Computes b_vectors such that a_i . b_j = 2pi delta_ij"""
    # 'A' columns are a1, a2, a3. 
    # b_matrix columns will be b1, b2, b3
    # b_matrix = 2pi * (at^-1).T
    return 2 * np.pi * np.linalg.inv(at).T


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

def plot_EPC(g_bloch_full,g_bloch_perc,w_full,w_perc,nbnd,nmodes,x_axis,x_ticks,x_ticks_labels,perc,outdir = ""):
    atomic_mass = 1.66053906660e-27

    POMASS = 28.085
    mass = 2 * POMASS * atomic_mass   # kg, adjust if unit cell has different number of atoms

    # make sure w is in THz
    w_full_rad = w_full * 2.0 * np.pi * 1e12   # rad/s
    w_perc_rad = w_perc * 2.0 * np.pi * 1e12   # rad/s
    g2_full = 2.0 * np.sum(np.abs(g_bloch_full[0])**2, axis=(-2, -1)) / nbnd   # already did 2/Nb
    g2_perc = 2.0 * np.sum(np.abs(g_bloch_perc[0])**2, axis=(-2, -1)) / nbnd

    # convert g^2 from (Ry)^2 to J^2

    # according to eq. (9): D = (1/ħ) * sqrt( 2 ω M_uc * (1/Nb) * sum |g|^2 )
    # here g2_full_J2 already contains the 2/Nb factor, so:
    sqrt_full = np.sqrt(np.abs(mass * w_full_rad * g2_full))   # sqrt(2 ω M_uc * sum|g|^2 / Nb)
    sqrt_2perc = np.sqrt(np.abs(mass * w_perc_rad * g2_perc))

    D = sqrt_full     # now divide by ħ outside the sqrt
    D_2perc = sqrt_2perc 

    # convert from J/m to eV/Å if the quantity is J/m — check units:
    # after the algebra above D should be in units of J/m, so convert:
    D = D  * 1e5
    D_2perc = D_2perc * 1e5

    # plot
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x_axis, D[:, 0], label="full", color="blue",linewidth=2)
    ax.plot(x_axis, D_2perc[:, 0], label=f"{perc*100}%",color ="orange", linewidth=2,alpha = 0.9)
    for v in range(1, nmodes):
        ax.plot(x_axis, D[:, v], color="blue", linewidth=2)
        ax.plot(x_axis, D_2perc[:, v],color ="orange", linewidth=2,alpha = 0.9)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels)
    ax.set_title("E-P coupling Reconstruction")
    ax.legend()
    fig.savefig(outdir+"EPcoupling.jpg")

def get_random_k_primitive(b_lat, n_points=1000):
    """
    Generates random k-points in the primitive reciprocal cell (parallelepiped).
    b_lat: (3,3) matrix where columns are b1, b2, b3
    """
    # 1. Generate random fractional coordinates [-0.5, 0.5)
    k_frac = np.random.rand(n_points, 3) - 0.5
    
    # 2. Convert to Cartesian
    # Shape: (N, 3) @ (3, 3).T -> (N, 3)
    k_cart = k_frac @ b_lat.T 
    
    return k_cart

def performance_analysis(N,b_lat,H_sort,D_sort,G,R_sort,U,Vt,s,outdir=""):

    perc = np.hstack([np.linspace(1,9,15),np.linspace(10,100,7)])
    points = perc.shape[0]
    ranks = np.array(s.shape[0]*perc/100,dtype = int)
    time_counter = np.zeros(perc.shape) 
    errors = np.zeros(perc.shape)
    norm = 0
    K = get_random_k_primitive(b_lat,N)
    tot_time = time.time()
    for i_k in range(N):
        if(i_k%100==0): print(time.time()-tot_time,"s | ",i_k)
        start_time = time.time()
        g,_,_ = BENCHMARK(H_sort,D_sort,G,R_sort,K[i_k],np.array([0,0,0]))
        time_counter[-1]+= time.time()-start_time
        norm += np.sum(np.abs(g)**2)
        for i_r in range(points-1):
            U_red = U[:,:ranks[i_r]]
            Vt_red = Vt[:ranks[i_r]]
            s_red = s[:ranks[i_r]]
            start_time = time.time()
            g_r,_,_ = Wannier2Bloch(H_sort,D_sort,G,U_red,Vt_red,s_red,R_sort,K[i_k],np.array([0,0,0]))
            time_counter[i_r]+= time.time()-start_time 
            errors[i_r] += np.sum(np.abs(g_r - g)**2)

    errors = errors*100/norm
    
    fig, axes = plt.subplots(1,2 , figsize=(15, 6))
    axes[0].plot(perc[:-1],time_counter[:-1],"o-")
    axes[0].set_title(f"Time to interpolate {N} points")
    axes[0].set_ylabel("seconds")
    axes[0].set_xlabel("%SV")
    speedUp = time_counter[-1]/time_counter[:-1]
    axes[1].plot(perc[:-1],speedUp,"o-")
    axes[1].set_title("SpeedUp")
    axes[1].set_ylabel("x Full")
    axes[1].set_xlabel("%SV")
    plt.show()
    fig.savefig(outdir+f"performance_N={N}.jpg")

    fig, axes = plt.subplots(1,2 , figsize=(15, 6))
    axes[0].plot(perc,errors,"o-")
    axes[0].set_title("Relative Error")
    axes[0].set_ylabel("ERROR %")
    axes[0].set_xlabel("%SV")
    plt.grid()
    axes[1].plot(perc[:16],errors[:16],"o-")
    axes[1].set_title("Zoom Relative Error")
    axes[1].set_ylabel("ERROR %")
    axes[1].set_xlabel("%SV")
    plt.grid()
    plt.show()
    fig.savefig(outdir+"error.jpg")
    
    return perc,time_counter,errors






   
