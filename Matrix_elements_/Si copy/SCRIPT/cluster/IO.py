import numpy as np
import re
import os
from matplotlib import pyplot as plt
from tools import generate_R

def sort(R,H,D,plotDecay = False,outdir=""):
    dist = np.linalg.norm(R, axis=1)
    sorted_id = np.argsort(dist)
    R_sort = R[sorted_id]
    dist = dist[sorted_id]
    H_sort = H[sorted_id]
    D_sort = D[sorted_id]
    if(plotDecay):
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
        fig.savefig(outdir+"HD_Decay.jpg")
    return R_sort,H_sort,D_sort

def read_epwdata_fmt(fmt_file,at,k_mesh,plotDecay = False,outdir=""):
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
    
    R = generate_R(k_mesh,nrr_k,at)
    R_sort, H_sort, D_sort = sort(R,H,D,plotDecay,outdir)
    return H_sort, D_sort,R_sort,R ,ef ,nbnd, nrr_k, nmodes, nrr_q, nrr_g

def read_epmatwp(bin_file, nbnd, nrr_k, nmodes, nrr_g,R,PLOT=False,outdir=""):
    print(f"Reading {bin_file}...")

    # 1. Read Raw Bytes
    # Complex128 = 16 bytes per number
    G_raw = np.fromfile(bin_file, dtype=np.complex128)

    # 2. Reshape 
    # Fortran Order: (Band, Band, Re, Mode, Rp)
    G_ftn = G_raw.reshape((nbnd, nbnd, nrr_k, nmodes, nrr_g), order='F')

    # 3. Transpose to Python Logic
    # New Order: (Re, Rp, Mode, Band, Band)
    G = G_ftn.transpose(2, 4, 3, 0, 1)
    dist_G = np.linalg.norm(R, axis=1)
    sorted_id = np.argsort(dist_G)
    dist_sorted = dist_G[sorted_id]
    G_sort = G[sorted_id]
    G_sort = G_sort[:, sorted_id]
    print(f"G shape loaded: {G.shape}")
    if(PLOT):
        
        # Case A: Electronic Decay (Phonon fixed at R=0)
        # We look at G[0, :, :, :, :] -> varies with Re
        G_el = G_sort[:, 0, :, :, :]
        decay_el = np.array([np.max(np.abs(G_el[i,...]))for i in range(G_el.shape[0])])
        # Case B: Phonetic Decay (Electron fixed at R=0)
        # We look at G[:, 0, :, :, :] -> varies with Rp
        G_ph = G_sort[ 0,:, :, :, :]
        decay_ph = np.array([np.max(np.abs(G_ph[i,...]))for i in range(G_ph.shape[0])])
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # Electronic
        ax[0].scatter(dist_sorted, decay_el, color='royalblue', lw=1)
        ax[0].set_title(r"Electronic Decay ($R_p=0$)")
        ax[0].set_xlabel(r"Distance $|\mathbf{R}_e|$ ($\AA$)")
        ax[0].set_ylabel("Magnitude (Ry)")
        ax[0].grid(True, alpha=0.3)

        # Phonetic
        ax[1].scatter(dist_sorted, decay_ph, color='crimson', lw=1)
        ax[1].set_title(r"Phonetic Decay ($R_e=0$)")
        ax[1].set_xlabel(r"Distance $|\mathbf{R}_p|$ ($\AA$)")
        ax[1].grid(True, alpha=0.3)

        plt.yscale("log")
        plt.tight_layout()
        plt.show()
        fig.savefig(outdir+"matrixDecay.jpg")
    return G_sort

import numpy as np
from scipy.io import FortranFile

def read_epb_mixed_record(filename, nkc, nbnd, nmodes):
    """
    Reads a mixed-type EPW .epb file written as a single record.
    
    Structure:
      nqc      (int32)
      xqc      (3, nqc) float64
      et_loc   (nbnd, nkc) float64
      dynq     (nmodes, nmodes, nqc) complex128
      epmatq   (nbnd, nbnd, nmodes, nkc, nqc) complex128
    """
    print(f"Reading {filename}...")
    
    # 1. Read the full record as raw bytes
    f = FortranFile(filename, 'r')
    # Read everything as bytes (int8/byte)
    raw_bytes = f.read_record(dtype=np.byte)
    f.close()
    
    print(f"  - Total Record Bytes: {raw_bytes.size}")
    
    # 2. Setup a buffer iterator
    # We use offsets to slice the buffer
    offset = 0
    
    # Helper to read chunk
    def read_chunk(dtype, count):
        nonlocal offset
        item_size = np.dtype(dtype).itemsize
        total_bytes = count * item_size
        
        # Extract bytes
        chunk_bytes = raw_bytes[offset : offset + total_bytes]
        offset += total_bytes
        
        # Cast to correct type
        data = np.frombuffer(chunk_bytes, dtype=dtype)
        
        if data.size != count:
            raise ValueError(f"Read error: expected {count} items, got {data.size}")
        return data

    # --- A. Read nqc ---
    nqc_arr = read_chunk(np.int32, 1)
    nqc = nqc_arr[0]
    print(f"  - nqc (q-points in file): {nqc}")
    
    # --- B. Read xqc ---
    # Shape: (3, nqc)
    xqc = read_chunk(np.float64, 3 * nqc).reshape((3, nqc), order='F')
    
    # --- C. Read et_loc (Eigenvalues) ---
    # Shape: (nbnd, nkc)
    et_loc = read_chunk(np.float64, nbnd * nkc).reshape((nbnd, nkc), order='F')
    
    # --- D. Read dynq (Dynamical Matrix) ---
    # Shape: (nmodes, nmodes, nqc)  where nmodes = 3*natoms
    dynq = read_chunk(np.complex128, nmodes * nmodes * nqc).reshape((nmodes, nmodes, nqc), order='F')
    
    # --- E. Read epmatq (El-Ph Matrix) ---
    # Fortran Shape: (nbnd, nbnd, nmodes, nkc, nqc)
    # This is the massive array
    epmatq_size = nbnd * nbnd * nmodes * nkc * nqc
    print(f"  - Reading epmatq ({epmatq_size} complex elements)...")
    
    epmatq_flat = read_chunk(np.complex128, epmatq_size)
    epmatq_ftn = epmatq_flat.reshape((nbnd, nbnd, nmodes, nkc, nqc), order='F')
    
    # Transpose to Python friendly: (nqc, nkc, nmodes, nbnd, nbnd)
    # (4, 3, 2, 0, 1)
    epmatq = epmatq_ftn.transpose(4, 3, 2, 0, 1)
    
    
    # Check if we consumed all bytes
    remaining = raw_bytes.size - offset
    if remaining != 0:
        print(f"  ! Warning: {remaining} bytes remaining in record (padding?).")
    else:
        print("  - Successfully read all bytes.")
        
    return epmatq,xqc


