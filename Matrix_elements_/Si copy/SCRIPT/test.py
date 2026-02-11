import numpy as np

def read_epwdata(path='../DATA/epwdata.fmt'):
    with open(path,'r') as f:
        tokens = f.read().split()
    it = iter(tokens)
    ef = float(next(it))
    nbndsub = int(next(it))
    nrr_k   = int(next(it))
    nmodes  = int(next(it))
    nrr_q   = int(next(it))
    nrr_g   = int(next(it))
    # zstar: 3x3xnat was written; epsi: 3x3
    # You must know 'nat' (read from crystal.fmt) to parse zstar correctly.
    # For illustration assume you read crystal.fmt first to get nat.
    # After that read zstar (3*3*nat numbers) and epsi (3*3)
    # Next comes chw(ibnd,jbnd,irk) in loops ibnd=1..nbndsub, jbnd=1..nbndsub, irk=1..nrr_k
    n_chw = nbndsub * nbndsub * nrr_k
    chw_flat = np.empty(n_chw, dtype=np.complex128)  # actual values are complex? in code they WRITE chw(...) directly (could be real or complex)
    for i in range(n_chw):
        a = (next(it))#.split(",")
        print(a)
        #chw_flat[i] = float(a[0]) + 1J*float(a[1])
    # reshape into Fortran ordering:
    chw = chw_flat.reshape((nbndsub, nbndsub, nrr_k), order='F')
    # Likewise parse rdw (nmodes,nmodes,nrr_q) if present
    return {'ef':ef, 'nbndsub':nbndsub, 'nrr_k':nrr_k, 'nmodes':nmodes, 'nrr_q':nrr_q, 'nrr_g':nrr_g, 'chw':chw}

print(read_epwdata())