from scipy.linalg import svdvals, svd
import numpy as np

def svd_trend(matrix,nBand_1,nBand_2,nBand_ph ,second = False, matrix2 = np.empty(0),name=""):
    rmax = min(matrix.shape[0], matrix.shape[1])
    print("Number of SVs:", rmax)
    rel_error = np.zeros(rmax, dtype=float)
    frob = 0.0

    for m in range(nBand_1):
        for n in range(nBand_2):
            for v in range(nBand_ph):
                g_kq = matrix[ :, : ,v, n,m]
                s = svdvals(g_kq)
                if s.size < rmax:
                    s = np.pad(s, (0, rmax - s.size))
                else:
                    s = s[:rmax]
                s_2 = s**2
                frob += s_2.sum()
                for k in range(rmax):
                    rel_error[k] += s_2[k:].sum()

    rel_error = (rel_error / frob)

    if (second):
        rmax2 = min(matrix2.shape[0], matrix2.shape[1])
        rel_error2 = np.zeros(rmax, dtype=float)
        frob2 = 0.0

        for m in range(nBand_1):
            for n in range(nBand_2):
                for v in range(nBand_ph):
                    g_kq2 = matrix2[ :, :, v, n, m]
                    s2 = svdvals(g_kq2)
                    if s2.size < rmax2:
                        s2 = np.pad(s2, (0, rmax2 - s2.size))
                    else:
                        s2 = s2[:rmax2]
                    s_2 = s2**2
                    frob2 += s_2.sum()
                    for k in range(rmax):
                        rel_error2[k] += s_2[k:].sum()

        rel_error2 = (rel_error2 / frob2)

    # Size of rank r is r*(N_re+N_rp+1)*N_v*N_b*N_b
    x_percent = np.arange(1, rmax + 1)*((matrix.shape[0]+matrix.shape[1]+1)*nBand_ph*nBand_2*nBand_1) / matrix.size * 100
    return rel_error,x_percent

def SVD(matrix,nBand_n,nBand_m,nBand_ph ,name="",SVlist = np.empty(0)):
    rmax = min(matrix.shape[0], matrix.shape[1])
    U = np.zeros((matrix.shape[0],rmax,nBand_ph,nBand_n,nBand_m),dtype="complex")
    Vt = np.zeros((rmax,matrix.shape[1],nBand_ph,nBand_n,nBand_m) ,dtype="complex")
    s = np.zeros((rmax,nBand_ph,nBand_n,nBand_m),dtype="complex")

    for m in range(nBand_m):
        for n in range(nBand_n):
            for v in range(nBand_ph):
                U[...,v,n,m],s[...,v,n,m],Vt[...,v,n,m] = svd(matrix[...,v,n,m],full_matrices=False)
    return U,s,Vt


def getLowRank(U,s,Vt,rank = 0):
    #NB Vt has singular vectors as rows and U as columns
    if(rank<=0):
        rank = s.shape[0]
    if(rank>s.shape[0]):
        print("non valid rank")
        return
    s_red = s[:rank]
    U_red = U[:,:rank,...]
    Vt_red = Vt[:rank,...] 
    return  np.einsum("rvnm,irvnm,rkvnm -> ikvnm",s_red,U_red,Vt_red)


import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import multi_mode_dot

# Imposta il backend su numpy per compatibilità con il tuo codice
tl.set_backend('numpy')

def HOOI_decomposition(matrix, ranks, n_iter=10, tol=1e-4):
    """
    Esegue la decomposizione Tucker tramite algoritmo HOOI 
    Sostituisce la tua funzione SVD(...) ma lavora sull'intero tensore 5D.
    
    Args:
        matrix: Tensore 5D (Nk, Nq, Nph, Nn, Nm)
        ranks: Lista di 5 interi [Rk, Rq, Rph, Rn, Rm]
    Returns:
        core: Tensore Core compresso
        factors: Lista delle matrici di base [U_k, U_q, U_ph, U_n, U_m]
    """
    # Inizializzazione: HOSVD (Step 1 dell'Algoritmo 1)
    # TensorLy gestisce l'inizializzazione SVD internamente in modo efficiente
    core, factors = tucker(matrix, rank=ranks, init='svd', n_iter_max=n_iter, tol=tol)
    
    return core, factors

def HOOI_trend(matrix, nBand_1, nBand_2, nBand_ph, step=1,rmax=0, collect=True):
    """
    Calcola l'errore di ricostruzione al variare del rank, simile a svd_trend.
    
    NOTA: A differenza di svd_trend, qui NON possiamo calcolare tutti i rank in un colpo solo.
    Dobbiamo eseguire l'algoritmo HOOI da capo per ogni rank.
    """
    # Dimensioni originali
    Nk, Nq = matrix.shape[0], matrix.shape[1]
    if rmax==0:
        rmax = min(Nk, Nq) 
    matrix_size = matrix.size
    print(f"Number of Ranks to test: {rmax} (Step: {step})")
    
    # Liste per salvare i risultati
    rel_error = []
    x_percent = []
    CORES = []
    FACTORS = []
        
    
    # Norm totale per calcolare l'errore relativo (frobenius)
    norm_total = np.linalg.norm(matrix)

    # Loop sui rank (come il loop interno implicito in svd_trend)
    for r in range(1, rmax + 1, step):
        
        # DEFINIZIONE DEI RANK TARGET
        # Qui sta la differenza fisica chiave:
        # Per confrontarlo col tuo SVD, comprimiamo k e q (rank r)
        # ma manteniamo bande e modi "Full Rank" (o quasi full).
        current_ranks = [r, r, nBand_ph, nBand_1, nBand_2]

        try:
            # 1. Esegui HOOI per questo specifico rank
            core, factors = HOOI_decomposition(matrix, ranks=current_ranks)
            
            # 2. Ricostruisci il tensore approssimato
            # Formula: X ~ G x1 U1 x2 U2 ... [cite: 82]
            matrix_approx = tl.tucker_to_tensor((core, factors))
            
            # Calcola errore relativo
            diff = matrix - matrix_approx
            err = (np.linalg.norm(diff) / norm_total)**2 # Al quadrato per coerenza con s^2 del tuo codice
            
            rel_error.append(err)
            #correspond to r^2N_vN_B^2 + N_v^2 + N_B^2*2 + r* N_B + r*N_B
            reduced_size = np.sum([U.size for U in factors]) + core.size
            x_percent.append(reduced_size/ matrix.size * 100)
            if collect:
                CORES.append(core)
                FACTORS.append(factors)
            print(f"Rank {r}/{rmax} -> Rel. Error^2: {err:.2e}")
            
        except Exception as e:
            print(f"Rank {r} failed: {e}")
            rel_error.append(rel_error[-1] if rel_error else 1.0)
            #correspond to r^2N_vN_B^2 + N_v^2 + N_B^2*2 + r* N_B + r*N_B
            reduced_size = np.sum([U.size for U in factors]) + core.size
            x_percent.append(reduced_size/ matrix.size * 100)

    return np.array(rel_error), np.array(x_percent), CORES, FACTORS 

def getLowRank_HOOI(core, factors):
    """
    Ricostruisce il tensore approssimato dai componenti HOOI.
    Analogo alla tua funzione getLowRank.
    """
    return tl.tucker_to_tensor((core, factors))








