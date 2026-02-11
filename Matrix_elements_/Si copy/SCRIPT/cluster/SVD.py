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
    x_percent = np.arange(1, rmax + 1) / rmax * 100
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


    








