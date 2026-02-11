import numpy as np
import IO
import SVD
from matplotlib import pyplot as plt
import time
import tools
import Wannier2Bloch

alat = 10.26
#FCC SILICON
outdir = "./Plots/"
DataDir = "../"
prefix = "silicon"
at = (alat / 2.0) * np.array([[-1., 0., 1.], [0., 1., 1.], [-1., 1., 0.]]).T
k_mesh = [10,10,10]
G = np.array([0,0,0]) #gamma 
L = np.array([0.5,0.5,0.5])
X = np.array([0.5, 0.0, 0.5])
K = np.array([0.375, 0.3750, 0.75])
W = np.array([ 0.5000000000,0.2500000000,0.7500000000])
path1 = np.array([[K,G],[G,L],[L,W],[W,X]])
xticks_labels1 = ['K','G','L','W','X']
path2 = np.array([[L,G],[G,K]])
xticks_labels2 = ['L','G','K']

#Import H,D
file_fmt = DataDir + "epwdata.fmt"
H_sort, D_sort,R_sort ,R, ef ,nbnd, nrr_k, nmodes, nrr_q, nrr_g = IO.read_epwdata_fmt(file_fmt, at,k_mesh,True,outdir)
#Import G in bloch space
file_epb = DataDir + prefix + ".epb1"
nq = 1000
G_bloch,q_bloch = IO.read_epb_mixed_record(file_epb,nq,nbnd,nmodes)
#Import G in Wannier Space
file_epmatwp = DataDir + prefix +  ".epmatwp"
G =  IO.read_epmatwp(file_epmatwp,nbnd, nrr_k, nmodes, nrr_g,R,PLOT=True,outdir = outdir)

#SVD
#Wannier
rel_error, x_percent = SVD.svd_trend(G,nbnd,nbnd,nmodes)
U,s,Vt = SVD.SVD(G,nbnd,nbnd,nmodes)

#Bloch
rel_error_k,x_perc_k = SVD.svd_trend(G_bloch,nbnd,nbnd,nmodes)
U_k,s_k,Vt_k = SVD.SVD(G_bloch,nbnd,nbnd,nmodes)

#Full Trend
fig = plt.figure()
plt.plot(x_percent[:], rel_error*100, label="Wannier")
plt.plot(x_perc_k, rel_error_k*100, label="Bloch")
plt.xlabel("% of singular values retained")
plt.ylabel("relative Frobenius error")
plt.legend()
plt.title("Truncated-SVD reconstruction error")
plt.tight_layout()
plt.show()
fig.savefig(outdir+"SVD_trend.jpg")


#ZOOM
fig = plt.figure()
plt.plot(x_percent[1:50], rel_error[1:50]*100,"o-", label="unit-cell")
plt.xlabel("% of singular values retained")
plt.ylabel("relative Frobenius error")
plt.title("Truncated-SVD reconstruction error")
plt.tight_layout()
plt.show()
fig.savefig(outdir+"SVD_zoom.jpg")


#Heatmap
G_abs = np.abs(G)
perc = 4
rank = int(nrr_k * perc / 100)
rank_bloch = int(nq*perc/100)
# Calculate Low Rank Approximation
start_time = time.time()
G_LR2 = SVD.getLowRank(U, s, Vt, rank)
print("LR factorization: %s seconds " % (time.time() - start_time))


# Calculate Error
diff = np.abs(G_LR2 - G)**2
relative_error = np.sum(diff) / np.sum(G_abs**2)
print(f"Relative error with {perc}% SV = {relative_error*100:0.3}%")

start_time = time.time()
G_LR2_k = SVD.getLowRank(U_k, s_k, Vt_k, rank_bloch)
print("LR factorization: %s seconds " % (time.time() - start_time))


# Calculate Error
diff = np.abs(G_LR2_k - G_bloch)**2
relative_error = np.sum(diff) / np.sum(np.abs(G_bloch)**2)
print(f"Relative error with {perc}% SV = {relative_error*100:0.3}%")

# Pre-calculate the log10 data slices
#tiny epsilon (1e-16) prevents log(0) errors
slice_orig = np.log10(np.abs(G[..., 4, 2, 0]) + 1e-16)
slice_lr   = np.log10(np.abs(G_LR2[..., 4, 2, 0]) + 1e-16)
slice_bloch = np.log10(np.abs(G_bloch[..., 4, 2, 0]) + 1e-16)
slice_lr_bloch   = np.log10(np.abs(G_LR2_k[..., 4, 2, 0]) + 1e-16)

# Determine Global Min/Max for the Color Scale
global_min = min(slice_orig.min(), slice_lr.min(), slice_bloch.min(),slice_lr_bloch.min())
global_max = max(slice_orig.max(), slice_lr.max(), slice_bloch.max(),slice_lr_bloch.max())

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
# Plot Original
im1 = axes[0][0].matshow(slice_orig, vmin=global_min, vmax=global_max)
axes[0][0].set_title("Original (Wannier)")
# Plot Low Rank
im2 = axes[0][1].matshow(slice_lr, vmin=global_min, vmax=global_max)
axes[0][1].set_title(f"Low Rank ({perc}%)")
# Plot Bloch
im3 = axes[1][0].matshow(slice_bloch, vmin=global_min, vmax=global_max)
axes[1][0].set_title("Bloch Space")
#Plot Bloch Low Rank
im4 = axes[1][1].matshow(slice_lr_bloch, vmin=global_min, vmax=global_max)
axes[1][1].set_title(f"Bloch Low Rank ({perc}%)")

# Add a single shared colorbar
fig.colorbar(im1, ax=axes.ravel().tolist(), label="log10(|g|)")

plt.show()
fig.savefig(outdir+"matrixPattern.jpg")


#Interpolate
b_lat = tools.get_reciprocal_lattice(at)
q_cart1, x_axis1, xticks1 = tools.build_path_and_ticks(b_lat,path1,30)
q_cart2, x_axis2, xticks2 = tools.build_path_and_ticks(b_lat,path2,30)

E1,_ = Wannier2Bloch.interpolate_H(H_sort,q_cart1,R_sort)
Wannier2Bloch.plotBands(x_axis1,nbnd,E1,xticks1,xticks_labels1,outdir)
w1,_ = Wannier2Bloch.interpolate_D(D_sort,q_cart1,R_sort)
Wannier2Bloch.plotModes(x_axis1,nmodes,w1,xticks1,xticks_labels1,outdir)

# To simulate the condition in which I read from file all the SVD info I preelaborate
# the data for my benchmark
perc = 0.03
rank = int(s.shape[0]*perc)
print(f"Number of SV {rank} corresponding to {perc*100}%")
U_red = U[:,:rank]
Vt_red = Vt[:rank]
s_red = s[:rank]
start_time = time.time()
G_bloch,wq,Ek = Wannier2Bloch.Wannier2Bloch(H_sort,D_sort,G,U_red,Vt_red,s_red,R_sort,q_cart2,np.array([0,0,0]))
print("LowRank Wannierization: %s seconds " % (time.time() - start_time))

start_time = time.time()
G_bloch_full,wq_full,Ek_full = Wannier2Bloch.BENCHMARK(H_sort,D_sort,G,R_sort,q_cart2,np.array([0,0,0]))
print("FullRank Wannierization: %s seconds " % (time.time() - start_time))

tools.plot_EPC(
    G_bloch_full,G_bloch,
    wq_full,wq,
    nbnd,nmodes,
    x_axis2,xticks2,xticks_labels2,perc,outdir)

#Performance Analysis
N = int(1e3)
perc, times, errors = tools.performance_analysis(N,b_lat,H_sort,D_sort,G,R_sort,U,Vt,s,outdir)
