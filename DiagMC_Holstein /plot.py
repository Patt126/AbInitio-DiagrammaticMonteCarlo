import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model  # scikit-learn linear regression


def readSimulationFile(filename):
    """
    readSimulationFile:
    parse the .csv file written by saveSimulationToFile()

    The file has two types of lines:
    1) parameter lines like "g = 0.5"
    2) data lines like "histogram_data,0.01,0.02,..."

    Output:
    data: dict with keys:
        'N', 'bins', 'g', 'mu', 'w0', 'k', 't', 'up_lim', 'energyEstimator'
        'histogram_data'  -> list[float]
        'order_data'      -> list[float]
        'Green_estimator' -> list[float]
    """
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            # arrays: histogram_data, order_data, Green_estimator
            if (
                line.startswith("histogram_data")
                or line.startswith("order_data")
                or line.startswith("Green_estimator")
            ):
                parts = line.strip().split(',')
                # first token is the key, the rest are numeric values
                data[parts[0]] = [float(x) for x in parts[1:]]
            # scalar parameters: lines like "g = 0.5"
            elif (not line.startswith("===")) and len(line) > 2:
                parts = line.strip().split(" = ", 1)
                if len(parts) == 2:
                    data[parts[0]] = float(parts[1])
    return data


# base name chosen in saveSimulationToFile()
name = "data"
path = "./results/"

data = readSimulationFile(path + name + ".csv")

# unpack simulation parameters
bins    = int(data["bins"])
g       = data["g"]
w0      = data["w0"]
k       = data["k"]
mu      = data["mu"]
up_lim  = data["up_lim"]

# build tau bins
bin_edges    = np.linspace(0, up_lim, bins + 1)
bin_centers  = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_width    = bin_edges[1] - bin_edges[0]

# G(tau) estimators from file
histogram        = np.array(data["histogram_data"])    # naive bin-count based estimator
Green_estimator  = np.array(data["Green_estimator"])   # "exact_estimator" style

# plot both estimators on the same figure
plt.scatter(
    bin_centers,
    Green_estimator,
    label='Green_estimator (improved)',
    zorder=2,
    s=10
)

plt.bar(
    bin_centers,
    histogram,
    width=bin_width,
    alpha=0.7,
    label='histogram_data (naive)'
)

plt.title(f'Green function  g={g:.2f}  mu={mu:.3f}  k={k:.3f}')
# uncomment if you want to see exponential tail clearly
# plt.yscale('log')
plt.xlabel('tau')
plt.ylabel('G(tau)')


# fit ln G(tau) ~ a + b * tau on the long-time tail to estimate energy
# we try to pick a monotonic tail region automatically:
lin_model = linear_model.LinearRegression()

t_list = []
x = np.array([])
offset = 0

# heuristic:
# if histogram is increasing with tau (rare, noisy), we scan from the end backward
# else we scan forward from 0 until we hit zeros
if histogram[0] < histogram[-1]:
    start = len(histogram)
    for l in np.flip(histogram):
        if l != 0:
            t_list.append(np.log(l))
            start -= 1
        else:
            break
    t = np.flip(np.array(t_list))
    x = bin_centers[start:start + t.size]
else:
    for l in histogram[offset:]:
        if l != 0:
            t_list.append(np.log(l))
        else:
            break
    t = np.array(t_list)
    x = bin_centers[offset:t.size + offset]

# linear regression on log G(tau)
# t ~ a + b * x  => G(tau) ~ Z * exp(b * tau), so slope b ~ - (E - mu)
lin_model.fit(x.reshape(-1,1), t)
Egs  = mu - lin_model.coef_[0]    # estimate ground-state energy
Zgs  = np.exp(lin_model.intercept_)  # quasiparticle residue (normalization)

# info box to overlay on the plot
# note: "energyEstimator" in file is the direct estimator from the MC loop
textstr = (
    f"MC energy estimator: {data['energyEstimator']:.4f}\n"
    "STD: 0.0003 (Blocking)\n"
    "(naive STD: 0.00005)\n"
    "Fit tail ln G(tau):\n"
    f"E_fit: {Egs:.4f}\n"
    f"Z_fit: {Zgs:.4f}"
)

plt.legend()
plt.text(
    0.65,
    0.75,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='center',
    multialignment='center',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=1)
)

# save the first figure before adding the fit curve
plt.savefig(path + name + "_green_.png")

# overlay best-fit exponential curve on top of the scatter/bar
plt.plot(
    x,
    np.exp(lin_model.intercept_ + x * lin_model.coef_[0]),
    label="exp fit"
)
plt.legend()
plt.show()


# second figure: distribution of diagram orders
orders = np.arange(0, 2 * len(data["order_data"]), 2)
plt.bar(orders, data["order_data"])
plt.title("Order Distribution")
plt.xlabel("order")
plt.ylabel("counts")
plt.savefig(path + name + "_orders.png")
plt.show()
