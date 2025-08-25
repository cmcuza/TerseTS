# test_mixture.py
from tersets import Method, compress, decompress
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

def block_linear(n, theta1, theta2, eps, noise=0.05):
    x = np.arange(n, dtype=float)
    y = theta1 * x + theta2
    y += rng.normal(0, noise, size=n)
    return y + eps  # inside the +eps tube

def block_power(n, theta1, theta2, eps, noise=0.02):
    x = np.arange(n, dtype=float)
    y = theta2 * np.power(x + 1.0, theta1)   # (x_rel + 1)^θ1
    y += rng.normal(0, noise, size=n)
    return y + eps

def block_sqrt(n, theta1, theta2, eps, noise=0.02):
    x = np.arange(n, dtype=float)
    y = theta1 * np.sqrt(x + 1.0) + theta2   # sqrt(x_rel + 1)
    y += rng.normal(0, noise, size=n)
    return y + eps

def block_exp(n, theta1, theta2, eps, noise=0.02):
    x = np.arange(n, dtype=float)
    y = theta2 * np.exp(theta1 * x)          # exp(θ1 * x_rel)
    y += rng.normal(0, noise, size=n)
    # Ensure positivity even with noise (exp/power require y>0 pre-shift)
    y = np.maximum(y, 1e-6)
    return y + eps

def block_quadratic(n, theta1, theta2, eps, noise=0.02):
    x = np.arange(n, dtype=float)
    y = theta1 * (x**2) + theta2             # θ1 * x_rel^2 + θ2
    y += rng.normal(0, noise, size=n)
    return y + eps

if __name__ == "__main__":
    error_bound = 0.5
    n_per = 20

    # Make curvature obvious but not explosive.
    y1 = block_power    (n_per, theta1= 1.2, theta2=  0.9, eps=error_bound)
    y2 = block_linear   (n_per, theta1= 1.2, theta2=  2.5, eps=error_bound)
    y3 = block_sqrt     (n_per, theta1= 4.0, theta2=  1.0, eps=error_bound)
    y4 = block_exp      (n_per, theta1= 0.08,theta2=  1.2, eps=error_bound)
    y5 = block_quadratic(n_per, theta1= 0.1,theta2=  2.0, eps=error_bound)

    labels = ['Power', 'Linear', 'Sqrt', 'Exp', 'Quadratic']
    
    uncompressed = np.concatenate([y1, y2, y3, y4, y5]).astype(np.float64)
    x_axis = np.arange(uncompressed.size)

    plt.figure(figsize=(10,5))
    for i, y in enumerate([y1, y2, y3, y4, y5]):
        sx_axis = np.arange(i*n_per, (i+1)*n_per)
        plt.plot(sx_axis, y, label=labels[i], lw=1.5)

    comp = compress(uncompressed.tolist(), Method.NonLinearApproximation, error_bound)
    decomp = np.array(decompress(comp), dtype=float)

    for i, y in enumerate([y1, y2, y3, y4, y5]):
        sx_axis = np.arange(i*n_per, (i+1)*n_per)
        plt.plot(sx_axis, decomp[sx_axis], lw=1.5, color='black')
    

    lo = uncompressed - error_bound
    hi = uncompressed + error_bound
    for i, y in enumerate([y1, y2, y3, y4, y5]):
        sx_axis = np.arange(i*n_per, (i+1)*n_per)
        plt.fill_between(sx_axis, lo[sx_axis], hi[sx_axis], alpha=0.25)

    for b in [n_per, 2*n_per, 3*n_per, 4*n_per]:
        plt.axvline(b, color="#999", ls="--", lw=0.7)

    plt.legend()
    plt.tight_layout()
    plt.show()
