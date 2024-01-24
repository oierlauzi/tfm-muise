import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

if __name__ == '__main__':
    fig, ax = plt.subplots()

    # Generate samples from two Gaussian distributions
    samples = np.concatenate([
        np.random.normal(3.0, 2.0, 2048),
        np.random.normal(-4.0, 2.0, 4096)
    ])
    
    # Plot histogram of the samples
    _, bins, _ = ax.hist(samples, bins=64, density=True, color='black', label='Empirical data')
    x = (bins[1:] + bins[:-1]) / 2

    # Fit a Gaussian Mixture Model to the samples
    gmm = GaussianMixture(n_components=2, covariance_type='tied')
    gmm.fit(samples[...,None])

    # Obtain GMM model parameters
    weights = gmm.weights_
    means = gmm.means_[:,0]
    variances = gmm.covariances_[...,0,0]
    stddevs = np.sqrt(variances)
    
    # Plot individual GMM curves
    ys = weights*norm.pdf(x[:,None], means, stddevs)
    n = ys.shape[-1]
    ax.plot(x, ys[:,0], linestyle='dashed', label='$\\mathcal{L}(C_+)$')
    ax.plot(x, ys[:,1], linestyle='dashed', label='$\\mathcal{L}(C_-)$')

    # Plot the sum of the GMM curves     
    ax.plot(x, ys.sum(axis=-1), label='$\\mathcal{L}(C_+) + \\mathcal{L}(C_-)$')

    # Plot log likelihood ratio
    ax2 = ax.twinx()
    log_likelihood_ratio = np.log(ys[:,0] / ys[:,1])
    ax2.plot(x, log_likelihood_ratio, color='yellow', label='$\\log \\left(\\frac{\\mathcal{L}(C_+)}{\\mathcal{L}(C_-)} \\right)$')

    ax.set_xlabel('$\\rho$')
    ax.set_ylabel('probability')
    ax2.set_ylabel('Log likelihood ratio')

    # Show legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)

    # Show lines
    ax2.axvline(0, color='gray', linestyle='dotted')
    ax2.axhline(0, color='gray', linestyle='dotted')

    plt.show()
