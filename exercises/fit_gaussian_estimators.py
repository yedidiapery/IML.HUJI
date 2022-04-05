from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    uniGau = UnivariateGaussian().fit(samples)
    print(uniGau.mu_, uniGau.var_)

    # Question 2 - Empirically showing sample mean is consistent
    absDist = np.array([np.array([np.abs(uniGau.mu_ - UnivariateGaussian().fit(samples[0:i]).mu_), i])
                        for i in range(10, 1001, 10)])
    plt.scatter(absDist[:, 1], absDist[:, 0])
    plt.xlabel("batch size")
    plt.ylabel("distance between the true & estimated value")
    plt.title("difference in estimation per batch size")
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = uniGau.pdf(samples)
    plt.scatter(samples, pdf)
    plt.xlabel("samples value")
    plt.ylabel("samples pdf value")
    plt.title("the distribution of pdf as function of the value")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, sigma, 1000)
    multiGau = MultivariateGaussian().fit(samples)
    print(multiGau.mu_)
    print(multiGau.cov_)

    # Question 5 - Likelihood evaluation
    heatmap = np.zeros((200, 200))
    for i, f1 in enumerate(np.linspace(-10, 10, 200)):
        for j, f3 in enumerate(np.linspace(-10, 10, 200)):
            heatmap[i][j] = multiGau.log_likelihood(np.array([f1, 0, f3, 0]), sigma, samples)
    fig, ax = plt.subplots()
    min, max = np.min(heatmap), np.max(heatmap)
    c = ax.pcolormesh(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200), heatmap, cmap='RdBu', vmin=min, vmax=max)
    ax.set_title('log likelihood heatmap')
    ax.axis([-10, 10, -10, 10])
    fig.colorbar(c, ax=ax)
    plt.show()

    # Question 6 - Maximum likelihood
    ind = np.unravel_index(heatmap.argmax(), heatmap.shape)
    print(round(np.linspace(-10, 10, 200)[ind[0]], 3), round(np.linspace(-10, 10, 200)[ind[1]], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
