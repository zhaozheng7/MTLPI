import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def create_synth_data(n=1000, plot=False):
    """Create a synthetic sinusoidal dataset with varying PI width"""
    np.random.seed(7)
    X = np.linspace(-5, 5, num=n)
    randn = np.random.normal(size=n)
    gauss = (2 + 2 * np.cos(1.2 * X))
    noise = gauss * randn
    orig = 10 + 5 * np.cos(X + 2)
    Y = orig + noise
    P1 = orig + 1.96 * gauss
    P2 = orig - 1.96 * gauss
    if plot:
        plt.figure(figsize=(4, 3))  # (9.97, 7.66)
        plt.fill_between(X, P1, P2, color='gray', alpha=0.5, linewidth=0, label='Ideal 95% PIs')
        plt.scatter(X, Y, label='Data with noise')
        plt.plot(X, orig, 'r', label='True signal')
        plt.legend()
    nonconstant_noise = np.array([X, Y]).T

    nonconstant_noise = pd.DataFrame(nonconstant_noise, columns=['x', 'y'])

    nonconstant_noise.to_csv('syntheticdata.csv', index=False)
    return X, Y, P1, P2

if __name__ == '__main__':
    create_synth_data(1000, plot=True)
    plt.show()