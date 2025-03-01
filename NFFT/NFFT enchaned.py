import numpy as np
import scipy.fft as fft
import h5py
import matplotlib.pyplot as plt
from numba import jit, prange
import unittest
import argparse

Nx, Ny = 128, 128
x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
X, Y = np.meshgrid(x, y)
E_near = np.exp(-((X**2 + Y**2) / 0.1))
E_far = fft.fftshift(fft.fft2(E_near))

def plot_3d_field(E, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.linspace(-1, 1, E.shape[0]), np.linspace(-1, 1, E.shape[1]))
    ax.plot_surface(X, Y, np.abs(E), cmap='viridis')
    ax.set_title(title)
    plt.show()

def plot_polar_rcs(E_far):
    rcs = compute_rcs(E_far)
    angles = np.linspace(0, 2 * np.pi, len(rcs))
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(angles, rcs)
    ax.set_title("Radar Cross Section (Polar Plot)")
    plt.show()

@jit(nopython=True, parallel=True)
def nfft_transform(E_near):
    Nx, Ny = E_near.shape
    E_far = np.zeros((Nx, Ny), dtype=np.complex128)
    for i in prange(Nx):
        for j in prange(Ny):
            E_far[i, j] = np.sum(E_near * np.exp(-2j * np.pi * (i + j) / Nx))
    return E_far

def compute_rcs(E_far):
    power_pattern = np.abs(E_far)**2
    return 10 * np.log10(power_pattern / np.max(power_pattern))

def save_to_hdf5(filename, E_near, E_far):
    with h5py.File(filename, "w") as f:
        f.create_dataset("E_near", data=E_near)
        f.create_dataset("E_far", data=E_far)

class TestNFFT(unittest.TestCase):
    def test_fft_consistency(self):
        E_near = np.random.rand(64, 64)
        E_far = fft.fftshift(fft.fft2(E_near))
        self.assertEqual(E_far.shape, E_near.shape)
    
    def test_parseval_theorem(self):
        E_near = np.random.rand(64, 64)
        E_far = fft.fftshift(fft.fft2(E_near))
        power_time_domain = np.sum(np.abs(E_near)**2)
        power_freq_domain = np.sum(np.abs(E_far)**2) / (64 * 64)
        self.assertAlmostEqual(power_time_domain, power_freq_domain, places=5)
    
    def test_phase_consistency(self):
        E_near = np.random.rand(64, 64) + 1j * np.random.rand(64, 64)
        E_far = fft.fftshift(fft.fft2(E_near))
        phase_near = np.angle(E_near)
        phase_far = np.angle(E_far)
        self.assertTrue(np.all((phase_far >= -np.pi) & (phase_far <= np.pi)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, help="Save output to HDF5 file")
    args = parser.parse_args()
    
    plot_3d_field(E_near, "Near-field Data")
    plot_3d_field(E_far, "Far-field Approximation")
    plot_polar_rcs(E_far)
    
    if args.save:
        save_to_hdf5(args.save, E_near, E_far)
    
    unittest.main()
