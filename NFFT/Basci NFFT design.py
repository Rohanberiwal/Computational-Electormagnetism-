import numpy as np
import scipy.fft as fft
import h5py
import matplotlib.pyplot as plt
from numba import jit
import unittest

# ===========================
# 1. NFFT Transformation Prototype
# ===========================
Nx, Ny = 128, 128
x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
X, Y = np.meshgrid(x, y)

# Simulated near-field data (Gaussian source for demo purposes)
E_near = np.exp(-((X**2 + Y**2) / 0.1))

# Apply 2D Fourier Transform to approximate far-field response
E_far = fft.fftshift(fft.fft2(E_near))

# Plot results
def plot_fields():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(np.abs(E_near), extent=[-1, 1, -1, 1], cmap='viridis')
    ax[0].set_title("Near-field Data")
    ax[1].imshow(np.abs(E_far), extent=[-1, 1, -1, 1], cmap='magma')
    ax[1].set_title("Far-field Approximation")
    plt.show()

# ===========================
# 2. Handling gprMax Output Files
# ===========================
def load_gprmax_data(file_path):
    with h5py.File(file_path, "r") as f:
        return np.array(f["Ex"])  # Extracting Electric field data

# ===========================
# 3. Optimized NFFT with Numba
# ===========================
@jit(nopython=True, parallel=True)
def nfft_transform(E_near):
    Nx, Ny = E_near.shape
    E_far = np.zeros((Nx, Ny), dtype=np.complex128)
    for i in range(Nx):
        for j in range(Ny):
            E_far[i, j] = np.sum(E_near * np.exp(-2j * np.pi * (i + j) / Nx))
    return E_far

# ===========================
# 4. Radar Cross Section (RCS) Computation
# ===========================
def compute_rcs(E_far):
    power_pattern = np.abs(E_far)**2
    return 10 * np.log10(power_pattern / np.max(power_pattern))  # dB scale

def plot_rcs(E_far):
    rcs = compute_rcs(E_far)
    plt.plot(rcs)
    plt.xlabel("Angle")
    plt.ylabel("RCS (dB)")
    plt.title("Radar Cross Section")
    plt.show()

# ===========================
# 5. Unit Test for NFFT Validation
# ===========================
class TestNFFT(unittest.TestCase):
    def test_fft_consistency(self):
        E_near = np.random.rand(64, 64)
        E_far = fft.fftshift(fft.fft2(E_near))
        self.assertEqual(E_far.shape, E_near.shape)  # Shape should remain same

if __name__ == "__main__":
    print("Running NFFT Transformation and Plots...")
    plot_fields()
    print("Computing and Plotting RCS...")
    plot_rcs(E_far)
    print("Running Unit Tests...")
    unittest.main()
