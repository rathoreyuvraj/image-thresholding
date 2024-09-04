import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

def compute_relative_log_amplitude(image):
    # Compute the 2D Fourier Transform
    fft_result = np.fft.fft2(image)
    fft_result_shifted = np.fft.fftshift(fft_result)  # Center the zero frequency component

    # Compute the Amplitude Spectrum
    amplitude_spectrum = np.abs(fft_result_shifted)

    # Compute the Log Amplitude Spectrum
    log_amplitude_spectrum = np.log10(amplitude_spectrum + np.finfo(float).eps)  # Adding a small value to avoid log(0)

    # Compute Relative Log Amplitude
    relative_log_amplitude = log_amplitude_spectrum - np.max(log_amplitude_spectrum)

    # Generate the Frequency Axis
    rows, cols = image.shape
    freq_x = np.fft.fftfreq(cols, d=1) * 2 * np.pi
    freq_y = np.fft.fftfreq(rows, d=1) * 2 * np.pi
    freq_x, freq_y = np.meshgrid(freq_x, freq_y)

    # Convert the frequency axis to polar coordinates
    radius = np.sqrt(freq_x**2 + freq_y**2)

    return radius, relative_log_amplitude

def compute_binned_average(radius, relative_log_amplitude, num_bins=50):
    # Only consider frequencies up to π
    mask = radius <= np.pi
    radius_plot = radius[mask]
    relative_log_amplitude_plot = relative_log_amplitude[mask]

    # Bin the radius data
    hist, bin_edges = np.histogram(radius_plot, bins=num_bins, range=(0, np.pi))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute average log amplitude for each bin
    binned_amplitude = np.zeros_like(bin_centers)
    for i in range(len(bin_centers)):
        bin_mask = (radius_plot >= bin_edges[i]) & (radius_plot < bin_edges[i + 1])
        if np.sum(bin_mask) > 0:
            binned_amplitude[i] = np.mean(relative_log_amplitude_plot[bin_mask])
    
    return bin_centers, binned_amplitude

# Load and convert images to grayscale
image1 = imread('path_to_image1.png')  # Replace with your image path
image2 = imread('path_to_image2.png')  # Replace with your image path

if image1.ndim == 3:
    image1 = np.mean(image1, axis=2)  # Convert to grayscale if the image is RGB

if image2.ndim == 3:
    image2 = np.mean(image2, axis=2)  # Convert to grayscale if the image is RGB

# Compute the relative log amplitude for each image
radius1, relative_log_amplitude1 = compute_relative_log_amplitude(image1)
radius2, relative_log_amplitude2 = compute_relative_log_amplitude(image2)

# Compute binned average log amplitudes
bin_centers1, binned_amplitude1 = compute_binned_average(radius1, relative_log_amplitude1)
bin_centers2, binned_amplitude2 = compute_binned_average(radius2, relative_log_amplitude2)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(bin_centers1, binned_amplitude1, color='blue', label='Image 1')
plt.plot(bin_centers2, binned_amplitude2, color='red', label='Image 2')

plt.xlabel('Frequency (radians per sample)')
plt.ylabel('Average Relative Log Amplitude')
plt.title('Average Relative Log Amplitude vs Frequency for Two Images')
plt.legend()
plt.grid(True)

plt.show()
