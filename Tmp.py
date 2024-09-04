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

# Only plot the frequencies up to π
mask1 = radius1 <= np.pi
mask2 = radius2 <= np.pi
radius1_plot = radius1[mask1]
relative_log_amplitude1_plot = relative_log_amplitude1[mask1]
radius2_plot = radius2[mask2]
relative_log_amplitude2_plot = relative_log_amplitude2[mask2]

# Plotting
plt.figure(figsize=(10, 6))

# Plot for Image 1
plt.scatter(radius1_plot, relative_log_amplitude1_plot, c='blue', s=1, label='Image 1')

# Plot for Image 2
plt.scatter(radius2_plot, relative_log_amplitude2_plot, c='red', s=1, label='Image 2')

plt.xlabel('Frequency (radians per sample)')
plt.ylabel('Relative Log Amplitude')
plt.title('Relative Log Amplitude vs Frequency for Two Images')
plt.legend()
plt.colorbar(label='Relative Log Amplitude')
plt.grid(True)

plt.show()
