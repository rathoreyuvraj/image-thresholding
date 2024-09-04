import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from PIL import Image

# Load the two images
image_path1 = 'path_to_image1.jpg'  # Replace with the path to your first image
image_path2 = 'path_to_image2.jpg'  # Replace with the path to your second image

image1 = Image.open(image_path1).convert('L')
image2 = Image.open(image_path2).convert('L')

image_array1 = np.array(image1)
image_array2 = np.array(image2)

# Compute the 2D Fourier Transform and log amplitude for both images
def compute_log_amplitude(image_array):
    f_transform = np.fft.fft2(image_array)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    log_magnitude_spectrum = np.log1p(magnitude_spectrum)
    return log_magnitude_spectrum

log_magnitude_spectrum1 = compute_log_amplitude(image_array1)
log_magnitude_spectrum2 = compute_log_amplitude(image_array2)

# Normalize the log magnitude spectra together
max_value = max(np.max(log_magnitude_spectrum1), np.max(log_magnitude_spectrum2))
relative_log_amplitude1 = log_magnitude_spectrum1 / max_value
relative_log_amplitude2 = log_magnitude_spectrum2 / max_value

# Compute radial frequencies and mean relative log amplitude
def compute_mean_log_amplitude(log_amplitude, rows, cols, num_points):
    crow, ccol = rows // 2, cols // 2
    X, Y = np.ogrid[:rows, :cols]
    r = np.sqrt((X - crow) ** 2 + (Y - ccol) ** 2)

    relative_log_amplitude_plot = np.zeros(num_points)
    for i in range(num_points):
        mask = (r >= i) & (r < i + 1)
        relative_log_amplitude_plot[i] = np.mean(log_amplitude[mask])
    
    # Apply smoothing to reduce noise
    relative_log_amplitude_plot = gaussian_filter1d(relative_log_amplitude_plot, sigma=2)
    
    return relative_log_amplitude_plot

num_points = log_magnitude_spectrum1.shape[0] // 2  # Only take half for 0 to pi
relative_log_amplitude_plot1 = compute_mean_log_amplitude(relative_log_amplitude1, image_array1.shape[0], image_array1.shape[1], num_points)
relative_log_amplitude_plot2 = compute_mean_log_amplitude(relative_log_amplitude2, image_array2.shape[0], image_array2.shape[1], num_points)

# Generate frequencies from 0 to pi
frequencies = np.linspace(0, np.pi, num_points)

# Plot the results
plt.figure(figsize=(10, 7))

# Plot for image 1
plt.plot(frequencies, relative_log_amplitude_plot1, label='Image 1', color='blue', linestyle='-', marker='o')

# Plot for image 2
plt.plot(frequencies, relative_log_amplitude_plot2, label='Image 2', color='red', linestyle='--', marker='x')

# Enhance visualization
plt.yscale('log')  # Logarithmic scale for the y-axis
plt.title('Enhanced Relative Log Amplitude for Two Images (0 to π)')
plt.xlabel('Frequency (radians)')
plt.ylabel('Relative Log Amplitude (Log Scale)')
plt.legend(title='Image')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.xlim([0, np.pi])  # Limit x-axis to 0 to π
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], 
           ['0', 'π/4', 'π/2', '3π/4', 'π'])

# Show the plot
plt.show()
