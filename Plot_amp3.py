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

# Compute radial frequencies and mean relative log amplitude, focusing on high frequencies
def compute_high_frequency_amplitude(log_amplitude, rows, cols, num_points, cutoff=0.5):
    crow, ccol = rows // 2, cols // 2
    X, Y = np.ogrid[:rows, :cols]
    r = np.sqrt((X - crow) ** 2 + (Y - ccol) ** 2) / np.sqrt(crow**2 + ccol**2)  # Normalize radius

    relative_log_amplitude_plot = np.zeros(num_points)
    for i in range(num_points):
        mask = (r >= i / num_points * cutoff) & (r < (i + 1) / num_points * cutoff)
        relative_log_amplitude_plot[i] = np.mean(log_amplitude[mask])
    
    # Apply smoothing to reduce noise
    relative_log_amplitude_plot = gaussian_filter1d(relative_log_amplitude_plot, sigma=2)
    
    return relative_log_amplitude_plot

# Define a cutoff value to isolate high frequencies
cutoff = 0.8  # Adjust this value to focus more on higher frequencies (0.5 means half of the frequencies)
num_points = log_magnitude_spectrum1.shape[0] // 2  # Only take half for 0 to pi

relative_log_amplitude_plot1 = compute_high_frequency_amplitude(relative_log_amplitude1, image_array1.shape[0], image_array1.shape[1], num_points, cutoff)
relative_log_amplitude_plot2 = compute_high_frequency_amplitude(relative_log_amplitude2, image_array2.shape[0], image_array2.shape[1], num_points, cutoff)

# Compute the difference between the high frequencies
high_frequency_loss = relative_log_amplitude_plot1 - relative_log_amplitude_plot2

# Generate frequencies from 0 to pi
frequencies = np.linspace(0, np.pi, num_points)

# Plot the results
plt.figure(figsize=(10, 7))

# Plot for high-frequency loss
plt.plot(frequencies, high_frequency_loss, label='High Frequency Loss (Image 1 - Image 2)', color='purple', linestyle='-', marker='o')

# Enhance visualization
plt.yscale('log')  # Logarithmic scale for the y-axis
plt.title('High Frequency Loss between Two Images (0 to π)')
plt.xlabel('Frequency (radians)')
plt.ylabel('Log Difference (Log Scale)')
plt.legend(title='Difference')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.xlim([0, np.pi])  # Limit x-axis to 0 to π
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], 
           ['0', 'π/4', 'π/2', '3π/4', 'π'])

# Show the plot
plt.show()
