import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the images
image_path1 = '/mnt/data/17254834298008182437309328465507.jpg'  # Replace with the correct path to your first image
image_path2 = '/mnt/data/17254795263625151736798368938786.jpg'  # Replace with the correct path to your second image

image1 = Image.open(image_path1).convert('L')
image2 = Image.open(image_path2).convert('L')

# Convert the images to numpy arrays
image_array1 = np.array(image1)
image_array2 = np.array(image2)

# Function to compute the relative log amplitude
def compute_relative_log_amplitude(image_array):
    # Compute the 2D Fourier Transform and log amplitude
    f_transform = np.fft.fft2(image_array)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    log_magnitude_spectrum = np.log1p(magnitude_spectrum)
    
    # Normalize the log amplitude and make it negative
    relative_log_amplitude = -log_magnitude_spectrum / np.max(log_magnitude_spectrum)
    
    return relative_log_amplitude

# Compute relative log amplitude for both images
relative_log_amplitude1 = compute_relative_log_amplitude(image_array1)
relative_log_amplitude2 = compute_relative_log_amplitude(image_array2)

# Function to compute the radial frequencies and mean relative log amplitude
def compute_radial_profile(relative_log_amplitude, num_points=50):
    rows, cols = relative_log_amplitude.shape
    crow, ccol = rows // 2, cols // 2
    X, Y = np.ogrid[:rows, :cols]
    r = np.sqrt((X - crow) ** 2 + (Y - ccol) ** 2)
    r = r / r.max()  # Normalize radius
    
    relative_log_amplitude_plot = np.zeros(num_points)
    
    for i in range(num_points):
        mask = (r >= i / num_points) & (r < (i + 1) / num_points)
        relative_log_amplitude_plot[i] = np.mean(relative_log_amplitude[mask])
    
    return relative_log_amplitude_plot

# Compute the radial profiles for both images
num_points = 50
relative_log_amplitude_plot1 = compute_radial_profile(relative_log_amplitude1, num_points)
relative_log_amplitude_plot2 = compute_radial_profile(relative_log_amplitude2, num_points)

# Generate frequencies from 0 to pi
frequencies = np.linspace(0, np.pi, num_points)

# Plot the results
plt.figure(figsize=(10, 7))
plt.plot(frequencies, relative_log_amplitude_plot1, label='Image 1 Relative Log Amplitude', color='blue', linestyle='-', marker='o')
plt.plot(frequencies, relative_log_amplitude_plot2, label='Image 2 Relative Log Amplitude', color='red', linestyle='-', marker='o')

# Customize plot appearance
plt.title('Relative Log Amplitude of Two Images (0 to π)')
plt.xlabel('Frequency (radians)')
plt.ylabel('Relative Log Amplitude')
plt.legend(title='Images')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.xlim([0, np.pi])  # Limit x-axis to 0 to π
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], 
           ['0', 'π/4', 'π/2', '3π/4', 'π'])
plt.ylim([-1, 0])  # Limit y-axis to negative values
plt.yticks(np.arange(-1, 0.1, 0.1))

# Show the plot
plt.show()
