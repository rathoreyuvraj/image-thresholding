import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the uploaded image
image_path = '/mnt/data/17254795263625151736798368938786.jpg'
image = Image.open(image_path).convert('L')
image_array = np.array(image)

# Compute the 2D Fourier Transform of the image
f_transform = np.fft.fft2(image_array)
f_transform_shifted = np.fft.fftshift(f_transform)

# Compute the magnitude spectrum (amplitude) and apply log scale
magnitude_spectrum = np.abs(f_transform_shifted)
log_magnitude_spectrum = np.log1p(magnitude_spectrum)

# Define a range of scaling factors
scaling_factors = [0.9, 0.95, 1.0, 1.05, 1.1]

# Set up the figure
plt.figure(figsize=(8, 6))

# Number of points for the x-axis (frequencies)
num_points = log_magnitude_spectrum.shape[0] // 2  # Only take half for 0 to pi

# Generate frequencies from 0 to pi
frequencies = np.linspace(0, np.pi, num_points)

# Loop over each scaling factor
for scale in scaling_factors:
    # Adjust the log magnitude spectrum by the scaling factor
    adjusted_spectrum = log_magnitude_spectrum * scale
    
    # Calculate the relative log amplitude by normalizing
    relative_log_amplitude = adjusted_spectrum / np.max(adjusted_spectrum)
    
    # Compute the radial frequency
    rows, cols = image_array.shape
    crow, ccol = rows // 2 , cols // 2
    X, Y = np.ogrid[:rows, :cols]
    r = np.sqrt((X - crow)**2 + (Y - ccol)**2)
    
    # Compute the mean relative log amplitude as a function of frequency
    relative_log_amplitude_plot = np.zeros(num_points)
    for i in range(num_points):
        mask = (r >= i) & (r < i + 1)
        relative_log_amplitude_plot[i] = np.mean(relative_log_amplitude[mask])
    
    # Plot the relative log amplitude curve for the current scale
    plt.plot(frequencies, relative_log_amplitude_plot, label=f'{scale}')

# Customize the plot
plt.title('Relative Log Amplitude for Different Scaling Factors (0 to π)')
plt.xlabel('Frequency (radians)')
plt.ylabel('Relative Log Amplitude')
plt.legend(title='Scaling Factor')
plt.grid(True)
plt.xlim([0, np.pi])  # Limit x-axis to 0 to π
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], 
           ['0', 'π/4', 'π/2', '3π/4', 'π'])

# Show the plot
plt.show()
