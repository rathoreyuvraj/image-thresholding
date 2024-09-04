import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the two images
image_path1 = 'path_to_image1.jpg'  # Replace with the path to your first image
image_path2 = 'path_to_image2.jpg'  # Replace with the path to your second image

image1 = Image.open(image_path1).convert('L')
image2 = Image.open(image_path2).convert('L')

image_array1 = np.array(image1)
image_array2 = np.array(image2)

# Compute the 2D Fourier Transform of the first image
f_transform1 = np.fft.fft2(image_array1)
f_transform_shifted1 = np.fft.fftshift(f_transform1)
magnitude_spectrum1 = np.abs(f_transform_shifted1)
log_magnitude_spectrum1 = np.log1p(magnitude_spectrum1)

# Compute the 2D Fourier Transform of the second image
f_transform2 = np.fft.fft2(image_array2)
f_transform_shifted2 = np.fft.fftshift(f_transform2)
magnitude_spectrum2 = np.abs(f_transform_shifted2)
log_magnitude_spectrum2 = np.log1p(magnitude_spectrum2)

# Number of points for the x-axis (frequencies)
num_points = log_magnitude_spectrum1.shape[0] // 2  # Only take half for 0 to pi

# Generate frequencies from 0 to pi
frequencies = np.linspace(0, np.pi, num_points)

# Compute relative log amplitude for image 1
relative_log_amplitude1 = log_magnitude_spectrum1 / np.max(log_magnitude_spectrum1)
rows1, cols1 = image_array1.shape
crow1, ccol1 = rows1 // 2, cols1 // 2
X1, Y1 = np.ogrid[:rows1, :cols1]
r1 = np.sqrt((X1 - crow1) ** 2 + (Y1 - ccol1) ** 2)

relative_log_amplitude_plot1 = np.zeros(num_points)
for i in range(num_points):
    mask = (r1 >= i) & (r1 < i + 1)
    relative_log_amplitude_plot1[i] = np.mean(relative_log_amplitude1[mask])

# Compute relative log amplitude for image 2
relative_log_amplitude2 = log_magnitude_spectrum2 / np.max(log_magnitude_spectrum2)
rows2, cols2 = image_array2.shape
crow2, ccol2 = rows2 // 2, cols2 // 2
X2, Y2 = np.ogrid[:rows2, :cols2]
r2 = np.sqrt((X2 - crow2) ** 2 + (Y2 - ccol2) ** 2)

relative_log_amplitude_plot2 = np.zeros(num_points)
for i in range(num_points):
    mask = (r2 >= i) & (r2 < i + 1)
    relative_log_amplitude_plot2[i] = np.mean(relative_log_amplitude2[mask])

# Plot the results
plt.figure(figsize=(8, 6))

# Plot for image 1
plt.plot(frequencies, relative_log_amplitude_plot1, label='Image 1')

# Plot for image 2
plt.plot(frequencies, relative_log_amplitude_plot2, label='Image 2', linestyle='--')

# Customize the plot
plt.title('Relative Log Amplitude for Two Images')
plt.xlabel('Frequency (radians)')
plt.ylabel('Relative Log Amplitude')
plt.legend(title='Image')
plt.grid(True)
plt.xlim([0, np.pi])  # Limit x-axis to 0 to π
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], 
           ['0', 'π/4', 'π/2', '3π/4', 'π'])

# Show the plot
plt.show()
