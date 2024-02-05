# Principle component analysis for fault detection
# Twenty sensors are used to evaluate if there is any fault, such as system failure or sensor failure

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

# Generate multi-dimensional data for healthy and damaged states
num_sensors = 20
data_healthy = np.random.normal(loc=50, scale=2, size=(10000, num_sensors))  # Data for healthy state
data_damaged = np.random.normal(loc=48, scale=2, size=(2000, num_sensors))   # Data for damaged state

# Combine into a single dataset
data_all = np.concatenate([data_healthy, data_damaged], axis=0)

# Create a time series
sampling_rate = 1000  # Sampling rate is 1000Hz
total_samples = len(data_all)
time = np.arange(total_samples) / sampling_rate  # Time axis

# Train PCA model on the first 500 data points of the healthy data
pca1 = PCA(n_components=5)       # The number of components must be less than the data dimension, i.e., ten
pca1.fit(data_all[:500, :])  # Fit using the first 500 samples from all sensors

# Apply PCA model to all data to compute T² and SPE values
data_all_scores = pca1.transform(data_all)
T2 = np.sum(data_all_scores**2, axis=1)
reconstruction = pca1.inverse_transform(data_all_scores)
SPE = np.sum((data_all - reconstruction)**2, axis=1)

# Determine thresholds for T² and SPE based on the 99th percentile
T2_threshold = np.percentile(T2[501:4000], 99.99)
SPE_threshold = np.percentile(SPE[501:4000], 99.99)

# Assuming damaged data immediately follows healthy data in the combined dataset
test_time = time[:len(data_all)]  # Corrected to include the entire dataset

plt.figure(1)
#plt.figure(figsize=(12, 8))
plt.plot(test_time[1:10000],data_healthy[1:10000,1], label='Healthy', color='blue')
plt.plot(test_time[10001:12000],data_damaged[1:10000,1], label='Damaged', color='red')
plt.xlabel('Time (seconds)')
plt.ylabel('Sensor signal (No unit)')
plt.legend()
plt.grid(True)
plt.tight_layout()



# Plot T² and SPE statistics over time
plt.figure(2)
#plt.figure(figsize=(12, 8))
# Plot for T²
plt.subplot(2, 1, 1)
plt.plot(test_time, T2, label='T²', color='blue')
plt.axhline(y=T2_threshold, color='red', linestyle='--', label='T² Threshold')
plt.title('T² Statistic Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('T² Value')
plt.legend()
plt.grid(True)

# Plot for SPE
plt.subplot(2, 1, 2)
plt.plot(test_time, SPE, label='SPE', color='green')
plt.axhline(y=SPE_threshold, color='red', linestyle='--', label='SPE Threshold')
plt.title('SPE Statistic Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('SPE Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()