# A widely used DATA-DRIVEN remaining useful life (RUL) prediction framework consists of 
# 1. Raw data collection
# 2. Feature extraction
# 3. Data-driven modeling, input (the statistical features) output (RUL, or RUL/End-of-life)
# 4. For testing specimen, repeat 1 & 2 and then calculate the output
# It is very important to note there are a lot of data-driven modeling methods, the selection is usually case-dependent.
# From simple to complex: Support vector machine [SVM], Gaussian process regression [GPR], Shallow neural networks, deep learning, etc.

# Let us consider there are three metallic specimens, each having one 'guided wave' sensor. Two specimen for training, one for testing.
# More specifically
# Step 1, Accoustic emission data, or more specifically, Guided wave data
# Step 2, Feature extraction, include one time-domain features, i.e., root-mean-square [RMS] value (same with the feature used in Waites demo, more to be added if neccessary)
# Step 3, GPR training: input - the RMS feature; output - RUL/EOL, which is also called as lifetime percentage [LP]
# Step 4, testing

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel as C

# Step 1, load the guided wave data
data1 = loadmat('E:\\AZ project\\AZ Data\\Signal_160kHz.mat') # data is stored in 'struct'
signal_data = data1['Signal_160kHz']

data2 = loadmat('E:\\AZ project\\AZ Data\\crack_data.mat') # data is stored in 'struct'
lc_data = data2['crack_growth']

lc = np.zeros((3,21))    # number of load cycles 
lp = np.zeros((3,21))    # lifetime percentage
SP = np.zeros((10000, 21, 3))
feature = np.zeros((3,21))

# print( lc_data['N'][0,5][0])

for i in range(3):

    SP[:,:,i] = signal_data['sig'][0,i+4]

# Step 2, extract the root-mean-square (RMS) feature for the training specimens
    start = 2930                                  # define a window for feature extraction
    duration = 501
    signal_baseline = SP[start:start+duration,0,i]
    rms_b = np.sqrt(np.mean(signal_baseline**2))  # Calculate RMS

    for j in range(21):
        signal_t = SP[start:start+duration,j,i]
        rms_t = np.sqrt(np.mean(signal_t**2))     # Calculate RMS
        feature[i,j] = 1 - rms_t/rms_b
        lc[i,j] = lc_data['N'][0,i+4][j]          # load cycle step

plt.figure(figsize=(8, 4))  # Initialize a new figure window
plt.plot(SP[:,0,1], label='Healthy')
plt.plot(SP[:,10,1], label='Damaged')
plt.title('Accoustic emission (sensor) data from one specimen')
plt.xlabel('Sample Index')
plt.ylabel('Sensor signal')

plt.figure(figsize=(8, 4))  # Initialize a new figure window
plt.plot(lc[0, :], feature[0, :], label='Specimen 1')
plt.plot(lc[1, :], feature[1, :], label='Specimen 2')
plt.plot(lc[2, :], feature[2, :], label='Specimen 3')
plt.title('Feature extraction')
plt.xlabel('Load cycle step [Can be transformed into time scale]')
plt.ylabel('Root-mean-square feature')
plt.legend()


# Step 3, train a GPR model. Note that a GPR model is often sufficient if the sensor data is 'of high quality'
for i in range(3):
    lp[i,:] =  1 - lc[i,:]/lc[i,20] # lifetime percentage

# plt.figure(figsize=(8, 4))  # Initialize a new figure window
# plt.plot(lp[0, :], feature[0, :], label='Specimen 1')
# plt.plot(lp[1, :], feature[1, :], label='Specimen 2')
# plt.plot(lp[2, :], feature[2, :], label='Specimen 3')
# plt.title('RMS features at different LPs')
# plt.xlabel('Lifetime percentage, LP')
# plt.ylabel('Root-mean-square feature')
# plt.legend()
# plt.show()

# First two specimens for training, Re-prepare the training data with corrected dimensions
x_train = feature[:2, :].reshape(-1, 1)  # Use all entries from the first two specimens as training data
y_train = lp[:2, :].reshape(-1, 1)       # Reshape to ensure 2D array for compatibility with sklearn

# plt.figure(figsize=(8, 4))  # Initialize a new figure window
# plt.plot(x_train, y_train)
# plt.show()

# Instantiate a Gaussian Process Regressor without explicitly defining a kernel
kernel = C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Fit the model to the data
gpr.fit(x_train, y_train)

# Step 4, Third specimen for testing
# Re-prepare the testing data with corrected dimensions
x_test = feature[2, :].reshape(-1, 1)  # Use all entries from the third specimen as testing data
y_test = lp[2, :].reshape(-1, 1)       # Reshape to ensure 2D array
y_pred, sigma = gpr.predict(x_test, return_std=True)
print("Sigma values:", sigma)

plt.figure(figsize=(8, 4))
plt.scatter(x_test, y_test, color='black', label='Actual Values')
plt.plot(x_test, y_pred, color='blue', label='Predicted Values')
plt.fill_between(x_test.flatten(), 
                 (y_pred - 1.96 * sigma).flatten(), 
                 (y_pred + 1.96 * sigma).flatten(), 
                 color='lightblue', alpha=0.5, 
                 label='95% Confidence Interval')
plt.title('GPR Predictions with Confidence Interval')
plt.xlabel('RMS feature')
plt.ylabel('Lifetime percentage, LP')
plt.legend()
plt.show()