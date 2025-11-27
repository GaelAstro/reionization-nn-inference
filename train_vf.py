import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import glob

# ============================================================================
# DATA LOADING
# ============================================================================

DATA_FOLDER = "../2D_spectra_hann_freq_no_rsd"
PARAM_FILE = "../param.dat"

def load_file(filepath):
	"""
	Load a data file and check if it exists and is not empty.
	
	Parameters:
	-----------
	filepath : str
		Path to the file to load
	
	Returns:
	--------
	array or None : Data array if file is valid, None otherwise
	"""
	if not os.path.exists(filepath):
		return None
	elif os.path.getsize(filepath) == 0:
		return None
	else:
		file_list = np.loadtxt(filepath)
		return file_list


def load_simulation(sim_number, folder_name=DATA_FOLDER):
	"""
	Load the three frequency band files for a single simulation.
	Each simulation has power spectra for three frequency ranges:
	- 151-166 MHz
	- 166-181 MHz  
	- 181-196 MHz
	
	Parameters:
	-----------
	sim_number : int
		Simulation number (e.g., 10001)
	folder_name : str
		Directory containing the simulation files
	
	Returns:
	--------
	array or None : Stacked array of shape (3, 10, 10) containing the three 
	                frequency bands, or None if any file is missing/empty
	"""
	file1 = f"{folder_name}/simu{sim_number}_2d_ps_151_166MHZ.dat"
	file2 = f"{folder_name}/simu{sim_number}_2d_ps_166_181MHz.dat"
	file3 = f"{folder_name}/simu{sim_number}_2d_ps_181_196MHz.dat"
	
	data1, data2, data3 = load_file(file1), load_file(file2), load_file(file3)
	
	if data1 is None or data2 is None or data3 is None:
		return None
	
	data_file = np.stack([data1, data2, data3])
	return data_file


def load_all_simulations(first_sim, last_sim, del_sims_zero=True, alpha=5, 
                         folder_name=DATA_FOLDER):
	"""
	Load all valid simulations in a given range and split into training/test sets.
	
	Parameters:
	-----------
	first_sim : int
		First simulation number to load
	last_sim : int
		Last simulation number to load (inclusive)
	del_sims_zero : bool
		If True, exclude simulations with bins near zero
	alpha : int
		Split ratio - every alpha-th simulation goes to test set (default: 5)
	folder_name : str
		Directory containing simulation files
	
	Returns:
	--------
	tuple : ((X_train, X_test), (sim_num_train, sim_num_test))
		- X_train, X_test: arrays of shape (n_samples, 3, 10, 10)
		- sim_num_train, sim_num_test: lists of actual simulation numbers
	"""
	train_data, sim_num_train = [], []
	test_data, sim_num_test = [], []
	sim_nb_forbidden = []
	
	# Counter for valid simulations only
	n_effective = 1
	
	if del_sims_zero:
		sim_nb_forbidden = simulations_near_zero(first_sim, last_sim)[2]
	
	for sim_number in range(first_sim, last_sim + 1):
		
		# Skip simulations with problematic bins
		if sim_number in sim_nb_forbidden:
			continue
			
		data = load_simulation(sim_number)
		
		if data is not None:
			# Every alpha-th valid simulation goes to test set
			if n_effective % alpha == 0:
				test_data.append(data)
				sim_num_test.append(sim_number)
			else:
				train_data.append(data)
				sim_num_train.append(sim_number)
		
			n_effective += 1
	
	X_train, X_test = np.array(train_data), np.array(test_data)
	
	return (X_train, X_test), (sim_num_train, sim_num_test)


def load_all_parameters(first_sim, sim_num_train, sim_num_test, 
                        file_name=PARAM_FILE):
	"""
	Load astrophysical parameters for the simulations in training and test sets.
	
	The parameter file contains:
	- Column 0: Simulation ID
	- Column 1: f_X (X-ray heating efficiency)
	- Column 2: tau (star formation timescale in 10 Myr)
	- Column 3: Ratio (not used)
	- Column 4: log(M_vir) (minimum halo mass for star formation)
	- Column 5: f_esc (escape fraction of ionizing photons)
	
	Parameters:
	-----------
	first_sim : int
		First simulation number in the dataset
	sim_num_train : list
		List of simulation numbers in training set
	sim_num_test : list
		List of simulation numbers in test set
	file_name : str
		Path to parameter file
	
	Returns:
	--------
	tuple : (Y_train, Y_test)
		Arrays of shape (n_samples, 5) containing the parameters
	"""
	interval = first_sim - 1
	train_param = []
	test_param = []
	
	f = open(file_name, "r")
	for line in f:
		values = line.split()
		param_id = int(values[0])
		sim_id = param_id + interval
		params = [float(v) for v in values[1:]]
		
		if sim_id in sim_num_train:
			train_param.append(params)
		elif sim_id in sim_num_test:
			test_param.append(params)
	f.close()
	
	Y_train, Y_test = np.array(train_param), np.array(test_param)
	
	return Y_train, Y_test


def filter_simulations_by_parameters(sim_num_train, sim_num_test, Y_train, Y_test, 
                                     fixed_logM=None, fixed_fesc=None, tolerance=1e-6):
	"""
	Filter datasets to keep only simulations with specific parameter values.
	Useful for training on a subset with fixed log(M) or f_esc.
	
	Parameters:
	-----------
	sim_num_train, sim_num_test : list
		Original simulation number lists
	Y_train, Y_test : array
		Original parameter arrays
	fixed_logM : float or None
		If specified, keep only simulations with this log(M) value (column 3)
	fixed_fesc : float or None
		If specified, keep only simulations with this f_esc value (column 4)
	tolerance : float
		Tolerance for floating point comparison
	
	Returns:
	--------
	tuple : (new_sim_num_train, new_sim_num_test, new_Y_train, new_Y_test)
		Filtered lists and arrays
	"""
	new_sim_num_train = []
	new_sim_num_test = []
	new_Y_train = []
	new_Y_test = []
	
	# Filter training data
	for i in range(len(sim_num_train)):
		sim_num = sim_num_train[i]
		params = Y_train[i]
		
		keep_this_sim = True
		
		if fixed_logM is not None:
			if abs(params[3] - fixed_logM) > tolerance:
				keep_this_sim = False
		
		if fixed_fesc is not None:
			if abs(params[4] - fixed_fesc) > tolerance:
				keep_this_sim = False
		
		if keep_this_sim:
			new_sim_num_train.append(sim_num)
			new_Y_train.append(params)
	
	# Filter test data
	for i in range(len(sim_num_test)):
		sim_num = sim_num_test[i]
		params = Y_test[i]
		
		keep_this_sim = True
		
		if fixed_logM is not None:
			if abs(params[3] - fixed_logM) > tolerance:
				keep_this_sim = False
		
		if fixed_fesc is not None:
			if abs(params[4] - fixed_fesc) > tolerance:
				keep_this_sim = False
		
		if keep_this_sim:
			new_sim_num_test.append(sim_num)
			new_Y_test.append(params)
	
	new_Y_train = np.array(new_Y_train)
	new_Y_test = np.array(new_Y_test)
	
	print(f"  Before filtering: {len(sim_num_train)} train, {len(sim_num_test)} test")
	print(f"  After filtering:  {len(new_sim_num_train)} train, {len(new_sim_num_test)} test")
	
	return new_sim_num_train, new_sim_num_test, new_Y_train, new_Y_test


def show_unique_parameter_values(first_sim, last_sim):
	"""
	Display all unique values for each astrophysical parameter in the dataset.
	Useful for understanding the parameter space and choosing fixed values.
	
	Parameters:
	-----------
	first_sim, last_sim : int
		Range of simulations to analyze
	
	Returns:
	--------
	array : Combined parameter array from all simulations
	"""
	# Load all simulations
	(X_train, X_test), (sim_num_train, sim_num_test) = load_all_simulations(
		first_sim, last_sim, del_sims_zero=True
	)
	
	# Load parameters
	Y_train, Y_test = load_all_parameters(first_sim, sim_num_train, sim_num_test)
	
	# Combine all parameters
	all_params = np.vstack([Y_train, Y_test])
	
	param_names = ['f_X', 'tau', 'Ratio', 'log(M)', 'f_esc']
	
	print("\n" + "="*60)
	print("UNIQUE PARAMETER VALUES IN DATASET")
	print("="*60)
	print(f"Total simulations: {len(all_params)}")
	print("="*60)
	
	for i, name in enumerate(param_names):
		unique_vals = np.unique(all_params[:, i])
		print(f"\n{name} (column {i}):")
		print(f"  Number of unique values: {len(unique_vals)}")
		print(f"  Values: {unique_vals}")
		
		# Count occurrences
		print(f"  Counts:")
		for val in unique_vals:
			count = np.sum(np.abs(all_params[:, i] - val) < 1e-6)
			print(f"    {val}: {count} simulations")
	
	print("\n" + "="*60)
	
	# Show combinations of log(M) and f_esc
	print("\nCOMBINATIONS of log(M) and f_esc:")
	print("="*60)
	logM_vals = all_params[:, 3]
	fesc_vals = all_params[:, 4]
	
	unique_logM = np.unique(logM_vals)
	unique_fesc = np.unique(fesc_vals)
	
	print(f"\n{'log(M)':<10} {'f_esc':<10} {'Count':<10}")
	print("-"*30)
	
	for logM in unique_logM:
		for fesc in unique_fesc:
			mask = (np.abs(logM_vals - logM) < 1e-6) & (np.abs(fesc_vals - fesc) < 1e-6)
			count = np.sum(mask)
			if count > 0:
				print(f"{logM:<10.2f} {fesc:<10.2f} {count:<10}")
	
	print("="*60 + "\n")
	
	return all_params


# ============================================================================
# PREPROCESSING
# ============================================================================

def log_transform(array, column=-1):
	"""
	Apply base-10 logarithm transformation to data.
	Useful for parameters that vary over orders of magnitude.
	
	Parameters:
	-----------
	array : ndarray
		Input array to transform
	column : int
		Column index to transform. If -1, transform entire array
	
	Returns:
	--------
	array or None : Transformed array, or None if non-positive values exist
	"""
	array_transformed = array.copy()
	
	if column == -1:
		if array.min() <= 0:
			print("Array has non-positive values")
			return None
		return np.log10(array_transformed)
	else:
		if array[:, column].min() <= 0:
			print("This specific column has non-positive values")
			return None
		else:
			array_transformed[:, column] = np.log10(array[:, column])
		return array_transformed


def minmax_transform(array, column=-1):
	"""
	Apply min-max normalization: (x - min) / (max - min).
	Scales values to range [0, 1].
	
	Parameters:
	-----------
	array : ndarray
		Input array to normalize
	column : int
		Column index to normalize. If -1, normalize entire array
	
	Returns:
	--------
	tuple : (normalized_array, (min_val, max_val))
		Normalized array and the scaling parameters needed for inverse transform
	"""
	array_transformed = array.copy()
	
	if column == -1:
		min_val = array.min()
		max_val = array.max()
		if max_val == min_val:
			print("All values are the same, cannot normalize")
			return None, None
		return (array_transformed - min_val) / (max_val - min_val), (min_val, max_val)
	else:
		min_val = array[:, column].min()
		max_val = array[:, column].max()
		if max_val == min_val:
			print("All values in this column are the same, cannot normalize")
			return None, None
		array_transformed[:, column] = (
			(array[:, column] - min_val) / (max_val - min_val)
		)
		return array_transformed, (min_val, max_val)


def standardize_transform(array, column=-1):
	"""
	Apply standardization (z-score normalization): (x - mean) / std.
	Centers data around zero with unit variance.
	
	Parameters:
	-----------
	array : ndarray
		Input array to standardize
	column : int
		Column index to standardize. If -1, standardize entire array
	
	Returns:
	--------
	tuple : (standardized_array, (mean, std))
		Standardized array and the scaling parameters needed for inverse transform
	"""
	array_transformed = array.copy()
	
	if column == -1:
		mean = array.mean()
		std = array.std()
		if std == 0:
			print("Standard deviation is zero, cannot standardize")
			return None, None
		return (array_transformed - mean) / std, (mean, std)
	else:
		mean = array[:, column].mean()
		std = array[:, column].std()
		if std == 0:
			print("Standard deviation of this column is zero, cannot standardize")
			return None, None
		array_transformed[:, column] = (array[:, column] - mean) / std
		return array_transformed, (mean, std)


# ============================================================================
# INVERSE PREPROCESSING
# ============================================================================

def inverse_log_transform(array, column=-1):
	"""
	Apply inverse base-10 logarithm: 10^x.
	Reverses log_transform().
	
	Parameters:
	-----------
	array : ndarray
		Log-transformed array
	column : int
		Column index to inverse transform. If -1, transform entire array
	
	Returns:
	--------
	array : Array in original scale
	"""
	array_transformed = array.copy()
	
	if column == -1:
		return np.power(10, array_transformed)
	else:
		array_transformed[:, column] = np.power(10, array[:, column])
		return array_transformed


def inverse_minmax_transform(array, min_val, max_val, column=-1):
	"""
	Apply inverse min-max normalization: x * (max - min) + min.
	Reverses minmax_transform().
	
	Parameters:
	-----------
	array : ndarray
		Normalized array
	min_val, max_val : float
		Original min and max values used in forward transform
	column : int
		Column index to inverse transform. If -1, transform entire array
	
	Returns:
	--------
	array : Array in original scale
	"""
	array_transformed = array.copy()
	
	if column == -1:
		return array_transformed * (max_val - min_val) + min_val
	else:
		array_transformed[:, column] = (
			array[:, column] * (max_val - min_val) + min_val
		)
		return array_transformed


def inverse_standardize_transform(array, mean, std, column=-1):
	"""
	Apply inverse standardization: x * std + mean.
	Reverses standardize_transform().
	
	Parameters:
	-----------
	array : ndarray
		Standardized array
	mean, std : float
		Original mean and standard deviation used in forward transform
	column : int
		Column index to inverse transform. If -1, transform entire array
	
	Returns:
	--------
	array : Array in original scale
	"""
	array_transformed = array.copy()
	
	if column == -1:
		return array_transformed * std + mean
	else:
		array_transformed[:, column] = array[:, column] * std + mean
		return array_transformed


# ============================================================================
# BIN STRUCTURE VISUALIZATION WITH K-MODES
# ============================================================================

def get_k_values(box_size=300):
	"""
	Calculate k-mode values for the power spectrum bins.
	
	In a simulation box, k values are quantized: k = 2π/L * n, where L is 
	the box size and n is the mode number (1, 2, 3, ...).
	
	Parameters:
	-----------
	box_size : float
		Simulation box size in Mpc/h (default: 300)
	
	Returns:
	--------
	array : k values for bins 0-9 in units of h/Mpc
	"""
	k_fundamental = 2 * np.pi / box_size
	k_values = k_fundamental * np.arange(1, 11)  # modes n=1 to 10
	return k_values


def analyze_bins(first_sim, last_sim):
	"""
	Visualize the average power spectrum across all simulations.
	Shows log10(P(k)) for each frequency band with proper k-mode labels.
	
	Parameters:
	-----------
	first_sim, last_sim : int
		Range of simulations to analyze
	"""
	(X_train, X_test) = load_all_simulations(first_sim, last_sim, False)[0]
	
	all_data = np.vstack((X_train, X_test))  # Shape: (n_sims, 3, 10, 10)
	avg_data = np.mean(all_data, axis=0)  # Shape: (3, 10, 10)
	
	X1, X2, X3 = avg_data[0], avg_data[1], avg_data[2]
	
	k_values = get_k_values(box_size=300)
	
	fig, ax = plt.subplots(1, 3, figsize=(20, 6))
	
	# Plot each frequency band
	im0 = ax[0].imshow(np.log10(X1 + 1e-20), cmap="viridis", 
	                   extent=[k_values[0], k_values[-1], k_values[0], k_values[-1]],
	                   origin='lower', aspect='auto')
	im1 = ax[1].imshow(np.log10(X2 + 1e-20), cmap="viridis",
	                   extent=[k_values[0], k_values[-1], k_values[0], k_values[-1]],
	                   origin='lower', aspect='auto')
	im2 = ax[2].imshow(np.log10(X3 + 1e-20), cmap="viridis",
	                   extent=[k_values[0], k_values[-1], k_values[0], k_values[-1]],
	                   origin='lower', aspect='auto')
	
	ax[0].set_xlabel(r'$k_\perp$ [h Mpc$^{-1}$]', fontsize=12)
	ax[0].set_ylabel(r'$k_\parallel$ [h Mpc$^{-1}$]', fontsize=12)
	ax[0].set_title("Log Power Spectrum: 151-166 MHz", fontsize=14)
	
	ax[1].set_xlabel(r'$k_\perp$ [h Mpc$^{-1}$]', fontsize=12)
	ax[1].set_ylabel(r'$k_\parallel$ [h Mpc$^{-1}$]', fontsize=12)
	ax[1].set_title("Log Power Spectrum: 166-181 MHz", fontsize=14)
	
	ax[2].set_xlabel(r'$k_\perp$ [h Mpc$^{-1}$]', fontsize=12)
	ax[2].set_ylabel(r'$k_\parallel$ [h Mpc$^{-1}$]', fontsize=12)
	ax[2].set_title("Log Power Spectrum: 181-196 MHz", fontsize=14)
	
	plt.tight_layout()
	fig.subplots_adjust(right=0.9)
	cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
	fig.colorbar(im0, cax=cbar_ax, label="log₁₀ P(k)")
	
	plt.savefig("./images/all_avg_bins.png", dpi=150)
	plt.show()


def analyze_bins_near_zero(first_sim, last_sim, threshold=1e-15):
	"""
	Count how many simulations have near-zero values in each bin.
	Helps identify problematic bins affected by numerical issues.
	
	Parameters:
	-----------
	first_sim, last_sim : int
		Range of simulations to analyze
	threshold : float
		Power spectrum values below this are considered "near zero"
	"""
	(X_train, X_test) = load_all_simulations(first_sim, last_sim, False)[0]
	all_data = np.vstack((X_train, X_test))  # Shape: (n_sims, 3, 10, 10)
	
	# Count how many times each bin is below threshold
	count_below = (all_data < threshold).sum(axis=0)  # Shape: (3, 10, 10)
	
	X1, X2, X3 = count_below[0], count_below[1], count_below[2]
	
	k_values = get_k_values(box_size=300)
	
	fig, ax = plt.subplots(1, 3, figsize=(20, 6))
	
	im0 = ax[0].imshow(X1, cmap="viridis",
	                   extent=[k_values[0], k_values[-1], k_values[0], k_values[-1]],
	                   origin='lower', aspect='auto')
	im1 = ax[1].imshow(X2, cmap="viridis",
	                   extent=[k_values[0], k_values[-1], k_values[0], k_values[-1]],
	                   origin='lower', aspect='auto')
	im2 = ax[2].imshow(X3, cmap="viridis",
	                   extent=[k_values[0], k_values[-1], k_values[0], k_values[-1]],
	                   origin='lower', aspect='auto')
	
	ax[0].set_xlabel(r'$k_\perp$ [h Mpc$^{-1}$]', fontsize=12)
	ax[0].set_ylabel(r'$k_\parallel$ [h Mpc$^{-1}$]', fontsize=12)
	ax[0].set_title(f"Count below {threshold:.0e} (151-166 MHz)", fontsize=14)
	
	ax[1].set_xlabel(r'$k_\perp$ [h Mpc$^{-1}$]', fontsize=12)
	ax[1].set_ylabel(r'$k_\parallel$ [h Mpc$^{-1}$]', fontsize=12)
	ax[1].set_title(f"Count below {threshold:.0e} (166-181 MHz)", fontsize=14)
	
	ax[2].set_xlabel(r'$k_\perp$ [h Mpc$^{-1}$]', fontsize=12)
	ax[2].set_ylabel(r'$k_\parallel$ [h Mpc$^{-1}$]', fontsize=12)
	ax[2].set_title(f"Count below {threshold:.0e} (181-196 MHz)", fontsize=14)
	
	plt.tight_layout()
	fig.subplots_adjust(right=0.9)
	cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
	fig.colorbar(im0, cax=cbar_ax, label="Number of samples below threshold")
	
	plt.savefig("./images/count_bins_below_threshold.png", dpi=150)
	plt.show()


def analyze_simulations_near_zero(first_sim, last_sim, threshold=1e-15):
	"""
	Analyze the distribution of near-zero bins across simulations.
	Lists all simulations with problematic bins.
	
	Parameters:
	-----------
	first_sim, last_sim : int
		Range of simulations to analyze
	threshold : float
		Power spectrum values below this are considered "near zero"
	"""
	(X_train, X_test), (sim_num_train, sim_num_test) = load_all_simulations(
		first_sim, last_sim, False
	)
	all_data = np.vstack((X_train, X_test))  # Shape: (n_sims, 3, 10, 10)
	all_sim_numbers = sim_num_train + sim_num_test
	
	# Count bins below threshold for each simulation (out of 300 total)
	count_below = (all_data < threshold).sum(axis=(1, 2, 3))
	
	fig, ax = plt.subplots(1, 1, figsize=(20, 6))
	
	ax.hist(count_below, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
	ax.set_xlabel('Number of bins below threshold (out of 300 total)', fontsize=12)
	ax.set_ylabel('Number of simulations', fontsize=12)
	ax.set_title(f'Distribution of bins below {threshold:.0e} across all simulations', 
	             fontsize=14)
	ax.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig("./images/simulations_below_threshold.png", dpi=150)
	plt.show()
	
	print(f"\nStatistics for threshold = {threshold:.0e}:")
	
	# Find simulations with problematic bins
	sims_with_below = np.where(count_below > 0)[0]
	sims_all_below = np.where(count_below == 300)[0]
	
	print(f"\nTotal simulations with at least one bin below threshold: {len(sims_with_below)}")
	print(f"Simulations with ALL bins below threshold: {len(sims_all_below)}")
	
	print(f"\n{'Simulation':<15} {'Status'}")
	print("-" * 40)
	for idx in sims_with_below:
		actual_sim_number = all_sim_numbers[idx]
		if count_below[idx] == 300:
			status = "ALL BINS BELOW"
		else:
			status = f"{count_below[idx]}/300 bins"
		print(f"{actual_sim_number:<15} {status}")


def simulations_near_zero(first_sim, last_sim, threshold=1e-15):
	"""
	Return lists of simulations with bins below threshold.
	Used internally to exclude problematic simulations from training.
	
	Parameters:
	-----------
	first_sim, last_sim : int
		Range of simulations to check
	threshold : float
		Power spectrum values below this are considered "near zero"
	
	Returns:
	--------
	tuple : (sims_with_below, sims_all_below, sims_below)
		- sims_with_below: indices with at least one bin below threshold
		- sims_all_below: indices with all bins below threshold
		- sims_below: combined list of problematic simulation numbers
	"""
	(X_train, X_test), (sim_num_train, sim_num_test) = load_all_simulations(
		first_sim, last_sim, False
	)
	all_data = np.vstack((X_train, X_test))
	all_sim_numbers = sim_num_train + sim_num_test
	
	count_below = (all_data < threshold).sum(axis=(1, 2, 3))
	
	sims_with_below = np.where(count_below > 0)[0]
	sims_all_below = np.where(count_below == 300)[0]
	
	# Convert indices to actual simulation numbers
	sims_below = [all_sim_numbers[i] for i in sims_with_below] + \
	             [all_sim_numbers[i] for i in sims_all_below]
	
	return sims_with_below, sims_all_below, sims_below


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(
	first_sim, last_sim, affine_transform, neurons=[64, 32], ep=100, bach=32, 
	columns=[1,3,4], earlyStop=True, fixed_logM=None, fixed_fesc=None, 
	logTransform=True
):
	"""
	Train a neural network to predict astrophysical parameters from 21cm power spectra.
	
	This function handles the complete training pipeline:
	1. Load and filter data
	2. Apply preprocessing (log transform + normalization)
	3. Build and train neural network
	4. Evaluate performance and create visualizations
	5. Save model and preprocessing parameters
	
	Parameters:
	-----------
	first_sim, last_sim : int
		Range of simulations to use
	affine_transform : function
		Normalization function (minmax_transform or standardize_transform)
	neurons : list of int
		Number of neurons in each hidden layer (e.g., [64, 32])
	ep : int
		Maximum number of training epochs
	bach : int
		Batch size for training
	columns : list of int
		Which parameters to predict (0=f_X, 1=tau, 2=Ratio, 3=log(M), 4=f_esc)
	earlyStop : bool
		Whether to use early stopping and learning rate reduction
	fixed_logM : float or None
		If set, only use simulations with this log(M) value
	fixed_fesc : float or None
		If set, only use simulations with this f_esc value
	logTransform : bool
		Whether to apply log10 transform to f_X and tau before normalization
	
	Returns:
	--------
	tuple : (model, transform_info, history)
		- model: trained Keras model
		- transform_info: dict with all preprocessing parameters
		- history: training history dict
	"""
	param_names = ['f_X', 'tau', 'Ratio', 'log(M)', 'f_esc']
	param_units = ['No units', '10 Myr', 'No units', 'log10(M☉)', 'No units']
	
	# ========================================================================
	# STEP 1: LOAD DATA
	# ========================================================================
	print("="*50)
	print("LOADING DATA")
	print("="*50)
	
	(X_train, X_test), (sim_num_train, sim_num_test) = load_all_simulations(
		first_sim, last_sim, del_sims_zero=True
	)
	
	if len(X_train) == 0 or len(X_test) == 0:
		print("\nERROR: No data loaded")
		exit()
	
	print(f"{len(X_train)} training data loaded and {len(X_test)} validation data loaded")
	print(f"Total simulations: {len(X_train) + len(X_test)}")
	
	# Flatten power spectra for neural network input
	X_train = X_train.reshape(len(X_train), -1)
	X_test = X_test.reshape(len(X_test), -1)
	
	# Load corresponding parameters
	Y_train, Y_test = load_all_parameters(first_sim, sim_num_train, sim_num_test)
	
	# Filter by fixed parameter values if requested
	if fixed_logM is not None or fixed_fesc is not None:
		print(f"\nFiltering simulations: log(M)={fixed_logM}, f_esc={fixed_fesc}")
		sim_num_train, sim_num_test, Y_train, Y_test = filter_simulations_by_parameters(
			sim_num_train, sim_num_test, Y_train, Y_test, 
			fixed_logM=fixed_logM, fixed_fesc=fixed_fesc
		)
		
		# Reload the filtered simulations
		X_train_list = [load_simulation(sim_num).reshape(-1) for sim_num in sim_num_train]
		X_test_list = [load_simulation(sim_num).reshape(-1) for sim_num in sim_num_test]
		X_train = np.array(X_train_list)
		X_test = np.array(X_test_list)
		
		print(f"After filtering: {len(X_train)} training, {len(X_test)} validation")
	
	# ========================================================================
	# STEP 2: PREPROCESS DATA
	# ========================================================================
	print("\n" + "="*50)
	print("PREPROCESSING")
	print("="*50)
	
	X_train_transformed = X_train.copy()
	X_test_transformed = X_test.copy()
	Y_train_transformed = Y_train.copy()
	Y_test_transformed = Y_test.copy()
	
	# Store all preprocessing info for later inverse transform
	transform_info = {
		'affine_transform': affine_transform.__name__,
		'logTransform': logTransform,
		'log_columns': [],
		'Y_transforms': [],
		'X_transform': None,
		'columns': columns
	}
	
	if not logTransform:
		# Option 1: Only apply affine transform (no log)
		X_train_scaled, X_transform_params = affine_transform(X_train_transformed, column=-1)
		X_test_scaled = X_test_transformed.copy()
		
		# Apply same parameters to test data
		if affine_transform == standardize_transform:
			mean, std = X_transform_params
			X_test_scaled = (X_test_scaled - mean) / std
		else:  # minmax_transform
			min_val, max_val = X_transform_params
			X_test_scaled = (X_test_scaled - min_val) / (max_val - min_val)
		
		transform_info['X_transform'] = X_transform_params
		
		# Transform Y parameters
		Y_train_scaled = Y_train_transformed.copy()
		Y_test_scaled = Y_test_transformed.copy()
		
		for col in range(Y_train.shape[1]):
			result, Y_params = affine_transform(Y_train_scaled, column=col)
			
			if Y_params is None:
				print(f"  Warning: Column {col} ({param_names[col]}) has constant values, skipping normalization")
				transform_info['Y_transforms'].append(None)
				continue
			
			Y_train_scaled = result
			
			# Apply same parameters to test data
			if affine_transform == standardize_transform:
				mean, std = Y_params
				Y_test_scaled[:, col] = (Y_test_scaled[:, col] - mean) / std
			else:  # minmax_transform
				min_val, max_val = Y_params
				Y_test_scaled[:, col] = (
					(Y_test_scaled[:, col] - min_val) / (max_val - min_val)
				)
			
			transform_info['Y_transforms'].append(Y_params)
	
	else:
		# Option 2: Apply log transform first, then affine transform
		Y_train_transformed = log_transform(Y_train_transformed, column=0)
		Y_train_transformed = log_transform(Y_train_transformed, column=1)
		Y_test_transformed = log_transform(Y_test_transformed, column=0)
		Y_test_transformed = log_transform(Y_test_transformed, column=1)
		transform_info['log_columns'] = [0, 1]
		
		# Transform X
		X_train_scaled, X_transform_params = affine_transform(X_train_transformed, column=-1)
		X_test_scaled = X_test_transformed.copy()
		
		if affine_transform == standardize_transform:
			mean, std = X_transform_params
			X_test_scaled = (X_test_scaled - mean) / std
		else:  # minmax_transform
			min_val, max_val = X_transform_params
			X_test_scaled = (X_test_scaled - min_val) / (max_val - min_val)
		
		transform_info['X_transform'] = X_transform_params
		
		# Transform Y
		Y_train_scaled = Y_train_transformed.copy()
		Y_test_scaled = Y_test_transformed.copy()
		
		for col in range(Y_train.shape[1]):
			result, Y_params = affine_transform(Y_train_scaled, column=col)
			
			if Y_params is None:
				print(f"  Warning: Column {col} ({param_names[col]}) has constant values, skipping normalization")
				transform_info['Y_transforms'].append(None)
				continue
			
			Y_train_scaled = result
			
			if affine_transform == standardize_transform:
				mean, std = Y_params
				Y_test_scaled[:, col] = (Y_test_scaled[:, col] - mean) / std
			else:  # minmax_transform
				min_val, max_val = Y_params
				Y_test_scaled[:, col] = (
					(Y_test_scaled[:, col] - min_val) / (max_val - min_val)
				)
			
			transform_info['Y_transforms'].append(Y_params)
	
	# Select only the parameters we want to predict
	Y_train_scaled_final = Y_train_scaled[:, columns]
	Y_test_scaled_final = Y_test_scaled[:, columns]
	
	# ========================================================================
	# STEP 3: BUILD NEURAL NETWORK
	# ========================================================================
	print("\n" + "="*50)
	print("BUILDING NEURAL NETWORK")
	print("="*50)
	
	n_input = X_train_scaled.shape[1]    # 300 (flattened power spectrum)
	n_output = Y_train_scaled_final.shape[1]  # Number of parameters to predict
	
	model = keras.Sequential()
	model.add(layers.Dense(neurons[0], activation='relu', input_shape=(n_input,)))
	
	for n_neurons in neurons[1:]:
		model.add(layers.Dense(n_neurons, activation='relu'))
	
	model.add(layers.Dense(n_output, activation='linear'))
	
	model.compile(
		optimizer=keras.optimizers.Adam(0.001),
		loss='mse',
		metrics=['mae']
	)
	
	# ========================================================================
	# STEP 4: TRAIN MODEL
	# ========================================================================
	print("\n" + "="*50)
	print("TRAINING MODEL")
	print("="*50)
	
	if earlyStop:
		callbacks = [
			EarlyStopping(
				monitor='val_loss',
				patience=100,
				restore_best_weights=True
			),
			keras.callbacks.ReduceLROnPlateau(
				monitor='val_loss',
				factor=0.5,
				patience=30,
				min_lr=1e-7,
				verbose=1
			)
		]
		
		history = model.fit(
			X_train_scaled, Y_train_scaled_final,
			epochs=ep,
			batch_size=bach,
			shuffle=True,
			validation_data=(X_test_scaled, Y_test_scaled_final),
			callbacks=callbacks,
			verbose=1
		)
	else:
		history = model.fit(
			X_train_scaled, Y_train_scaled_final,
			epochs=ep,
			batch_size=bach,
			shuffle=True,
			validation_data=(X_test_scaled, Y_test_scaled_final),
			verbose=1
		)
	
	# ========================================================================
	# STEP 5: MAKE PREDICTIONS AND INVERSE TRANSFORM
	# ========================================================================
	print("\n" + "="*50)
	print("INVERSE TRANSFORM & EVALUATION")
	print("="*50)
	
	predictions_scaled = model.predict(X_test_scaled, batch_size=None, verbose=0)
	
	# Inverse affine transform
	predictions_transformed = predictions_scaled.copy()
	for i, col in enumerate(columns):
		params = transform_info['Y_transforms'][col]
		
		if params is None:
			continue
		
		if affine_transform == standardize_transform:
			mean, std = params
			predictions_transformed = inverse_standardize_transform(
				predictions_transformed, mean, std, column=i
			)
		else:  # minmax_transform
			min_val, max_val = params
			predictions_transformed = inverse_minmax_transform(
				predictions_transformed, min_val, max_val, column=i
			)
	
	# Inverse log transform if needed
	predictions_original = predictions_transformed.copy()
	for i, col in enumerate(columns):
		if col in transform_info['log_columns']:
			predictions_original = inverse_log_transform(predictions_original, column=i)
	
	# Print final performance metrics
	min_val_loss = min(history.history['val_loss'])
	min_val_mae = min(history.history['val_mae'])
	best_epoch = history.history['val_loss'].index(min_val_loss) + 1
	best_epoch_mae = history.history['val_mae'].index(min_val_mae) + 1
	
	print(f"Best Validation Loss: {min_val_loss:.6f} at epoch {best_epoch}")
	print(f"Best percentage error: {min_val_mae*100:.3f}% at epoch {best_epoch_mae}")
	print("="*50)
	
	# ========================================================================
	# STEP 6: VISUALIZATION
	# ========================================================================
	print("="*50)
	print("Creating plots")
	print("="*50)
	
	param_names_final = [param_names[i] for i in columns]
	param_units_final = [param_units[i] for i in columns]
	Y_test_subset = Y_test[:, columns]
	
	error_array = Y_test_subset - predictions_original
	
	# Create figure with training curves and prediction plots
	n_cols = max(len(param_names_final), 2)
	fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
	
	if n_cols == 1:
		axes = axes.reshape(2, 1)
	
	# Top row: Training curves
	axes[0, 0].loglog(history.history['loss'], label='Training Loss', linewidth=2)
	axes[0, 0].loglog(history.history['val_loss'], label='Validation Loss', linewidth=2)
	axes[0, 0].set_xlabel('Epoch', fontsize=10)
	axes[0, 0].set_ylabel('Loss (MSE)', fontsize=10)
	axes[0, 0].set_title('Model Loss', fontsize=12, pad=10)
	axes[0, 0].legend(fontsize=8)
	axes[0, 0].grid(True, alpha=0.3)
	
	axes[0, 1].loglog(history.history['mae'], label='Training MAE', linewidth=2)
	axes[0, 1].loglog(history.history['val_mae'], label='Validation MAE', linewidth=2)
	axes[0, 1].set_xlabel('Epoch', fontsize=10)
	axes[0, 1].set_ylabel('MAE', fontsize=10)
	axes[0, 1].set_title('Model MAE', fontsize=12, pad=10)
	axes[0, 1].legend(fontsize=8)
	axes[0, 1].grid(True, alpha=0.3)
	
	for j in range(2, n_cols):
		axes[0, j].axis('off')
	
	# Bottom row: Prediction vs True value plots
	for i in range(len(param_names_final)):
		X_values = Y_test_subset[:, i]
		Y_values = predictions_original[:, i]
		
		# Get the SCALED values for this specific parameter
		X_values_scaled = Y_test_scaled_final[:, i]
		Y_values_scaled = predictions_scaled[:, i]
		
		# Compute error metrics for this parameter in SCALED space
		rmse_scaled = np.sqrt(np.mean((X_values_scaled - Y_values_scaled)**2))*100
		
		min_val = min(X_values.min(), Y_values.min())
		max_val = max(X_values.max(), Y_values.max())
		margin = 0.05 * (max_val - min_val)
		min_val -= margin
		max_val += margin
		
		axes[1, i].plot(X_values, Y_values, "b+", markersize=2)
		axes[1, i].plot([min_val, max_val], [min_val, max_val], 'r--', 
			            alpha=0.5, linewidth=1)
		axes[1, i].set_xlim(min_val, max_val)
		axes[1, i].set_ylim(min_val, max_val)
		axes[1, i].set_xlabel(f"True values ({param_units_final[i]})", fontsize=10)
		axes[1, i].set_ylabel(f"Predictions ({param_units_final[i]})", fontsize=10)
		axes[1, i].set_title(param_names_final[i], fontsize=12, pad=10)
		axes[1, i].set_aspect('equal', adjustable='box')
		axes[1, i].grid(True, alpha=0.3)
		
		# Add error metrics as text in the plot (in scaled space)
		textstr = f'RMSE: {rmse_scaled:.2f}%'
		axes[1, i].text(0.05, 0.95, textstr, transform=axes[1, i].transAxes,
			            fontsize=9, verticalalignment='top',
			            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
		
	for k in range(len(param_names_final), n_cols):
		axes[1, k].axis('off')
	
	plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
	
	# Generate filename with timestamp
	columns_str = '_'.join(map(str, columns))
	neurons_str = '_'.join(map(str, neurons))
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	
	plt.savefig(
		f"./images/{timestamp}_error{min_val_mae*100:.3f}%_earlyStop{earlyStop}_cols{columns_str}_ep{ep}_neurons{neurons_str}_batch{bach}.png", 
		dpi=150, bbox_inches='tight'
	)
	plt.show()
	
	# ========================================================================
	# STEP 7: ERROR CORRELATION ANALYSIS
	# ========================================================================
	n_params = len(param_names_final)
	fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
	
	if n_params == 1:
		axes = [axes]
	
	for j in range(n_params):
		x_error = error_array[:, j]
		y_error = error_array[:, (j+1) % n_params]
		
		# Standardize errors for visualization
		mean_x = x_error.mean()
		std_x = x_error.std()
		mean_y = y_error.mean()
		std_y = y_error.std()
		
		x_standardized = (x_error - mean_x) / std_x
		y_standardized = (y_error - mean_y) / std_y
		
		limits = max(np.abs(x_standardized).max(), np.abs(y_standardized).max())
		
		pearson_corr = np.corrcoef(x_error, y_error)[0, 1]
		
		axes[j].scatter(x_standardized, y_standardized, alpha=0.8, s=1, 
		                color='steelblue', edgecolors='none')
		
		axes[j].set_xlabel(f"{param_names_final[j]} error (standardized)", fontsize=11)
		axes[j].set_ylabel(f"{param_names_final[(j+1) % n_params]} error (standardized)", 
		                   fontsize=11)
		axes[j].set_title(
			f"Error: {param_names_final[j]} vs {param_names_final[(j+1) % n_params]}\nPearson: {pearson_corr:.3f}", 
			fontsize=11, pad=10
		)
		
		axes[j].set_xlim(-limits, limits)
		axes[j].set_ylim(-limits, limits)
		axes[j].grid(True, alpha=0.3)
	
	plt.tight_layout(pad=2.0, w_pad=2.0)
	plt.savefig(
		f"./images/{timestamp}_error_earlyStop{earlyStop}_cols{columns_str}_ep{ep}_neurons{neurons_str}_batch{bach}.png", 
		dpi=150, bbox_inches='tight'
	)
	plt.show()
	
	# ========================================================================
	# STEP 8: SAVE MODEL AND METADATA
	# ========================================================================
	model_name = f"./models/{timestamp}_model_{affine_transform.__name__}_log{logTransform}_cols{columns_str}_ep{ep}_neurons{neurons_str}_batch{bach}.keras"
	transform_name = f"./models/{timestamp}_transform_{affine_transform.__name__}_log{logTransform}_cols{columns_str}_ep{ep}_neurons{neurons_str}_batch{bach}.pkl"
	history_name = f"./models/{timestamp}_history_{affine_transform.__name__}_log{logTransform}_cols{columns_str}_ep{ep}_neurons{neurons_str}_batch{bach}.pkl"
	
	model.save(model_name)
	with open(transform_name, 'wb') as f:
		pickle.dump(transform_info, f)
	with open(history_name, 'wb') as f:
		pickle.dump(history.history, f)
	
	print("\n" + "="*50)
	print(f"Model saved as '{model_name}'")
	print(f"Transform info saved as '{transform_name}'")
	print(f"History saved as '{history_name}'")
	print("="*50)
	
	return model, transform_info, history


if __name__ == "__main__":
	train_model(
		10001, 19823, 
		minmax_transform, 
		[256, 256], 
		800, 
		2, 
		[1],
		earlyStop=True, 
		fixed_logM=9.07, 
		fixed_fesc=0.275, 
		logTransform=True
	)
