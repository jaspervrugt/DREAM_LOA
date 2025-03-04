# ####################################################################### #
#                                                                         #
#  DDDDD     RRRRR     EEEEEEE     AAA     MM    MM                       #
#  DDDDDD    RRRRRR    EEEEEEE    AAAAA    MM    MM                       #
#  DD  DD    RR   RR   EE        AA   AA   MMM  MMM                       #
#  DD   DD   RR  RR    EEEE      AA   AA   MMMMMMMM                       #
#  DD   DD   RRRRR     EEEE      AAAAAAA   MMM  MMM                       #
#  DD  DD    RR RR     EE        AAAAAAA   MM    MM LL      OOOO    AA    #
#  DDDDDD    RR  RR    EEEEEEE   AA   AA   MM    MM LL     OO  OO  A  A   #
#  DDDDD     RR   RR   EEEEEEE   AA   AA   MM    MM LL     OO  OO AAAAAA  #
#                                                   LL     OO  OO AA  AA  #
#                                                   LLLLLL  OOOO  AA  AA  #
#                                                                         #
# ####################################################################### #
#                                                                         #
# DREAM_LOA: DiffeRential Evolution Adaptive Metropolis algorithm for     #
# limits of acceptability (LOA). The LOA method was developed by Keith    #
# Beven (2006) as a means to confront/quantify the impact of nonaleatory  #
# error sources in on model parameter and predictive uncertainty of       #
# hydrologic models. The default implementation of this method uses brute #
# force - uniform - sampling of the parameter space. The DREAM_LOA method #
# uses the search capabilities of the DREAM algorithm to more efficiently #
# delinate the behavioral and non-behavioral parameter space. The         #
# transition kernel splits the chain in two parts with Markovian and non- #
# Markovian properties. After convergence, the chains will be resersible. #
# The DREAM_LOA algorithm is part of eDREAM Package. I would recommend    #
# using this much more elaborative toolbox instread.                      #
#                                                                         #
# ####################################################################### #
#                                                                         #
# SYNOPSIS                                                                #
#  chains = DREAM_LOA(prior,N,T,d,problem)                                #
# where                                                                   #
#   prior        [input] Function that returns initial chain states       #
#                        X = prior(N,d)                                   #
#   N            [input] # chains                                         #
#   T            [input] # samples in chain                               #
#   d            [input] # sampled parameters                             #
#   problem      [input] structure DREAM_LOA & 2nd argument fitness func  #
#    .y_obs              nx1 vector of training data record               #
#    .epsilon            nx1 vector of LOAs for each y_obs                #
#    .t                  measurement times of precipitation               #
#    .tmax               simulation end time in days [= max(t)]           #
#    .P                  nx1 vector of daily precipitation (mm/d)         #
#   chains       [outpt] Txd+1xN array of sampled chain trajectories      #
#                                                                         #
# ALGORITHM HAS BEEN DESCRIBED IN                                         #
#   Vrugt, J.A. and K.J. Beven (2018), Embracing equifinality with        #
#       efficiency: Limits of acceptability sampling using the            #
#       DREAM_{(LOA)} algorithm, Journal of Hydrology, 559, pp. 954-971,  #
#           https://doi.org/10.1016/j.hydrol.2018.02.026                  #
#                                                                         #
# ####################################################################### #
#                                                                         #
# COPYRIGHT (c) 2024  the author                                          #
#                                                                         #
#   This program is free software: you can modify it under the terms      #
#   of the GNU General Public License as published by the Free Software   #
#   Foundation, either version 3 of the License, or (at your option)      #
#   any later version                                                     #
#                                                                         #
#   This program is distributed in the hope that it will be useful, but   #
#   WITHOUT ANY WARRANTY; without even the implied warranty of            #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU      #
#   General Public License for more details                               #
#                                                                         #
# ####################################################################### #
#                                                                         #
#  PYTHON CODE:                                                           #
#  © Written by Jasper A. Vrugt using GPT-4 OpenAI's language model       #
#    University of California Irvine                                      #
#  Version 2.0    Dec 2024                                                #
#                                                                         #
# ####################################################################### #

import numpy as np                                      
import time, os, sys, random
import scipy.special as sp

def fitness(x, problem):
    """
    Computes the fitness of a parameter vector based on the Limits of Acceptability (LOA).
    
    Parameters:
    x (array-like): 1d array of model parameters.
    problem (dict): Dictionary containing the following problem data:
        - y_obs (array): n x 1 vector of observed data.
        - epsilon (array): n x 1 vector of LOAs for each y_obs.
        - t (array): measurement times of precipitation.
        - tmax (float): simulation end time in days (max(t)).
        - P (array): n x 1 vector of daily precipitation (mm/d).
        
    Returns:
    f (int): Fitness, the number of observations within the LOAs.
    """
    
    # Run the forward model using the parameter vector x
    y = Nash_Cascade_2(x)       # Forward model (defined elsewhere)
    
    # Compute the fitness: count observations within the LOAs
    f = np.sum(np.abs(problem['y_obs'] - y) <= problem['epsilon'])
    
    return f


def genparset(chain):
    # ####################################################################### #
    # This function generates a matrix P from sampled chain trajectories      #
    # ####################################################################### #
    """
    :param chain: A 3D numpy array of shape (T, d, N) where:
        T = number of samples, d = number of parameters, N = number of chains
    :return: A 2D numpy array P with shape (N*T, d) containing the chain data, sorted by sample ID.
    """
    T, d, N = chain.shape  # Get dimensions: #samples, #parameters, #chains

    if T == 0:
        return np.array([])  # If no samples, return an empty array
    else:
        id_ = np.arange(1, T + 1)  # ID for each chain sample (1, 2, ..., T)
        P = np.full((N * T, d + 1), np.nan)  # Initialize matrix P with NaNs

        for z in range(N):  # For each chain
            # Copy each chain's data to P and add sample IDs
            P[z * T:(z + 1) * T, 0:d] = chain[:, :, z]  # Parameters for chain z
            P[z * T:(z + 1) * T, d] = id_  # Sample IDs for chain z
        
        # Sort P based on the last column (the sample ID), and remove the ID column
        P_sorted = P[np.argsort(P[:, d]), :]  # Sort by the last column (sample ID)
        P_sorted = P_sorted[:, 0:d]  # Remove the ID column

        return P_sorted


def Boundary_handling(X, Par_info):
    # ####################################################################### #
    # This function checks that parameters are in prior bounds, corrects them #
    # ####################################################################### #
    """
    Parameters:
    X (numpy.ndarray): Required, N x d matrix of candidate points
    Par_info (dict): Required, dictionary containing parameter bounds (min/max) and boundary treatment method
    
    Returns:
    Xr (numpy.ndarray): N x d matrix with revised candidate points
    v (numpy.ndarray): N x 1 vector with 0 for in bound and 1 for out of bound
    """
    
    Xr = np.copy(X)                                         # Create a copy of X for the revised values
    if X.ndim == 1:
        N = 1
        d = len(Xr)
        mn = np.array(Par_info['min'])
        mx = np.array(Par_info['max'])
    else:
        N, d = X.shape
        mn = np.tile(Par_info['min'], (N, 1))                   # Lower bounds replicated N times
        mx = np.tile(Par_info['max'], (N, 1))                   # Upper bounds replicated N times

    v = np.zeros(N, dtype=bool)                             # Logical array indicating out-of-bound values
    
    # Positions where X is below lower bound or above upper bound
    id_l = np.where(X < mn)                                 # Smaller than lower bound
    id_u = np.where(X > mx)                                 # Larger than upper bound
    
    # Boundary handling options
    if Par_info['boundhandling'] == 'reflect':              # Reflection method
        Xr[id_l] = 2 * mn[id_l] - X[id_l]                   # Reflect below the lower bound
        Xr[id_u] = 2 * mx[id_u] - X[id_u]                   # Reflect above the upper bound
    elif Par_info['boundhandling'] == 'bound':              # Bound method
        Xr[id_l] = mn[id_l]                                 # Set to lower bound
        Xr[id_u] = mx[id_u]                                 # Set to upper bound
    elif Par_info['boundhandling'] == 'fold':               # Folding method
        Xr[id_l] = mx[id_l] - (mn[id_l] - X[id_l])          # Fold below the lower bound
        Xr[id_u] = mn[id_u] + (X[id_u] - mx[id_u])          # Fold above the upper bound
    elif Par_info['boundhandling'] == 'reject':             # Reject method
        o = np.zeros_like(X, dtype=bool)                    # Initialize out-of-bound array
        o[id_l] = 1                                         # Mark positions below the lower bound
        o[id_u] = 1                                         # Mark positions above the upper bound
        v = np.sum(o, axis = 1) > 0                           # Identify rows with any out-of-bound values
    
    # Reflection or folding: Check if all elements are within bounds
    # Both methods can go out of bounds if violation exceeds |mx - mn|
    if Par_info['boundhandling'] in ['reflect', 'fold']:
        id_l = np.where(Xr < mn)                            # Smaller than lower bound
        id_u = np.where(Xr > mx)                            # Larger than upper bound
        Xr[id_l] = np.random.uniform(mn[id_l], mx[id_l])    # Random draw in [mn, mx]
        Xr[id_u] = np.random.uniform(mn[id_u], mx[id_u])    # Random draw in [mn, mx]

    # BMA model training if applicable
    # if 'unit_simplex' in Par_info:
    #     wght_sum = np.sum(Xr[:N, :int(K)], axis = 1)
    #     Xr[:, :int(K)] = Xr[:, :int(K)] / wght_sum[:, np.newaxis]  # Normalize weights in the unit simplex

    return Xr


def Remove_outlier(method, DREAMPar, X, t, loglik, options):
    # ####################################################################### #
    # This function identifies and removes outlier chains                     #
    # ####################################################################### #

    # Initial flag
    flag = 0
    
    # Check if diagnostic Bayes is used
    if options['DB'] == 'yes':
        logPR = X[:DREAMPar['N'], DREAMPar['d'] + 1]
        if np.all(logPR > 0):  # --> outlier based on likelihood not prior
            y = np.mean(loglik, axis = 0)
            flag = 1
        else:
            y = logPR  # --> outlier first based on prior only
    else:
        y = np.mean(loglik, axis = 0)

    # Choose outlier detection method
    if DREAMPar['outlier'] == 'iqr':
        chain_out = iqr(y)
    elif DREAMPar['outlier'] == 'grubbs':
        chain_out = grubbs(y)
    elif DREAMPar['outlier'] == 'peirce':
        chain_out = peirce(y)
    elif DREAMPar['outlier'] == 'chauvenet':
        chain_out = chauvenet(y)
    
    # Number of outlier chains
    N_out = len(chain_out)
    if N_out > 0:
        outlier = np.column_stack([np.full(N_out, t), chain_out])
        
        # Select good chains to replace outliers
        chain_in = list(range(DREAMPar['N']))
        chain_in = [i for i in chain_in if i not in chain_out]
        chain_select = random.sample(chain_in, N_out)
        for j in range(N_out):
            # Replace loglikelihood of outlier chain with a selected chain
            if options['DB'] == 'no' or flag == 1:
                loglik[:, chain_out[j]] = loglik[:, chain_select[j]]
            
            # Replace the state of outlier chain with selected chain
            X[chain_out[j], :DREAMPar['d'] + 2] = X[chain_select[j], :DREAMPar['d'] + 2]
            
            # Write warning to file
            with open('warning_file.txt', 'a+') as fid:
                fid.write(f"{method} WARNING: Irreversible jump chain {chain_out[j]} at generation {t}\n")
    else:
        outlier = None
    
    return X, loglik, outlier


# Secondary functions for outlier detection
def iqr(data):
    
    Q1, Q3 = np.percentile(data, [75, 25])
    IQR = Q1 - Q3
    return np.where(data < (Q3 - 2 * IQR))[0]


def grubbs(data, alpha=0.05):

    # Number of samples (chains)
    N = len(data)   
    # Calculate Grubbs statistic (for minimum only - one-sided interval)
    G = (np.mean(data) - np.min(data)) / np.std(data)
    # Compute critical t value for one-sided interval (1 - alpha same result!)
    # t_crit = t.ppf(1 - alpha / N, N - 2) ** 2
    t_crit = t.ppf(alpha / N, N - 2) ** 2
    # Now calculate Grubbs critical value
    T_c = (N - 1) / np.sqrt(N) * np.sqrt(t_crit / (N - 2 + t_crit))
    # Check whether to reject null-hypothesis (whether the min is an outlier)
    if G > T_c:
        # Minimum of data is an outlier
        id_outlier = np.argmin(data)
    else:
        id_outlier = None
    
    return id_outlier


def peirce(data):
    # Peirce's table (r values for different sample sizes)
    peirce_r = np.array([
        [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [3, 1.196, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 1.383, 1.078, -1, -1, -1, -1, -1, -1, -1],
        [5, 1.509, 1.200, -1, -1, -1, -1, -1, -1, -1],
        [6, 1.610, 1.299, 1.099, -1, -1, -1, -1, -1, -1],
        [7, 1.693, 1.382, 1.187, 1.022, -1, -1, -1, -1, -1],
        [8, 1.763, 1.453, 1.261, 1.109, -1, -1, -1, -1, -1],
        [9, 1.824, 1.515, 1.324, 1.178, 1.045, -1, -1, -1, -1],
        [10, 1.878, 1.570, 1.380, 1.237, 1.114, -1, -1, -1, -1],
        [11, 1.925, 1.619, 1.430, 1.289, 1.172, 1.059, -1, -1, -1],
        [12, 1.969, 1.663, 1.475, 1.336, 1.221, 1.118, 1.009, -1, -1],
        [13, 2.007, 1.704, 1.516, 1.379, 1.266, 1.167, 1.070, -1, -1],
        [14, 2.043, 1.741, 1.554, 1.417, 1.307, 1.210, 1.120, 1.026, -1],
        [15, 2.076, 1.775, 1.589, 1.453, 1.344, 1.249, 1.164, 1.078, -1],
        [16, 2.106, 1.807, 1.622, 1.486, 1.378, 1.285, 1.202, 1.122, 1.039],
        [17, 2.134, 1.836, 1.652, 1.517, 1.409, 1.318, 1.237, 1.161, 1.084],
        [18, 2.161, 1.864, 1.680, 1.546, 1.438, 1.348, 1.268, 1.195, 1.123],
        [19, 2.185, 1.890, 1.707, 1.573, 1.466, 1.377, 1.298, 1.226, 1.158],
        [20, 2.209, 1.914, 1.732, 1.599, 1.492, 1.404, 1.326, 1.255, 1.190],
        [21, 2.230, 1.938, 1.756, 1.623, 1.517, 1.429, 1.352, 1.282, 1.218],
        [22, 2.251, 1.960, 1.779, 1.646, 1.540, 1.452, 1.376, 1.308, 1.245],
        [23, 2.271, 1.981, 1.800, 1.668, 1.563, 1.475, 1.399, 1.332, 1.270],
        [24, 2.290, 2.000, 1.821, 1.689, 1.584, 1.497, 1.421, 1.354, 1.293],
        [25, 2.307, 2.019, 1.840, 1.709, 1.604, 1.517, 1.442, 1.375, 1.315],
        [26, 2.324, 2.037, 1.859, 1.728, 1.624, 1.537, 1.462, 1.396, 1.336],
        [27, 2.341, 2.055, 1.877, 1.746, 1.642, 1.556, 1.481, 1.415, 1.356],
        [28, 2.356, 2.071, 1.894, 1.764, 1.660, 1.574, 1.500, 1.434, 1.375],
        [29, 2.371, 2.088, 1.911, 1.781, 1.677, 1.591, 1.517, 1.452, 1.393],
        [30, 2.385, 2.103, 1.927, 1.797, 1.694, 1.608, 1.534, 1.469, 1.411],
        [31, 2.399, 2.118, 1.942, 1.812, 1.710, 1.624, 1.550, 1.486, 1.428],
        [32, 2.412, 2.132, 1.957, 1.828, 1.725, 1.640, 1.567, 1.502, 1.444],
        [33, 2.425, 2.146, 1.971, 1.842, 1.740, 1.655, 1.582, 1.517, 1.459],
        [34, 2.438, 2.159, 1.985, 1.856, 1.754, 1.669, 1.597, 1.532, 1.475],
        [35, 2.450, 2.172, 1.998, 1.870, 1.768, 1.683, 1.611, 1.547, 1.489],
        [36, 2.461, 2.184, 2.011, 1.883, 1.782, 1.697, 1.624, 1.561, 1.504],
        [37, 2.472, 2.196, 2.024, 1.896, 1.795, 1.711, 1.638, 1.574, 1.517],
        [38, 2.483, 2.208, 2.036, 1.909, 1.807, 1.723, 1.651, 1.587, 1.531],
        [39, 2.494, 2.219, 2.047, 1.921, 1.820, 1.736, 1.664, 1.600, 1.544],
        [40, 2.504, 2.230, 2.059, 1.932, 1.832, 1.748, 1.676, 1.613, 1.556],
        [41, 2.514, 2.241, 2.070, 1.944, 1.843, 1.760, 1.688, 1.625, 1.568],
        [42, 2.524, 2.251, 2.081, 1.955, 1.855, 1.771, 1.699, 1.636, 1.580],
        [43, 2.533, 2.261, 2.092, 1.966, 1.866, 1.783, 1.711, 1.648, 1.592],
        [44, 2.542, 2.271, 2.102, 1.976, 1.876, 1.794, 1.722, 1.659, 1.603],
        [45, 2.551, 2.281, 2.112, 1.987, 1.887, 1.804, 1.733, 1.670, 1.614],
        [46, 2.560, 2.290, 2.122, 1.997, 1.897, 1.815, 1.743, 1.681, 1.625],
        [47, 2.568, 2.299, 2.131, 2.006, 1.907, 1.825, 1.754, 1.691, 1.636],
        [48, 2.577, 2.308, 2.140, 2.016, 1.917, 1.835, 1.764, 1.701, 1.646],
        [49, 2.585, 2.317, 2.149, 2.026, 1.927, 1.845, 1.775, 1.711, 1.657],
        [50, 2.593, 2.326, 2.157, 2.035, 1.937, 1.855, 1.785, 1.721, 1.667],
	    [51, 2.600, 2.334, 2.167, 2.044, 1.945, 1.863, 1.792, 1.730, 1.675],
    	[52, 2.608, 2.342, 2.175, 2.052, 1.954, 1.872, 1.802, 1.740, 1.685],
    	[53, 2.615, 2.350, 2.184, 2.061, 1.963, 1.881, 1.811, 1.749, 1.694],
    	[54, 2.622, 2.358, 2.192, 2.069, 1.972, 1.890, 1.820, 1.758, 1.703],
    	[55, 2.629, 2.365, 2.200, 2.077, 1.980, 1.898, 1.828, 1.767, 1.711],
    	[56, 2.636, 2.373, 2.207, 2.085, 1.988, 1.907, 1.837, 1.775, 1.720],
    	[57, 2.643, 2.380, 2.215, 2.093, 1.996, 1.915, 1.845, 1.784, 1.729],
    	[58, 2.650, 2.387, 2.223, 2.101, 2.004, 1.923, 1.853, 1.792, 1.737],
    	[59, 2.656, 2.394, 2.230, 2.109, 2.012, 1.931, 1.861, 1.800, 1.745],
    	[60, 2.663, 2.401, 2.237, 2.116, 2.019, 1.939, 1.869, 1.808, 1.753] ])

    # Number of samples (chains)
    N = len(data)
    # Find the row index to use in the table for this sample
    if 2 < N < 61:
        n_ind = np.where(peirce_r[:, 0] == N)[0][0]
    else:
        if N >= 61:
            print("WARNING: DREAMPar.N > 60; using Peirce r-values for N = 60")
            # We continue with N = 60 (last row of peirce_r)
            n_ind = peirce_r.shape[0] - 1

    # Find the current r value
    r_curr = peirce_r[n_ind, 1]
    # One-sided interval! (thus negative distance)
    max_neg_dev_allowed = -r_curr * np.std(data)
    # Calculate distance to mean of each data point
    dev_L = data - np.mean(data)
    # Now apply the test (one-sided)
    id_outlier = np.where(dev_L < max_neg_dev_allowed)[0]

    return id_outlier


def chauvenet(data):

    # Number of samples (chains)
    N = len(data)
    # Calculate deviation from mean
    dev_L_ratio = (data - np.mean(data)) / np.std(data)
    # Define table with critical deviations
    n_sample = np.array([3, 4, 5, 6, 7, 10, 15, 25, 50, 100, 300, 500, 1000])
    max_dev_ratio = np.array([1.38, 1.54, 1.65, 1.73, 1.80, 1.96, 2.13, 2.33, 2.57, 2.81, 3.14, 3.29, 3.48])
    # Interpolate (linearly) the max deviation allowable (one-sided & negative)
    max_neg_dev_allowed = -np.interp(N, n_sample, max_dev_ratio)
    # Apply test (one-sided)
    id_outlier = np.where(dev_L_ratio < max_neg_dev_allowed)[0]
    
    return id_outlier


def Gelman(chain, t, method):
    # ####################################################################### #
    # This function computes univariate/multivariate scale reduction factors  #
    # ####################################################################### #
    """   
    For more information, refer to:
    Gelman, A. and D.R. Rubin, (1992) Inference from Iterative
    Simulation Using Multiple chains, Statistical Science, Volume 7, Issue 4, 457-472
    Brooks, S.P. and A. Gelman, (1998) General Methods for Monitoring
    Convergence of Iterative Simulations, Journal of Computational and Graphical Statistics, Volume 7, 434-455

    Arguments:
        chain (np.ndarray): The MCMC chain data (n, d, N)
        t (int): The iteration step number (not used directly in this function)
        method (str): The method used for calculations (for warnings)

    Returns:
        hatR (np.ndarray): Univariate Gelman-Rubin diagnostics for each parameter
        hatRd (float): Multivariate Gelman-Rubin diagnostic
    """

    n, d, N = chain.shape
    
    # Early exit if there are fewer than 10 iterations
    if n < 10:
        return np.nan * np.ones(d), np.nan

    # STEP 0: Compute the chain means and store in a N x d matrix
    mu_chains = np.mean(chain, axis = 0).T  # N x d
    # STEP 1: Compute the N within-chain variances
    s2_chains = np.array([np.var(chain[:, :, i], axis = 0, ddof = 1) for i in range(N)])  # N x d
    # STEP 2: Compute the N within-chain covariances
    cov_chains = np.array([np.cov(chain[:, :, i], rowvar=False) for i in range(N)])  # N x d x d
    
    # Univariate hatR diagnostics
    # STEP 1: Compute variance B of N chain means
    B = n * np.var(mu_chains, axis = 0)  # d
    # STEP 2: Compute 1xd vector W with mean of within-chain variances
    W = np.mean(s2_chains, axis = 0)  # d
    # STEP 3: Estimate target variance = sum of within- and between-chain s2's
    sigma2 = ((n - 1) / n) * W + (1 / n) * B
    # STEP 4: Compute univariate hatR diagnostic for each parameter
    hatR = np.sqrt((N + 1) / N * (sigma2 / W) - (n - 1) / (N * n))

    # Multivariate hatRd diagnostic
    # STEP 1: Compute dxd matrix with mean W of within-chain covariances
    W_cov = np.mean(cov_chains, axis = 0) + np.finfo(float).eps * np.eye(d)
    # STEP 2: Compute covariance B of N chain means
    B_cov = np.cov(mu_chains.T) + np.finfo(float).eps * np.eye(d)
    # Now check the covariance matrix, C, is it singular or not?
    if np.linalg.det(W_cov) == 0:
        with open("warning_file.txt", "a+") as fid:
            warning_message = (f"eDREAM_package WARNING: Singular covariance matrix detected: W_cov"
                               f"R-statistic of Brooks and Gelman at iteration {t}.\n")
            fid.write(warning_message)
        # Apply Tikhonov regularization
        W_inv = np.linalg.inv(W_cov + 1e-6 * np.eye(d))  
    else:
        W_inv = np.linalg.inv(W_cov)

    # STEP 3: Compute multivariate scale reduction factor, hatRd
    hatRd = np.sqrt((N + 1) / N * np.max(np.abs(np.linalg.eigvals(np.linalg.inv(W_cov) @ B_cov))) + (n - 1) / n)

    return hatR, hatRd


# Latin Hypercube Sampling function
def LH_sampling(mn, mx, N):
    # ####################################################################### #
    # This function performs Latin Hypercube sampling                         #
    # ####################################################################### #
    """
    Args:
        mn: Lower bound vector
        mx: Upper bound vector
        N: Number of samples to generate

    Returns:
        N x d matrix of Latin Hypercube samples
    """
    if len(mn.shape) == 2:
        d = mn.shape[1]                         # Number of parameters
    else:
        d = len(mn)
    rng = np.array(mx) - np.array(mn)       # 1 x d vector with parameter ranges
    y =  np.random.rand(N, d)               # N x d matrix with uniform random labels
    # really important change below so that X stays in bound! as list is from 0 - N-1 rather than 1 to N
    id_matrix = 1 + np.argsort(np.random.rand(N, d), axis = 0)  # Random sort (1:N without replacement)
    M = (id_matrix - y) / N                 # Multiplier matrix (y introduces randomness)
    R = np.add(np.multiply(M, rng), mn)     # N x d matrix of stratified LH samples

    return R


############################
### Case study 1
############################
def Nash_Cascade_2(x):
    # Nash-Cascade unit hydrograph -- series of three linear reservoirs
    
    if not hasattr(Nash_Cascade_2, "initialized"):                      # Store local variables in memory
        Nash_Cascade_2.maxT = 25                                        # Maximum time       
        Nash_Cascade_2.P =  [ 10, 25, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, \
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]                      # Precipitation
        Nash_Cascade_2.Time = np.arange(1, Nash_Cascade_2.maxT + 1)     # Define Time
        Nash_Cascade_2.initialized = True                               # Flag to indicate that initialization is complete        
    
    # ------------
    # Model script
    # ------------
    k = x[0]
    n = x[1]
    if k < 1:                                                           # Write to screen
        print('Nash_Cascade_2: Recession constant < 1 day --> numerical errors possible')

    A = np.zeros((Nash_Cascade_2.maxT, Nash_Cascade_2.maxT))            # Define help matrix
    IUH = 1 / (k * sp.gamma(n)) * (Nash_Cascade_2.Time / k) \
        ** (n - 1) * np.exp(-Nash_Cascade_2.Time / k)                   # Instantaneous unit hydrograph
    for t in range(Nash_Cascade_2.maxT):                                # Loop over time
        id = np.arange(0, Nash_Cascade_2.maxT - t)                      # Define id
        A[t, t:Nash_Cascade_2.maxT] = Nash_Cascade_2.P[t] * IUH[id]     # Calculate flow

    sim_Q = np.sum(A, axis = 0)                                         # Now determine total flow
    
    return sim_Q


############################
### Case study 2
############################
def Lotka_Volterra(x):

    if not hasattr(Lotka_Volterra, "initialized"):          # Store local variables in memory
        Lotka_Volterra.y0 = [30, 4]                         # Initial conditions: initial population of prey and predator
        Lotka_Volterra.tout = np.arange(0, 20 + 1/12, 1/12) # Time vector: from 0 to 20 years with monthly intervals
        def dydt_func(y, t, alpha, beta, gamma_, delta):    # Define the Lotka-Volterra equations as a function
            dy1 = alpha * y[0] - beta * y[0] * y[1]         # y[0] = prey population, y[1] = predator population
            dy2 = -gamma_ * y[1] + delta * y[0] * y[1]
            return [dy1, dy2] + [0, 0, 0, 0]                # zeros(4,1) in MATLAB equivalent

        Lotka_Volterra.dydt = dydt_func                     # Store the dydt function
        Lotka_Volterra.initialized = True                   # Flag to indicate that initialization is complete        

    y = y0 + list(x)                                        # Unpack the parameters (alpha, beta, gamma_, delta) and add initial states
    result = odeint(dydt, y, tout, args=tuple(x))           # Solve the system of differential equations using odeint
    Y = result[1:, :2]                                      # Return only the prey and predator populations
    Y_vector = Y.T.reshape(-1,1)                            # Flatten to return as a one-dimensional vector

    return Y_vector.flatten()

