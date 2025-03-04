# ####################################################################### #
#                                                                         #
#   EEEEEE  XX  XX   AAAA   MM   MM  PPPPPP  LL      EEEEEE        1111   #  
#   EE       XXXX   AA  AA  MMM MMM  PP  PP  LL      EE           11 11   #
#   EEEEE     XX    AA  AA  MMMMMMM  PPPPPP  LL      EEEEE       11  11   #
#   EE       XXXX   AAAAAA  MM   MM  PP      LL      EE              11   #
#   EEEEEE  XX  XX  AA  AA  MM   MM  PP      LLLLLL  EEEEEE          11   #
#                                                                         #
# ####################################################################### #
#                                                                         #
# Example 1: Nash-cascade unit hydrograph from paper                      #
#   Vrugt, J.A. and K.J. Beven (2018), Embracing equifinality with        #
#       efficiency: Limits of acceptability sampling using the            #
#       DREAM_{(LOA)} algorithm, Journal of Hydrology, 559, pp. 954-971,  #
#           https://doi.org/10.1016/j.hydrol.2018.02.026                  #
#                                                                         #
# ####################################################################### #

import sys
import os

# Get the current working directory
current_directory = os.getcwd()
# Go up one directory
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
# add this to path
sys.path.append(parent_directory)
# Add another directory
misc_directory = os.path.abspath(os.path.join(parent_directory, 'miscellaneous'))
# add this to path
sys.path.append(misc_directory)

import numpy as np
import matplotlib.pyplot as plt
from DREAM_LOA import DREAM_LOA, genparset
from Nash_Cascade_2 import Nash_Cascade_2

# Define model parameters
k = 4  # Recession constant (d)
m = 2  # Number of reservoirs (-)

# Call the forward model
y = Nash_Cascade_2([k, m])  						# Simulated discharge data (mm/d)
# Define the problem parameters
problem = {}
problem['y_obs'] = np.random.normal(y, 0.1 * y)  	# Training data = perturbed
problem['epsilon'] = 0.25 * problem['y_obs']  		# Limits of acceptability
problem['min'] = [1, 1]
problem['max'] = [10, 10]
problem['boundhandling'] = 'reflect'

# Define the prior distribution (uniform)
def prior(N, d):
    return np.random.uniform(1, 10, (N, d))

d = 2  		# Number of unknown parameters, [k, m]
N = 10  	# Number of Markov chains
T = 2500  	# Number of samples in each chain


if __name__ == '__main__':
	# Run DREAM_LOA to sample behavioural space
	chains = DREAM_LOA(prior, N, T, d, problem)  # Chain is a T x (d+1) x N array of samples + fitness

	# Generate parameter set
	P = genparset(chains)  # NxT x d+1 matrix
	nt = P.shape[0]  # nt = NxT
	P = P[int(np.ceil(3 * nt / 4)):nt, :d+1]  # Burn-in (use convergence diagnostics!)

	par_name = ['$k$', '$m$']
	for i in range(d):
		np_hist, edges = np.histogram(P[:, i], bins=12, density=True)
		p = 0.5 * (edges[:-1] + edges[1:])
		plt.subplot(1, 2, i+1)
		plt.bar(p, np_hist, align='center')
		plt.title(f'Parameter {par_name[i]}', fontsize = 20)
		if i == 0:
			plt.plot(k, 0, 'rx', markersize=15, markeredgewidth=3, linewidth=3, clip_on=False)
		else:
			plt.plot(m, 0, 'rx', markersize=15, markeredgewidth=3, linewidth=3, clip_on=False)

	plt.show()

	# Identify solutions that satisfy all limits (those within the LOAs)
	id_valid = P[:, d] == len(y)
	P_valid = P[id_valid, :d+1]

	# Plot the valid solutions
	plt.figure()
	for i in range(d):
		np_hist, edges = np.histogram(P_valid[:, i], bins = 12, density=True)
		p = 0.5 * (edges[:-1] + edges[1:])
		plt.subplot(1, 2, i+1)
		plt.bar(p, np_hist, align='center')
		plt.title(f'Parameter {par_name[i]}', fontsize = 20)
		if i == 0:
			plt.plot(k, 0, 'rx', markersize=15, markeredgewidth=3, linewidth=3, clip_on=False)
		else:
			plt.plot(m, 0, 'rx', markersize=15, markeredgewidth=3, linewidth=3, clip_on=False)

	plt.show()
