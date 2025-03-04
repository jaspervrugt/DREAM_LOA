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
#  Version 1.0    Dec 2024                                                #
#                                                                         #
# ####################################################################### #

import numpy as np                                      
import time, os, sys

# For parallel workers
module_path = os.getcwd()
if module_path not in sys.path:
    sys.path.append(module_path)

# Get the current working directory
parent_directory = os.path.join(module_path, 'miscellaneous')
sys.path.append(parent_directory)

# Import other functions
from DREAM_LOA_functions import *


## MAIN PROGRAM
def DREAM_LOA(prior, N, T, d, problem):

    # Default values algorithmic parameters
    delta, c, c_star, nCR, p_unit = 3, 0.1, 1e-12, 3, 0.2

    print('\n')
    print('  -------------------------------------------------------------------------------------------------              ')
    print('  DDDDDDDD   RRRRRR     EEEEEEEEEE     AAA     MMM        MMM      LLL        OOOOOOOO      AAA                  ')
    print('  DDDDDDDDD  RRREERRR   EEEEEEEEE     AAAAA    MMMM      MMMM      LLL       OOOOOOOOOO    AAAAA                 ')
    print('  DDD    DDD RRR   RRR  EEE          AAA AAA   MMMMM    MMMMM      LLL       OOO    OOO   AAA AAA                ')
    print('  DDD    DDD RRR   RRR  EEE         AAA   AAA  MMMMMM  MMMMMM      LLL       OOO    OOO  AAA   AAA               ')
    print('  DDD    DDD RRR  RRR   EEEEEE     AAA     AAA MMM MMMMMM MMM ---- LLL       OOO    OOO AAA     AAA     /^ ^\    ')
    print('  DDD    DDD RRR RRR    EEEEEE     AAAAAAAAAAA MMM  MMMM  MMM ---- LLL       OOO    OOO AAAAAAAAAAA    / 0 0 \   ')
    print('  DDD    DDD RRRRRR     EEE        AAA     AAA MMM   MM   MMM      LLL       OOO    OOO AAA     AAA    V\ Y /V   ')
    print('  DDD    DDD RRR  RRR   EEE        AAA     AAA MMM        MMM      LLL       OOO    OOO AAA     AAA     / - \    ')
    print('  DDDDDDDDD  RRR   RRR  EEEEEEEEE  AAA     AAA MMM        MMM      LLLLLLLLL OOOOOOOOOO AAA     AAA    /     |   ')
    print('  DDDDDDDD   RRR    RRR EEEEEEEEEE AAA     AAA MMM        MMM      LLLLLLLLL  OOOOOOOO  AAA     AAA    V__) ||   ')
    print('  -------------------------------------------------------------------------------------------------              ')
    print('  © Jasper A. Vrugt, University of California Irvine & GPT-4 OpenAI''s language model')
    print('    ________________________________________________________________________')
    print('    Version 1.0, Dec. 2025, Beta-release: MATLAB implementation is benchmark')
    print('\n')



    # Initialize individual chain matrix
    ind = np.full((N, N-1), np.nan)
    for i in range(N):
        ind[i, :] = np.setdiff1d(np.arange(N), i)  		    # Each chain index other chains

    ind = ind.astype(int)
    CR = (np.arange(1, nCR + 1)) / nCR  			        # Crossover values
    pCR = np.ones(nCR) / nCR  					            # Selection probabilities
    
    chains = np.full((T, d+1, N), np.nan)  			        # Initialize chains
    
    X = prior(N, d)                                         # Draw initial state of each chain

    f_X = np.full((N, 1), np.nan)  				            # Initialize fitness
    for i in range(N):  					                # Fitness of initial chain states
        f_X[i, 0] = fitness(X[i, :d], problem)
    
    # Store initial states and fitness
    chains[0, :d+1, :] = np.reshape(np.concatenate([X, f_X], axis=1).T, (1, d+1, N))
    
    Xp = np.full((N, d), np.nan)  				            # Initialize children
    f_Xp = np.full((N, 1), np.nan)  				        # Initialize fitness of children
    
    # Dynamic: Evolve N chains for T-1 steps
    for t in range(1, T):
        draw = np.argsort(np.random.rand(N-1, N), axis=0)  	# Random permute 1,...,N-1 N times
        dX = np.zeros((N, d))  					            # Set zero jump vector for each chain
        lambda_ = np.random.uniform(-c, c, (N, 1))  		# Draw N different lambda values
        
        for i in range(N):  					            # Evolve each chain one step
            D = np.random.choice(np.arange(1, delta+1))  	# Select delta
            r1 = ind[i, draw[:D, i]]  				        # Unpack r1
            r2 = ind[i, draw[D:2*D, i]]  			        # Unpack r2; r1 n.e. r2 n.e. i
            
            cr = np.random.choice(CR, p=pCR)  			    # Draw at random crossover value
            A = np.where(np.random.rand(1, d) < cr)[1]  	# Subset A dimensions to sample
            if len(A) == 0:
                A = np.random.permutation(d)[:1]
            
            d_star = len(A)  					            # Cardinality of A
            g_RWM = 2.38 / np.sqrt(2 * D * d_star)  		# Jump rate for RWM
            A = np.array(A).flatten()                       # Ensure 'A' is 1D     
            
            # Select gamma: 80/20 mix of default/unity
            gamma = np.random.choice([g_RWM, 1], p=[1-p_unit, p_unit])

            dX[i, A] = (1 + lambda_[i]) * gamma * np.sum(X[np.ix_(r1, A)] - X[np.ix_(r2, A)], axis = 0) + c_star * np.random.randn(d_star)
            Xp[i, :] = X[i, :] + dX[i, :]  			        # Proposal
            
            Xp[i, :] = Boundary_handling(Xp[i, :], problem)

            # Fitness of ith proposal
            f_Xp[i, 0] = fitness(Xp[i, :d], problem)
            
            P_acc = f_Xp[i, 0] >= f_X[i, 0]  			    # Acceptance probability (0 or 1)
            
            if P_acc:  # Accept proposal
                X[i, :d] = Xp[i, :]
                f_X[i, 0] = f_Xp[i, 0]
        
        # Store current position & fitness
        chains[t, :d+1, :] = np.reshape(np.concatenate([X, f_X], axis=1).T, (1, d+1, N))
        
        # Monitor convergence of sampled chains (optional outlier detection)
        # X, f_X = remove_outlier(X, f_X)  			        # Uncomment if outlier detection is needed
    
    return chains
