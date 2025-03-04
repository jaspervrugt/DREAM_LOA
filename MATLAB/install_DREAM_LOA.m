% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% DiffeRential Evolution Adaptive Metropolis (DREAM) algorithm for        %
% Limits of Acceptability sampling using discharge simulation with the    %
% Nash-Cascade of three linear reservoirs with recession constant k and   %
% parameter m                                                             %
%                                                                         %
% SYNOPSIS                                                                %
%  chains = DREAM_LOA(prior,N,T,d,problem)                                %
% where                                                                   %
%   prior        [input] Function that returns initial chain states       %
%                        X = prior(N,d)                                   %
%   N            [input] # chains                                         %
%   T            [input] # samples in chain                               %
%   d            [input] # sampled parameters                             %
%   problem      [input] structure DREAM_LOA & 2nd argument fitness func  %
%    .y_obs              nx1 vector of training data record               %
%    .epsilon            nx1 vector of LOAs for each y_obs                %
%    .t                  measurement times of precipitation               %
%    .tmax               simulation end time in days [= max(t)]           %
%    .P                  nx1 vector of daily precipitation (mm/d)         %
%   chains       [outpt] Txd+1xN array of sampled chain trajectories      %
%                                                                         %
% ALGORITHM HAS BEEN DESCRIBED IN                                         %
%   Vrugt, J.A. and K.J. Beven (2018), Embracing equifinality with        %
%       efficiency: Limits of acceptability sampling using the            %
%       DREAM_{(LOA)} algorithm, Journal of Hydrology, 559, pp. 954-971,  %
%           https://doi.org/10.1016/j.hydrol.2018.02.026                  %
%                                                                         %
% MATLAB CODE                                                             %
%  © Written by Jasper A. Vrugt                                           %
%    University of California Irvine                                      %
%  Version 1.0    July 2016                                               %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
% FURTHER CHECKING                                                        %
%  Website:  http://faculty.sites.uci.edu/jasper                          %
%  Papers: http://faculty.sites.uci.edu/jasper/publications/              %
%  Google Scholar: https://scholar.google.com/citations?user=...          %
%                      zkNXecUAAAAJ&hl=nl                                 %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

% In root directory of DREAM_LOA
addpath(pwd,[pwd '/miscellaneous']);
% Now go to example directory; say example_1
cd example_1
% Now execute this example by typing in command prompt: "example_1" 

% After DREAM_LOA terminates, create a 2d matrix from 3D-chain array
% P = genparset(chain);
% This matrix will have all samples of the joint chains (thinning active)
% Burn-in of P is required to get posterior samples
% Last column of P lists the # limits satisfied (= fitness function)
