function chains = DREAM_LOA(prior,N,T,d,problem)
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

[delta,c,c_star,nCR,p_unit] = ...
    deal(3,0.1,1e-12,3,.2);             % Default values algorithmic pars
ind = nan(N,N-1);                       % Initialize indivdual chain matrix
for i = 1:N 
    ind(i,1:N-1) = setdiff(1:N,i);      % Each chain index other chains
end
CR = (1:nCR)/nCR;                       % Crossover values
pCR = ones(1,nCR)/nCR;                  % selection probabilities
chains = nan(T,d+1,N);                  % Initialie chains

X = prior(N,d);                         % Draw initial state of each chain            
f_X = nan(N,1);                         % Initialize fitness
for i = 1:N                             % Fitness of initial chain states
    f_X(i,1) = ...
        fitness(X(i,1:d),problem); 
end    
chains(1,1:d+1,1:N) = ...               % Store initial states and fitness
    reshape([X f_X]',1,d+1,N);          
Xp = nan(N,d); f_Xp = nan(N,1);         % Initialize children & fitness

for t = 2:T                             % Dynamic: Evolve N chains T-1 stps
    [~,draw] = sort(rand(N-1,N));       % Random permute 1,...,N-1 N times
    dX = zeros(N,d);                    % Set zero jump vector each chain
    lambda = unifrnd(-c,c,N,1);         % Draw N different lambda values
    for i = 1:N                         % Evolve each chain one step 
        D = randsample(1:delta,1);      % Select delta 
        r1 = ind(i,draw(1:D,i));        % Unpack r1
        r2 = ind(i,draw(D+1:2*D,i));    % Unpack r2; r1 n.e. r2 n.e. i
        cr = randsample(CR,1,true,pCR); % Draw at random crossover value
        A = find(rand(1,d) < cr);       % subset A dimensions to sample
        d_star = numel(A);              % Cardinality of A
        g_RWM = 2.38/sqrt(2*D*d_star);  % Jump rate RWM for D chains
        gamma = randsample( ...         % Select gamma: 80/20 mix def/unity
            [g_RWM 1],1,true, ...
            [1-p_unit p_unit]); 
        dX(i,A) = (1+lambda(i)) * ...   % ith jump differential evolution
            gamma*sum(X(r1,A)-X(r2,A),1) ...  
            + c_star*randn(1,d_star);                             
        Xp(i,1:d) = X(i,1:d) + ...      % ith proposal
            dX(i,1:d);                         
    % --> Add: correct Xp if out of bound <-- 
        f_Xp(i,1) = ...                 % Fitness of ith proposal
            fitness(Xp(i,1:d),problem);
        P_acc = f_Xp(i,1) >= f_X(i,1);  % Acceptance probability (0 or 1)
        if P_acc                        % True: Accept proposal
            X(i,1:d) = Xp(i,1:d); 
            f_X(i,1) = f_Xp(i,1); 
        end
    end                                 % End chain step
    chains(t,1:d+1,1:N) = ...           % Add current position & fitness
        reshape([X f_X]',1,d+1,N);          
    % --> Monitor convergence of sampled chains <-- 
    % [X,f_X] = outlier(X,f_X);         % Outlier detection and correction
end                                     % End dynamic part

end
