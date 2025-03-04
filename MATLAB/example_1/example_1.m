% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
%   EEEEEE  XX  XX   AAAA   MM   MM  PPPPPP  LL      EEEEEE        1111   %  
%   EE       XXXX   AA  AA  MMM MMM  PP  PP  LL      EE           11 11   %
%   EEEEE     XX    AA  AA  MMMMMMM  PPPPPP  LL      EEEEE       11  11   %
%   EE       XXXX   AAAAAA  MM   MM  PP      LL      EE              11   %
%   EEEEEE  XX  XX  AA  AA  MM   MM  PP      LLLLLL  EEEEEE          11   %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
%                                                                         %
% Example 1: Nash-cascade unit hydrograph from paper                      %
%   Vrugt, J.A. and K.J. Beven (2018), Embracing equifinality with        %
%       efficiency: Limits of acceptability sampling using the            %
%       DREAM_{(LOA)} algorithm, Journal of Hydrology, 559, pp. 954-971,  %
%           https://doi.org/10.1016/j.hydrol.2018.02.026                  %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

clc; clear; close all hidden;               % clear workspace and figures

problem.tmax = 25;                          % Max simulation time (days)
problem.P = [ 10 25 8 2 zeros(1,21) ]';     % 25-day precipitation record
problem.t = 1:problem.tmax;                 % Times (days) simulated output
k = 4;                                      % Recession constant (d)
m = 2;                                      % # reservoirs (-)
y = nash_cascade([k m],problem);            % Simultd discharge data (mm/d)
problem.y_obs = normrnd(y,0.1*y);           % Training data = perturbed 
problem.epsilon = 0.25 * problem.y_obs;     % Limits of acceptability
prior = @(N,d) unifrnd(1,10,N,d);           % Handle for prior distribution
d = 2;                                      % # unknown parameters, [k m]
N = 10;                                     % # Markov chains
T = 2500;                                   % # samples in each chain

% Run DREAM_LOA to sample behavioural space
chains = DREAM_LOA(prior,N,T,d,problem);     
% chain is a Txd+1xN array of chain samples + fitness
% fitness is equal to # limits satisfied

P = genparset(chains);                      % NxT x d+1 matrix
nt = size(P,1);                             % nt = NxT
P = P(ceil(3*nt/4):nt,1:d+1);               % Burn-in (use conv dgnstics!!)
par_name = {'$k$','$m$'};
for i = 1:d
    [np,edges] = histcounts(P(:,i),12, ...
        'Normalization','pdf');
    p = 1/2*(edges(1:end-1)+edges(2:end));
    subplot(1,2,i),bar(p,np); hold on;
    title(strcat('Parameter',{' '}, ...
        par_name(i)),'interpreter', ...
        'latex','fontsize',20);
    if i == 1
        plot(k,0,'rx','markersize',15, ...
            'linewidth',3);
    else
        plot(m,0,'rx','markersize',15, ...
            'linewidth',3);
    end        
end

% 1. outlier chains persist         [= resolved in DREAM Package]
% 2. no convergence diagnostics     [= resolved in DREAM Package]
% 3. no visual/tabulated results    [= resolved in DREAM Package]

% As a brute-force solution we simply can take all solutions of P that
% satify all the limits; 

P = genparset(chains);                      % NxT x d+1 matrix
id = P(1:nt,d+1) == numel(y);
P = P(id,1:d+1);
figure(2)
for i = 1:d
    [np,edges] = histcounts(P(:,i),12, ...
        'Normalization','pdf');
    p = 1/2*(edges(1:end-1)+edges(2:end));
    subplot(1,2,i),bar(p,np); hold on;
    title(strcat('Parameter',{' '}, ...
        par_name(i)),'interpreter', ...
        'latex','fontsize',20);
    if i == 1
        plot(k,0,'rx','markersize',15, ...
            'linewidth',3);
    else
        plot(m,0,'rx','markersize',15, ...
            'linewidth',3);
    end        
end


