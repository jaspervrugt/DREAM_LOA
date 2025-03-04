function y = nash_cascade(x,problem)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% Nash-Cascade unit hydrograph -- series of linear reservoirs             %
%                                                                         %
% SYNOPSIS                                                                %
%  y = nash_cascade(x,problem)                                            %
% where                                                                   %
%   x            [input] 1x2 vector of parameter values; k and m          %
%   problem      [input] structure DREAM_LOA & 2nd argument fitness func  %
%    .t                  measurement times of precipitation               %
%    .tmax               simulation end time in days [= max(t)]           %
%    .P                  nx1 vector of daily precipitation (mm/d)         %
%   y            [outpt] nx1 vector of simulated outflow (discharge)      %
%                                                                         %
% MATLAB CODE                                                             %
%  © Written by Jasper A. Vrugt                                           %
%    University of California Irvine                                      %
%  Version 1.0    July 2016                                               %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

k = x(1); m = x(2);                     % Recession constant, # reservoirs
if k < 1
    warning(['nash_cascade:Recession constant < 1 day ' ...
        '--> numerical errors']);
end
W = zeros(problem.tmax,problem.tmax);   % Initialize matrix W
IUH = 1/(k*gamma(m)) * ...              % Instantaneous unit hydrograph
    (problem.t/k).^(m-1) .* ...
    exp(-problem.t/k);   
for t = 1:problem.tmax                  % Time loop
    W(t:problem.tmax,t) = ...           % Discharge according to UH 
        problem.P(t) * ...              % # 3 linear reservoirs    
        IUH(1:problem.tmax-(t-1)); 
end                                     % End of time loop
y = sum(W,2);                           % Daily discharge (mm/d)

end
