function f = fitness(x,problem)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %
% This function computes the fitness of a parameter vector                %
%                                                                         %
% SYNOPSIS                                                                %
%  f = fitness(x,problem)                                                 %
% where                                                                   %
%   x            [input] 1xd parameter vector                             %
%   problem      [input] structure DREAM_LOA & 2nd argument fitness func  %
%    .y_obs              nx1 vector of training data record               %
%    .epsilon            nx1 vector of LOAs for each y_obs                %
%    .t                  measurement times of precipitation               %
%    .tmax               simulation end time in days [= max(t)]           %
%    .P                  nx1 vector of daily precipitation (mm/d)         %
%   f            [outpt] fitness (= scalar), # observations within LOAs   %
%                                                                         %
% ALGORITHM HAS BEEN DESCRIBED IN                                         %
%   Vrugt, J.A. and K.J. Beven (2018), Embracing equifinality with        %
%       efficiency: Limits of acceptability sampling using the            %
%       DREAM_{(LOA)} algorithm, Journal of Hydrology, 559, pp. 954-971,  %
%           https://doi.org/10.1016/j.hydrol.2018.02.026                  %
%                                                                         %
%  MATLAB CODE                                                            %
%  © Written by Jasper A. Vrugt                                           %
%    University of California Irvine                                      %
%  Version 1.0    July 2016                                               %
%                                                                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %

y = nash_cascade(x,problem);            % Frwd model Eq. 1 of REF for x
f = sum(abs(problem.y_obs-y) ...
        <= problem.epsilon);            % Eq. 12, # observations in limits?

end
