function [MAE,fx] = holtwinters(param,s,x)
%HOLTWINTERS Compute forecasts of the Holt-Winters exponential smoothing model.
%   [MAE,FX]=HOLTWINTERS(PARAM,S,X) returns the Mean Absolute Error (MAE) 
%   of 1-step ahead forecasts (FX) of the Holt-Winters model with additive 
%   trend and additive seasonality for parameter vector PARAM = [alpha, beta,
%   gamma], seasonality of period S and data series X.
%
% Sample use:
%   % Load sample data (with 7-day periodocity)
%   x = load ...
%   % Estimate parameters
%   initial_param = [.5 .5 .5]; period = 7;
%   [param] = fminsearch(@(param) holtwinters(param,period,x),initial_param);
%   % Compute forecasts (fx)
%   [MAE,fx] = holtwinters(param,s,p);
%
% Reference(s): 
%   [1] E. S. Gardner Jr. (2006) Exponential smoothing: The state of the 
%   art - Part II, International Journal of Forecasting 22, 637-666.

%   Written by Rafal Weron (2017.04.24)

% Recover alpha, beta and gamma from PARAM
alpha = param(1);
beta = param(2);
gamma = param(3);

% Initialize L, T, S and FX vectors
L = zeros(size(x)); 
T = L; 
S = L;
fx = L;

% Set initial values of L, T and S
L(s) = sum(x(1:s))/s;
T(s) = sum(x(s+1:2*s) - x(1:s))/s^2;
S(1:s) = x(1:s) - L(s);

% Iterate to compute L(t), T(t), S(t) and FX(t) 
for t=(s+1):(length(x)-1)
    L(t) = alpha*(x(t)-S(t-s)) + (1-alpha)*(L(t-1)+T(t-1));
    T(t) = beta*(L(t)-L(t-1)) + (1-beta)*T(t-1);
    S(t) = gamma*(x(t)-L(t)) + (1-gamma)*S(t-s);
    fx(t+1) = L(t) + T(t) + S(t-s+1);
end

% Compute MAE + a penalty for parameters beyond the admitted range,
% i.e., 0 < alpha, beta, gamma < 1. The latter is required for 
% parameter estimation.
maxx = max(x);
MAE = mean(abs(x(2*s+1:end) - fx(2*s+1:end))) ...
    + maxx*(alpha<=0) + maxx*(alpha>=1) ...
    + maxx*(beta<=0) + maxx*(beta>=1) ...
    + maxx*(gamma<=0) + maxx*(gamma>=1);