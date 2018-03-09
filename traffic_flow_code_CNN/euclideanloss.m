
function Y = euclideanloss(X, c, lambda, dzdy)
%EUCLIDEANLOSS Summary of this function goes here
%   Detailed explanation goes here

assert(numel(X) == numel(c));
c= reshape(c,1,1,1,[]);
assert(all(size(X) == size(c)));

mu = 0.8170; %mean(label) of the preprocessed data 'traffic_images_H101_North_D7_27days.mat'
sigma =0.1864; %std(label) of the preprocessed data 'traffic_images_H101_North_D7_27days.mat'
    
if nargin == 3 || (nargin == 4 && isempty(dzdy))
    % ORGINAL VERSION 
    %Y = 1 / 2 * sum((X - c).^ 2); % Y is divided by d(4) in cnn_train.m 
    
    %WEIGHTED version  Probability Density Function
    Y = 1 / 2 * sum( (X - c).^ 2 + lambda .* abs(X - c) .*(1 - pdf('Normal',c,mu,sigma) .* (sigma*sqrt(2*pi))) ); % .* 2-pdf
        
elseif nargin == 4 && ~isempty(dzdy)
    
    assert(numel(dzdy) == 1);
    
    % ORIGINAL VERSION
    %Y = bsxfun(@times,dzdy ,(X - c));

    Xmc = X-c;
    Xmc(Xmc < 0) = -1;
    Xmc(Xmc >= 0) = 1;

    Y = bsxfun(@times,dzdy , (X - c) + 0.5*lambda.* Xmc .* (1 - pdf('Normal',c,mu,sigma) .* (sigma*sqrt(2*pi)))  );
     
end

end
