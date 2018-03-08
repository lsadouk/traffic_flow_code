
function Y = euclideanloss(X, c, dzdy)
%EUCLIDEANLOSS Summary of this function goes here
%   Detailed explanation goes here

assert(numel(X) == numel(c));
c= reshape(c,1,1,1,[]);
%dx = size(X,4);
%dc= size(c,2);
assert(all(size(X) == size(c)));

mu = 0.8024; %for US-101-North pr=3
sigma= 0.1873;%for US-101-North pr=3
% mu = 0.8170; %mean(label) of the preprocessed data 'traffic_images_H101_North_D7_27days.mat'
% sigma =0.1864; %std(label) of the preprocessed data 'traffic_images_H101_North_D7_27days.mat'
lambda = 1;  
    
%X = reshape(X,1,[]);
if nargin == 2 || (nargin == 3 && isempty(dzdy))
    % ORGINAL VERSION 
    %Y = 1 / 2 * sum((X - c).^ 2); % Y is divided by d(4) in cnn_train.m 
    
    %WEIGHTED version4  Probability Density Function
    Y = 1 / 2 * sum( (X - c).^ 2 + lambda .* abs(X - c) .*(1 - pdf('Normal',c,mu,sigma) .* (sigma*sqrt(2*pi))) ); % .* 2-pdf
    
    %WEIGHTED version5  step weight function w
%     w = zeros(size(c));
%     w = (c < 0.5);          %if c >= 0.5, w =0;  else w=1;  end
%     Y = 1 / 2 * sum( (X - c).^ 2 + abs(X - c) .* w );
    %OR    (but remove the code above (c= reshape(c,1,1,1,[]);)    
    
    %Y = 0.5*sum((squeeze(X)'-c).^2); % squeeze(X) turn X (1,1,1,N) to (N,1) -> squeeze(X)' = (1,N)
    
elseif nargin == 3 && ~isempty(dzdy)
    
    assert(numel(dzdy) == 1);
    
    % ORIGINAL VERSION
    %Y = bsxfun(@times,dzdy ,(X - c));
    %WEIGHTED VERSION 1
    %Y = bsxfun(@times,dzdy ,(X - c) .* (1- normcdf(c+epsilon,mu,sigma) + normcdf(c-epsilon,mu,sigma))  );
    %WEIGHTED VERSION 2
    %Y = bsxfun(@times,dzdy ,2*(X - c) + (1- normcdf(c+epsilon,mu,sigma) + normcdf(c-epsilon,mu,sigma))  );
    %WEIGHTED VERSION 4-2
    %Y = bsxfun(@times,dzdy , 2.*(X - c) + (abs(X) ./ X) .*(1 - pdf('Normal',c,mu,sigma) .* (sigma*sqrt(2*pi)))  );
    Xmc = X-c;
    Xmc(Xmc < 0) = -1;
    Xmc(Xmc >= 0) = 1;

    Y = bsxfun(@times,dzdy , (X - c) + 0.5*lambda.* Xmc .* (1 - pdf('Normal',c,mu,sigma) .* (sigma*sqrt(2*pi)))  );
    
    %WEIGHTED version5  step weight function w
%     w = zeros(size(c));
%     w = (c < 0.5);          %if c >= 0.5, w =0;  else w=1;  end
%     %Y = bsxfun(@times,dzdy ,(X - c) + 1/2 .* (abs(X) ./ X) .*w);
%     if(X >= 0)
%         Y = bsxfun(@times,dzdy ,(X - c) + 1/2 .*w);
%     else  % X < 0
%         Y = bsxfun(@times,dzdy ,(X - c) - 1/2 .*w);
%     end

    %OR 
    %Y = dzdy * (X - c); % Y is divided by d(4) in cnn_train.m / cnn_train_mgpu.m.
    %OR    (but remove the code above (c= reshape(c,1,1,1,[]);)
    %Y = +((squeeze(X)'-c))*dzdy;
    %Y = reshape(Y,size(X));
    
end

end
