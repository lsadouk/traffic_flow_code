classdef SquareCost < ErrorCost
    % SQUARECOST Square error cost function
    % SQUARECOST implements ErrorCost for the square error cost function.
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    methods
        
        function C = compute(~, O, Y)
            assert(isnumeric(Y), 'Only numeric O and Y are supported');
            
            nSamples = size(O, ndims(O));
            C        = mean(sum(reshape((O - Y) .^ 2, [], nSamples), 1));
        end
        
        function C = computeEach(~, O, Y)
            assert(isnumeric(Y), 'Only numeric O and Y are supported');
            C = sum(reshape((O - Y) .^ 2, [], nSamples), 1);
        end
        
         function C = gradient(~, O, Y,lambda)
           assert(isnumeric(Y), 'Only numeric O and Y are supported');             
           if(lambda==0)
                 C = (O - Y);
           else %lambda=1
            mu = 0.8170; %mean(label) of the preprocessed data 'traffic_images_H101_North_D7_27days.mat'
            sigma =0.1864; %std(label) of the preprocessed data 'traffic_images_H101_North_D7_27days.mat'
            lambda = 1;  
                Xmc = O-Y;
            Xmc(Xmc <= 0) = -1;
            Xmc(Xmc > 0) = 1;

            C = (O - Y)+ 0.5*lambda.* Xmc .* (1 - pdf('Normal',Y,mu,sigma) .* (sigma*sqrt(2*pi)));
           end
%             
        end
    end
    
end