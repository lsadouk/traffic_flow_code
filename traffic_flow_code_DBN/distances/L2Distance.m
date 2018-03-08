classdef L2Distance < handle & AbstractNet
    % L2Distance implements the L2 comparison between vectors as an
    % AbstractNet
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        inSize;
    end
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = L2Distance(inSize)
            % obj = L2DISTANCE(inSize) returns an instance of L2DISTANCE for
            % inputs of size inSize
            
            assert(isnumeric(inSize), 'Only numerical input is supported');
            obj.inSize = inSize;
        end
        
        % AbstractNet implementation----------------------------------------- %
     
        function S = insize(self)
            S = {self.inSize; self.inSize};
        end
        
        function S = outsize(~)
            S = 1;
        end
        
        function [Y, A] = compute(self, X)
            nSamples = size(X{1}, numel(self.inSize) + 1);
            X{1} = reshape(X{1}, [], nSamples);
            X{2} = reshape(X{2}, [], nSamples);
            Y    = sqrt(sum((X{1} - X{2}) .^2, 1));
            if nargout > 1
                A.D = X{1} - X{2};
                A.Y = Y;
            end
        end
        
        function [] = pretrain(~, ~)
        end
        
        function [G, inErr] = backprop(self, A, outErr)
            G     = [];
            D     = bsxfun(@times, A.D, outErr ./ A.Y);
            D     = reshape(D, [self.inSize numel(outErr)]);
            inErr = {D, -D};
        end
        
        function [] = gradientupdate(~, ~)
            % Nothing to do
        end
        
    end
end
