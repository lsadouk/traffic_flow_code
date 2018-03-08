classdef L1Distance < handle & AbstractNet
    % L1Distance implements the L1 comparison between vectors as an
    % AbstractNet
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        inSize;
    end
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = L1Distance(inSize)
            % obj = L1DISTANCE(inSize) returns an instance of L1DISTANCE for
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
            Y    = sum(abs(X{1} - X{2}), 1);
            if nargout > 1
                A.D = sign(X{1} - X{2});
            end
        end
        
        function [] = pretrain(~, ~)
        end
        
        function [G, inErr] = backprop(self, A, outErr)
            G     = [];
            D     = bsxfun(@times, A.D, outErr);
            D     = reshape(D, [self.inSize numel(outErr)]);
            inErr = {D, -D};
        end
        
        function [] = gradientupdate(~, ~)
            % Nothing to do
        end
        
    end
end
