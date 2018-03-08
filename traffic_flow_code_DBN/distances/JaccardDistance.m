classdef JaccardDistance < handle & AbstractNet
    % L2Distance implements the Jaccard comparison between vectors as an
    % AbstractNet
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        inSize;
        offset;
    end
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = JaccardDistance(inSize, varargin)
            % obj = JACCARDDISTANCE(S) returns an instance of JACCARDDISTANCE
            % for input vectors of size S.
            % obj = JACCARDDISTANCE(S, R) adds a regularization factor in
            % the computation of the distance (see implementation details).
            
            obj.inSize = inSize;
            obj.offset = 0;
            if ~isempty(varargin) && isnumeric(varargin{1})
                obj.offset = varargin{1};
            end
        end
        
        % AbstractNet implementation ---------------------------------------- %
        
        function S = insize(self)
            S = {self.inSize; self.inSize};
        end
        
        function S = outsize(~)
            S = 1;
        end
        
        function [Y, A] = compute(self, X)
            if nargout > 1
                m    = sum(min(X{1}, X{2}) + self.offset, 1);
                M    = sum(max(X{1}, X{2}) + self.offset, 1);
                Y    = m ./ M;
                t1   = 1 ./ M;
                t2   = - M .^ -2;
                t3   = X{1} < X{2};
                A    = {- bsxfun(@times, t3, t1) - bsxfun(@times, ~t3, t2), ...
                        - bsxfun(@times, ~t3, t1) - bsxfun(@times, t3, t2) };
                Y = 1 - Y;
            else
                Y = 1 - sum(min(X{1}, X{2}) + self.offset, 1) ...
                     ./ sum(max(X{1}, X{2}) + self.offset, 1);
            end
        end
        
        function [] = pretrain(~, ~)
        end
        
        function [G, inErr] = backprop(~, A, outErr)
            G = [];
            inErr = { bsxfun(@times, A{1}, outErr), ...
                      bsxfun(@times, A{2}, outErr) };
        end
        
        function [] = gradientupdate(~, ~)
            % Nothing to do
        end
        
    end
end
