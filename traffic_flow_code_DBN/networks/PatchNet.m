classdef PatchNet < handle & AbstractNet
    % PATCHNET Overlapping patch extractor
    %   Implementation of AbstractNet to extract overlapping patches from
    %   2D inputs
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        inSz;    % 2D input size
        patchSz; % patch size
        overlap; % # of overlapping pixel between patches
        nPatches % number of patches vertically and horizontally
    end % properties
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = PatchNet(inSz, patchSz, overlap)
            % obj = PATCHNET([ih iw], [ph pw], [ovt ohz]) returns an
            % instance of PATCHNET for inputs of size ih by iw which cuts
            % patches of size ovt by ohz with ovt (resp. ovz) pixels of
            % vertical (resp. horizontal) overlap.
            
            obj.inSz   = inSz;
            obj.patchSz = patchSz;
            obj.overlap = overlap;
            obj.nPatches = floor((inSz - patchSz) ./ (patchSz - overlap)) + 1;
        end
        
        % AbstractNet implementation ---------------------------------------- %
        
        function S = insize(self)
            S = self.inSz;
        end
        
        function S = outsize(self)
            S = cell(prod(self.nPatches), 1);
            S(:) = {self.patchSz};
        end
        
        function [Y, A] = compute(self, X)
            Y = cell(prod(self.nPatches), 1);
            for i = 1:self.nPatches(1)
                for j = 1:self.nPatches(2)
                    voff = (i-1) * (self.patchSz(1) - self.overlap(1)) + 1;
                    hoff = (j-1) * (self.patchSz(2) - self.overlap(2)) + 1;
                    Y{(j-1) * self.nPatches(1) + i} = ...
                        reshape(X(voff:voff + self.patchSz(1) - 1, ...
                                hoff:hoff + self.patchSz(2) - 1, :), ...
                                prod(self.patchSz), size(X, 3));
                end
            end
            A = [];
        end
        
        function [] = pretrain(~, ~)
        end
        
        function [G, inErr] = backprop(~, ~, ~)
            error('PatchNet backpropagation is not implemented');
        end
        
        function [] = gradientupdate(~, ~)
            % Nothing to do
        end
        
        function [] = train(~, ~, ~)
            % Nothing to do
        end
        
    end % methods
    
end % classdef RBM