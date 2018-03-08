classdef GBRBM < handle & RBM
    % GBRBM Restricted Boltzmann Machine Model object
    %   GBRBM implements AbstractNet for Restricted Boltzmann Machines with
    %   gaussian visible units and binary hidden units. Variance on input units
    %   is not learnt and assumed to be 1, so data should be normalized first.
    %
    %   Most of the configuration settings are inherited from RBM otherwise.

    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    methods
        function obj = GBRBM(nVis, nHid, pretrainOpts, trainOpts)
            % GBRBM Construct an instance of Restricted Boltzmann Machine
            %   Refer to RBM for the documentation
            obj@RBM(nVis, nHid, pretrainOpts, trainOpts)
        end
        
        function V = hid2vis(self, H, varargin)
            V = bsxfun(@plus, self.W * H, self.b);
        end
        
        function [dW, db, dc, hid0] = cd(self, X)
            opts = self.pretrainOpts;
            
            nObs = size(X, 2);
            vis0 = X;
            hid0 = self.vis2hid(vis0);
            
            % Dropout masks
            if opts.dropHid > 0
                hmask = rand(size(hid0)) < opts.dropHid;
            end
            if opts.dropVis > 0
                vmask = rand(size(X)) < opts.dropHid;
            end

            hid = hid0;
            for k = 1:opts.nGS
                if opts.sampleHid % sampling ?
                    hid = hid > rand(size(hid));
                end
                if opts.dropHid > 0 && k < opts.nGS % Dropout?
                    hid = hid .* hmask / (1 - opts.dropHid);
                end
                vis = self.hid2vis(hid);
                if opts.sampleVis && k < opts.nGS % sampling ?
                    vis = vis + randn(size(vis));
                end
                if opts.dropVis > 0 && k < opts.nGS  % Dropout?
                    vis = vis .* vmask / (1 - opts.dropVis);
                    % TODO keep non masked visibles for CD but mask for hid
                    % computation.
                end
                hid = self.vis2hid(vis);
            end
            
            dW      = - (vis0 * hid0' - vis * hid') / nObs;
            dc      = - (sum(hid0, 2) - sum(hid, 2)) / nObs;
            db      = - (sum(vis0, 2) - sum(vis, 2)) / nObs;
        end
        
    end
    
end
