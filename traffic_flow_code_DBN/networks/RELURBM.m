classdef RELURBM < handle & AbstractNet
    % RELURBM Restricted Boltzmann Machine model with ReLU activation units
    %   RELURBM implements AbstractNet for Restricted Boltzmann Machines
    %   with rectified linear units on both visible and hidden units.
    %   Unsupervized training relies on Hinton's Contrastive Divergence
    %   (CD1) while supervized training uses simple gradient update with
    %   backpropagation.
    %
    %   Pretraining regularization includes L2 and L1 weight decay and dropout.
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        nVis;                    % # of visible units
        nHid;                    % # of hidden units
        W;                       % connection weights
        b;                       % visible units bias
        c;                       % hidden units bias
        pretrainOpts = struct(); % pretraining settings
        trainOpts    = struct(); % supervized training settings
        
        hasHidBias = true;
    end
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = RELURBM(nVis, nHid, pretrainOpts, trainOpts, ...
                varargin)
            % RELURBM Constructor for RELURBM
            %   rbm = RELURBM(nVis, nHid, pretrainOpts, trainOpts)
            %   returns an rbm with nVis visible and nHid output hidden
            %   units.
            %   pretrainOpts is a structure with pretraining setting:
            %       lRate     -- learning rate
            %       dropVis   -- visible units dropout rate [optional]
            %       dropout   -- hidden units dropout rate [optional]
            %       momentum  -- gradient momentum [optional]
            %       wPenalty  -- L2 weight penalty coefficient [optional]
            %       gradThres -- maximum norm of the gradient [optional]
            %
            %   Similarly, trainOpts has configuration for supervized
            %   training:
            %       lRate     -- learning rate
            %       dropout   -- input units dropout rate [optional]
            %
            %   rbm = RELURBM(nVis, nHid, pretrainOpts, trainOpts, 'noBias')
            %   creates an RBM without bias on hidden units.
            
            obj.nVis = nVis;
            obj.nHid = nHid;
            
            if ~isfield(pretrainOpts, 'dropVis')
                pretrainOpts.dropVis = 0;
            end
            if ~isfield(pretrainOpts, 'dropout')
                pretrainOpts.dropout = 0;
            end
            if ~isfield(pretrainOpts, 'momentum')
                pretrainOpts.momentum = 0;
            end
            obj.pretrainOpts = pretrainOpts;
            
            if ~isfield(trainOpts, 'decayNorm')
                trainOpts.decayNorm = -1;
            end
            obj.trainOpts = trainOpts;
            
            % Initializing weights
            obj.W = 5 * randn(nVis, nHid) / sqrt(nVis * nHid);
            obj.b = 5 * ones(nVis, 1) / nVis;
            obj.c = 5 * ones(nHid, 1) / nHid;
            
            if ~isempty(varargin) && strcmp(varargin{1}, 'noBias')
                obj.hasHidBias = false;
                obj.c = 0;
            end
        end
        
        % AbstractNet implementation ---------------------------------------- %
        
        function S = insize(self)
            S = self.nVis;
        end
        
        function S = outsize(self)
            S = self.nHid;
        end
        
        function [Y, A] = compute(self, X)
            if nargout == 2 && isfield(self.trainOpts, 'dropout')
                A.mask  = rand(self.nVis, 1) > self.trainOpts.dropout;
                Wmasked = bsxfun(@times, self.W, ...
                    A.mask ./ (1 - self.trainOpts.dropout));
                Y       = max(0, bsxfun(@plus, Wmasked' * X, self.c));
            else
                Y = max(0, bsxfun(@plus, self.W' * X, self.c));
            end
            if nargout > 1
                A.x  = X;
                A.ds = Y > 0;
            end
        end
        
        function [] = pretrain(self, X)
            nObs  = size(X, 2);
            opts  = self.pretrainOpts;
            dWold = zeros(size(self.W));
            dbold = zeros(size(self.b));
            dcold = zeros(size(self.c));
            
            ndrop = 0;
            
            for e = 1:opts.nEpochs
                shuffle  = randperm(nObs);
                
                % Batch loop
                for batchBeg = 1:opts.batchSz:nObs
                    bind = shuffle(batchBeg : min(nObs, ...
                        batchBeg + opts.batchSz -1));
                    
                    % Gibbs sampling
                    [dW, db, dc] = self.cd(X(:, bind));
                    
                    % Weight decay
                    if isfield(opts, 'wPenalty')
                        dW = dW + opts.wPenalty * self.W;
                        db = db + opts.wPenalty * self.b;
                        dc = dc + opts.wPenalty * self.c;
                    end
                    
                    % Cap gradient value
                    if isfield(opts, 'gradThres')
                        gradNorm = max([norm(dW) / sqrt(numel(dW)), ...
                            norm(db) / numel(db), ...
                            norm(dc) / numel(dc)]) ...
                            * opts.lRate;
                        if gradNorm > opts.gradThres
                            fprintf(1, sprintf(...
                                'capped gradient from %d\n', gradNorm));
                            dW = dW * opts.gradThres / gradNorm;
                            db = db * opts.gradThres / gradNorm;
                            dc = dc * opts.gradThres / gradNorm;
                        end
                    end
                    
                    % Momentum
                    dW = dWold * opts.momentum + (1 - opts.momentum) * dW;
                    db = dbold * opts.momentum + (1 - opts.momentum) * db;
                    dc = dcold * opts.momentum + (1 - opts.momentum) * dc;
                    
                    % Apply gradient
                    self.W = self.W - opts.lRate * dW;
                    self.b = self.b - opts.lRate * db;
                    if self.hasHidBias
                        self.c = self.c - opts.lRate * dc;
                    end
                    
                    % Save gradient
                    dWold = dW;
                    dbold = db;
                    dcold = dc;
                end
                
                % Feature selection
                if isfield(opts, 'selectivity') ...
                        && mod(e, opts.selectivity) == 0  ...
                        && e > opts.selectAfter ...
                        && e < opts.nEpochs - opts.selectAfter
                    Y = self.compute(X);
                    drop = self.selectuseless(Y);
                    self.W(:, drop) = ...
                        randn(self.nVis, sum(drop)) / self.nVis;
                    self.c(drop) = ...
                        abs(randn(sum(drop), 1) / self.nVis);
                    ndrop = ndrop + sum(drop);
                    if sum(drop) > 0
                        fprintf('(dropped %d useless)\n', sum(drop));
                    end
                    drop = self.selectredundant(Y, 2 * self.nHid);
                    self.W(:, drop) = ...
                        randn(self.nVis, numel(drop)) / self.nVis;
                    self.c(drop) = ...
                        abs(randn(numel(drop), 1) / self.nVis);
                    ndrop = ndrop + numel(drop);
                    if numel(drop) > 0
                        fprintf('(dropped %d redundants)\n', numel(drop));
                    end
                end
                
                % Report
                if isfield(opts, 'displayEvery') ...
                        && mod(e, opts.displayEvery) == 0
                    % Reconstruct input samples
                    R = self.compute(X);
                    R = max(0, bsxfun(@plus, self.W * R, self.b));
                    % Mean square reconstruction error
                    rmse = sqrt(mean(sum((R - X) .^2)));
                    fprintf('%03d , rmse = %g, dropped = %d\n', ...
                        e, rmse, ndrop);
                    ndrop = 0;
                end
            end
        end
        
        function [G, inErr] = backprop(self, A, outErr)
            % Gradient computation
            delta  = outErr .* A.ds;
            G.dW   = A.x * delta';
            G.dc   = sum(delta, 2);
            
            % Error backpropagation
            inErr = self.W * delta;
            
            % Dropout
            if isfield(self.trainOpts, 'dropout')
                G.dW  = bsxfun(@times, G.dW, A.mask);
                inErr = bsxfun(@times, inErr, A.mask);
            end
        end
        
        function [] = gradientupdate(self, G)
            opts = self.trainOpts;
            % Gradient update
            self.W = self.W - opts.lRate * G.dW;
            if self.hasHidBias
                self.c = self.c - opts.lRate * G.dc;
            end
            
            % Weight decay
            if opts.decayNorm == 2
                self.W = self.W - opts.lRate * opts.decayRate * self.W;
                self.c = self.c - opts.lRate * opts.decayRate * self.c;
            elseif opts.decayNorm == 1
                self.W = self.W - opts.lRate * opts.decayRate * sign(self.W);
                self.c = self.c - opts.lRate * opts.decayRate * sign(self.c);
            end
        end
        
        % Methods ----------------------------------------------------------- %
        
        function [dW, db, dc] = cd(self, X)
            % CD Contrastive divergence (Hinton's CD(k))
            %   [dW, db, dc, act] = cd(self, X) returns the gradients of
            %   the weihgts, visible and hidden biases using Hinton's
            %   approximated CD. The sum of the average hidden units
            %   activity is returned in act as well.
            opts = self.pretrainOpts;
            
            nObs = size(X, 2);
            
            % Forward pass
            vis0 = X;
            
            if opts.dropVis > 0 % Visible units dropout
                mask = rand(self.nVis, 1) > opts.dropVis;
                X = bsxfun(@times, X, mask) / (1 - opts.dropVis);
            end
            
            act  = bsxfun(@plus, self.W' * X, self.c);
            hid0 = max(0, act);
            
            % Gibbs sampling
            hid  = max(0, act + randn(self.nHid, nObs) ...
                .* sqrt(1./(1+exp(-act))));
            if opts.dropout > 0
                mask = rand(self.nHid, 1) > opts.dropout;
                hid = bsxfun(@times, hid, mask) / (1 - opts.dropout);
            end
            act  = bsxfun(@plus, self.W * hid, self.b);
            vis  = max(0, act);
            hid  = max(0, bsxfun(@plus, self.W' * vis, self.c));
            
            % Contrastive divergence
            dW = vis * hid' - vis0 * hid0';
            if self.hasHidBias
                dc = sum(hid, 2) - sum(hid0, 2);
            else
                dc = zeros(self.nHid, 1);
            end
            db = sum(vis, 2) - sum(vis0, 2);
        end % cd(self, X)
        
    end % methods
    
    methods(Static)
        
        % Empirical selection of neurons with low significance
        function drop = selectuseless(Y)
            s = std(Y, 0, 2);
            drop = s < sqrt(1/size(Y, 2));
        end
        
        % Empirical selection of redundant neurons
        function drop = selectredundant(Y, n)
            % drop = selectredundant(Y, n) returns a binary vectors of
            % neurons which seem redundant (similar outputs). Comparison
            % are run on n random pairs.
            
            % original idea from
            % http://stackoverflow.com/questions/15793172/
            % efficiently-generating-unique-pairs-of-integers
            % #answer-15795308
            h = size(Y, 1);
            r = randperm(h/2*(h-1), n);
            q = floor(sqrt(8*(r-1) + 1)/2 + 1/2);
            p = r - q.*(q-1)/2;
            s = (p ~= q);
            p = p(s);
            q = q(s);
            d = sum(Y(q,:) .* Y(p,:), 2) ...
                ./ sqrt(sum(Y(q,:).^2, 2) .* sum(Y(p,:).^2, 2));
            drop = q(abs(d) > 0.95);
        end
    end
    
end % classdef RBM
