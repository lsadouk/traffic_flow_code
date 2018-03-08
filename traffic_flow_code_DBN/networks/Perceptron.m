classdef Perceptron < handle & AbstractNet
    % PERCEPTRON Single Layer Perceptron
    %   PERCEPTRON implements AbstractNet for single layer perceptrons of 
    %   neurons with bias. Neurons use sigmoïd activation.
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        W;         % connection weights
        b;         % bias
        trainOpts; % training options
    end
    
    methods
        
        % Constructor ------------------------------------------------------- %
        
        function obj = Perceptron(inSz, outSz, trainOpts)
            % PERCEPTRON Constructor for RELURBM
            %   net = PERCEPTRON(inSz, outSz, O) returns an instance
            %   PERCEPTRON with inSz input neurons fully connected to outSz
            %   output neurons. Training setting are stores in the
            %   structure O with the fields:
            %       lRate     -- learning rate
            %       dropout   -- input units dropout rate [optional]
            %       decayNorm -- type of weight decay penalty [optional]
            %       decayRate -- coeeficient on weight decay penalty
            %   By default, connection weights are initialized using a
            %   centered Gaussian distribution of variance 1/inSz.
            
            if isfield(trainOpts, 'decayNorm') ...
                    || isfield(trainOpts, 'decayRate')
                assert(isfield(trainOpts, 'decayNorm') ...
                    && isfield(trainOpts, 'decayRate'), ...
                    'specify both decay norm and rate');
            end
            obj.trainOpts = trainOpts;
            
            % Initializing weights
            obj.W = randn(inSz, outSz) / sqrt(inSz);
            obj.b = zeros(outSz, 1);
        end
        
        % AbstractNet implementation ---------------------------------------- %
        
        function S = insize(self)
            S = size(self.W, 1);
        end
        
        function S = outsize(self)
            S = size(self.W, 2);
        end
        
        function [Y, A] = compute(self, X)
            % training with dropout
            if nargout == 2 && isfield(self.trainOpts, 'dropout')
                A.mask  = rand(self.insize(), 1) > self.trainOpts.dropout;
                Wmasked = bsxfun(@times, self.W, ...
                    A.mask ./ (1 - self.trainOpts.dropout));
                % Save necessary values for gradient computation
                A.S = bsxfun(@plus, Wmasked' * X, self.b); % stimuli
                A.X = X;
                Y   = self.activation(A.S);
                A.Y = Y;
            elseif nargout == 2 % training
                % Save necessary values for gradient computation
                A.S = bsxfun(@plus, self.W' * X, self.b); % stimuli
                A.X = X;
                Y   = self.activation(A.S);
                A.Y = Y;
            else % normal
                Y = self.activation(bsxfun(@plus, self.W' * X, self.b));
            end
        end
        
        function [] = pretrain(~, ~)
            % Nothing to do
        end
        
        function [G, inErr] = backprop(self, A, outErr)            
            % Gradient computation of sigmoid
            delta  = outErr .* A.Y .* (1 - A.Y);
            % Gradient comp. of Relu
            %delta  = outErr .* double(A.Y >0);
            %delta  = outError .*
            G.dW   = A.X * delta';
            G.db   = sum(delta, 2);
            
            % Error backpropagation
            inErr = self.W * delta;
            
            % Dropout
            if isfield(self.trainOpts, 'dropout')
                G.dW  = bsxfun(@times, G.dW, A.mask) ...
                    * (1 - self.trainOpts.dropout);
                inErr = bsxfun(@times, inErr, A.mask) ...
                    * (1 - self.trainOpts.dropout);
            end
        end
        
        function [] = gradientupdate(self, G)
            opts = self.trainOpts;
            % Gradient update
            self.W = self.W - opts.lRate * G.dW;
            self.b = self.b - opts.lRate * G.db;
            
            % Weight decay
            if isfield(opts, 'decayNorm') && opts.decayNorm == 2
                self.W = self.W - opts.lRate * opts.decayRate * self.W;
                self.b = self.b - opts.lRate * opts.decayRate * self.b;
            elseif isfield(opts, 'decayNorm') && opts.decayNorm == 1
                self.W = self.W - opts.lRate * opts.decayRate * sign(self.W);
                self.b = self.b - opts.lRate * opts.decayRate * sign(self.b);
            end
        end
        
    end % methods
    
    methods(Static)
        
        function [Y] = activation(X)
            % Sigmoïd activation
            Y = 1 ./ (1 + exp(-X));
            % relu
            %Y = X .* double(X > 0);
         
        end
        
    end
    
end % classdef RBM
