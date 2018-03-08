% Common Interface for Neural Networks in nnbox

% author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
% licence : Public Domain

classdef AbstractNet < handle & matlab.mixin.Copyable
    % Interface for neural network models with gradient based parameter
    % optimization.
    
    methods (Abstract)
        
        % INSIZE Input size
        %   S = INSIZE(obj) returns the expected size of the input similarily 
        %   to size() function. If the network supports distinct groups of 
        %   inputs (eg: expert model), all size vectors are stored in a cell 
        %   array.
        S = insize(self)
        
        % OUTSIZE Output size
        %   S = OUTSIZE(obj) returns the expected size of the output similarily 
        %   to size() function. If the network supports distinct groups of 
        %   outputs (eg: expert model), the sizes of all groups are joined
        %   into a cell array.        
        S = outsize(self)

        % COMPUTE Compute network's output
        %   Y = COMPUTE(obj, X) returns the output of the network for a given 
        %   input X with the proper size as returned by insize(). X (and 
        %   consequently Y) may have one extra last dimension in order to pass 
        %   several inputs at once.
        %
        %   [Y, A] = COMPUTE(obj, X) also returns forward pass information in A 
        %   for further use in a backpropagation update (A is implementation 
        %   specific).
        Y = compute(self, X)
        
        % PRETRAIN Unsupervised training
        %   PRETRAIN(obj, X, opts) Perform unsupervized training on input 
        %   data X.
        pretrain(self, X)

        % BACKPROP Compute gradient update and back-propagated error
        %   [G, inErr] = BACKPROP(A, outErr) compute gradient and 
        %   backpropagated error signal given the error signal for this
        %   network's output and forward-pass information. This function 
        %   shall not alter the network (no side effect), see
        %   gradientupdate for parameters optimization.
        %
        %   A      -- forward pass data as returned by compute
        %   outErr -- error cost derivative w.r.t. the output of compute 
        %             (not deltas in usual notations).
        %
        %   G      -- parameters gradient (implementation specific)
        %   inErr  -- error cost derivative w.r.t. neurons input (not deltas).
        [G, inErr] = backprop(self, A, outErr)
        
        % GRADIENTUPDATE update network's parameters according to gradient 
        %   correction returned by BACKPROP.
        gradientupdate(self, G)
        
    end % methods (Abstract)
    
end

