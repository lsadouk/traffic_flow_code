function trained = train(net, costFn, X, Y, opts, lambda)
% TRAIN trains a neural network
%	trained = train(net, costFn, X, Y, opts) returns a trained copy of net.
%   Supervized training performs gradient updates to minimize the cost
%   function costFn on the dataset. 
%   The cost function must implement the ErrorCost interface.
%   Sample inputs and associated outputs are given by X and Y respectively.
%   opts is a structure containing training settings:
%       nIter        -- # of training epochs
%       batchSz      -- # of samples by mini-batch [optional]
%       displayEvery -- frequency of progress display [optional]
%       batchFn      -- function that returns mini-batches from the dataset
%                       see simpleBatchFn for an example and the expected 
%                       interface [optional]
%
%   train(net, costFn, X, Y, opts) takes net as a handle (reference) and
%   trains in-place
%
%   Note: This function is provided as a helper, it has not been tested for
%   all possible use-cases. Adapt it to your needs.

% author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
% licence : MIT

assert(isa(net, 'AbstractNet'), ...
    'net should implement AbstractNet');
assert(isnumeric(Y) || islogical(Y), 'Only numeric output is supported');

if ~isfield(opts, 'batchFn') && isfield(opts, 'batchSz')
    opts.batchFn = @simpleBatchFn;
end

if nargin == 1
    trained = net.copy();
else
    trained = net;
end

for i = 1:opts.nIter
    
    if isfield(opts, 'batchFn') % train over mini-batches
        idx = [];
        moreBatches = true; % set to any value to get inside the loop
        while moreBatches
            [batchX, batchY, idx] = opts.batchFn(X, Y, opts.batchSz, idx);
            [O, A] = trained.compute(batchX);     % forward pass
            moreBatches = ~isempty(idx);
            clear batchX;                         % release memory
            outGrad = costFn.gradient(O, batchY,lambda); % error derivative
            G = trained.backprop(A, outGrad);     % backpropagation
            trained.gradientupdate(G);            % update
        end
        
    else % train over the whole dataset
        
        [O, A] = trained.compute(X);      % forward pass
        outGrad = costFn.gradient(O, Y,lambda);  % error derivative
        G = trained.backprop(A, outGrad); % error derivative
        trained.gradientupdate(G);        % update
    end
    
    % Report
    if isfield(opts, 'displayEvery') && mod(i, opts.displayEvery) == 0
		if isfield(opts, 'batchFn')
			E           = [];
			S           = [];
			moreBatches = true;
			idx         = [];
			while moreBatches
				[batchX, batchY, idx] = ...
					opts.batchFn(X, Y, opts.batchSz, idx);
				nSamples    = size(batchY, ndims(batchY));
				O           = trained.compute(batchX);
				E           = [E costFn.compute(O, batchY)];
				S           = [S nSamples];
				moreBatches = ~isempty(idx);
			end
			MC = sum(E .* S ./ sum(S));
		else
			O  = trained.compute(X);
			MC = costFn.compute(O, Y);
		end
		fprintf('%03d , Error cost : %1.3e\n', i, MC);
    end
end
end
