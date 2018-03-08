function [batchX, batchY, idx] = simpleBatchFn(X, Y, batchSz, idx)
% Generates batches from a dataset
%   [batchX, batchY, I1] = SIMPLEBATCHFN(X, Y, N, []) Generates on batch of
%   N samples out of the array based dataset X and Y (assuming samples are
%   concatenated alongside the last dimension for each). I1 should be
%   passed to SIMPLEBATCHFN in order to generate the next batch. If less
%   than N samples are availables, the batch will contain all remaining
%   samples.
%
%   [batchX, batchY, I2] = simpleBatchFn(X, Y, N, I1) Generates a batch of
%   samples which avoids samples that led to I1 (I2 becomes I1 for the next
%   iteration).
%
%   Example:
%       N = 100;
%       [batchX, batchY, I] = opts.batchFn(X, Y, N, []);
%       while ~isempty(batchX)
%           % do somthing with batchX and batchY
%           [batchX, batchY, I] = opts.batchFn(X, Y, N, I);
%       end

    nS  = size(X, ndims(X)); % number of samples N
    szX = size(X); % 28*28(784) by  N
    szX = szX(1:end-1); %784 (dimension of each input image) = size of image
    szY = size(Y); % 31(labels) by N
    szY = szY(1:end-1); % 31 (dimension of each label)
    X   = reshape(X, [], nS);
    Y   = reshape(Y, [], nS);
    
    if isempty(idx) % first mini-batch
        idx = randperm(nS);
    end
    batchX = reshape(X(:, idx(1:min(batchSz, end))), szX, []);
    batchY = reshape(Y(:, idx(1:min(batchSz, end))), szY, []);
    
    if numel(idx) > batchSz
        idx = idx(batchSz:end);
    else
        idx = [];
    end
    
    
%     % perform data augmentation through jittering (rotation and scaling)
%     N = size(batchX, 2);
%     for i = 1:N % for each image in batch
%         %1. randomly rotate the training images for jittering
%         roll = rand();
%         if roll>0.5
%             image =  reshape(batchX(:,i), sqrt(szX), sqrt(szX)); % reshape the i image into 28*28
%             %imshow(image);
%             %imshow(rotating(image));
%             batchX(:,i) = reshape(rotating(image), szX, []); 
%         end
% 
%         % randomly scale the training images for jittering
%         roll = rand();
%         if roll>0.5
%             image =  reshape(batchX(:,i), sqrt(szX), sqrt(szX)); % reshape the i image into 28*28
%             %imshow(image);
%             %imshow(scaling2(image));
%             batchX(:,i) = reshape(scaling2(image), szX, []); 
%         end
%     end
    
end