function net1 = cnn_init_our_model(next_pred_point, lambda, test_freeway)

%% for w=10time points
net = load(strcat('result_data/data_',test_freeway,'_10by10_minP_R_1chan_WeigLossL',lambda,'N_convs_', next_pred_point,'pts/net-epoch-20.mat')) ;

%%layers
net.net.layers
net1 = net.net;
vl_simplenn_display(net1, 'inputSize', [10 10 1 50])

%We'll need to make some modifications to this network. First, the network
%accepts 

%This network is missing the dropout layers (because they're not needed at
%test time). It may be a good idea to reinsert dropout layers between the
%fully connected layers.

% [copied from the project webpage]
% proj6_part2_cnn_init.m will start with net = load('imagenet-vgg-f.mat');
% and then edit the network rather than specifying the structure from
% scratch.

% You need to make the following edits to the network: The final two
% layers, fc8 and the softmax layer, should be removed and specified again
% using the same syntax seen in Part 1. The original fc8 had an input data
% depth of 4096 and an output data depth of 1000 (for 1000 ImageNet
% categories). We need the output depth to be 250, instead. The weights can
% be randomly initialized just like in Part 1.


%f=1/100; 
% net.layers = net.layers(1:end-2);
% net.layers{end+1} = net.layers{end};
% net.layers{end-1} = net.layers{end-2};
% net.layers{end-2} = struct('type','dropout','rate',0.5); % add only one dropout layer between fc7 and fc8.
% net.layers{end+1} = struct('type','dropout','rate',0.5); % add only one dropout layer between fc7 and fc8.
% net.layers{end + 1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(1,1,4096,250, 'single'), zeros(1, 250, 'single')}}, ...
%                            'size', [1 1 4096 250], ...
%                            'stride', [1 1], ...
%                            'pad', [0 0 0 0], ...
%                            'name', 'fc8') ;
% net.layers{end+1} = struct('type', 'softmaxloss', ...
%                            'stride', 1, ...
%                            'pad', 0) ;  


% The dropout layers used to train VGG-F are missing from the pretrained
% model (probably because they're not used at test time). It's probably a
% good idea to add one or both of them back in between fc6 and fc7 and
% between fc7 and fc8.

% vl_simplenn_display(net, 'inputSize', [224 224 3 50])
%disp(size(net.layers));

