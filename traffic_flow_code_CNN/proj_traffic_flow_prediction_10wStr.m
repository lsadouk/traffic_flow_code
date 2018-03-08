function [net, info] = proj_traffic_flow_prediction_10wStr()
%code for Computer Vision, Georgia Tech by James Hays
%based off the MNIST and CIFAR examples from MatConvNet
run matconvnet-1.0-beta16/matlab/vl_setupnn ;
%run('matconvnet-1.0-beta16', 'matlab', 'vl_setupnn.m') ;

%It might actually be problematic to run vl_setup, because VLFeat has a
%version of vl_argparse that conflicts with the matconvnet version. You
%shouldn't need VLFeat for this project.
% run(fullfile('vlfeat-0.9.20', 'toolbox', 'vl_setup.m'));

next_pred_pt = input('Please enter next prediction speed point 1(5min)/2(10min)/...): ','s');
%opts.expDir is where trained networks and plots are saved.
opts.expDir = fullfile('result_data','data_D07_US101N10d_20by10_minP_R_1chan_WeigLossL1N_convs_', next_pred_pt,'pts') ;

% opts.learningRate is a critical parameter that can dramatically affect
% whether training succeeds or fails. For most of the experiments in this
% project the default learning rate is safe.
opts.learningRate = 0.01;% 0.01 for the first 20 epochs
% and 0.001 after 10 epochs


% An example of learning rate decay as an alternative to the fixed learning
% rate used by default. This isn't necessary but can lead to better
% performance.
% opts.learningRate = logspace(-4, -5.5, 300) ;
% opts.numEpochs = numel(opts.learningRate) ;

%opts.continue controls whether to resume training from the furthest
%trained network found in opts.batchSize. If you want to modify something
%mid training (e.g. learning rate) this can be useful. You might also want
%to resume a network that hit the maximum number of epochs if you think
%further training can improve accuracy.
opts.continue = true;

%GPU support is off by default.
% opts.gpus = [] ;

%select prediction task: classification (7 labels) or regression (1 label)
opts.prediction_type =input('Please select the prediction type: (c)classification / (r)regression  ','s');


%kfold = 0; % kfold=0 for TESTING and kfold = 4 for TRAINING (3 for training & 1 for testing)
kfold =input('Please enter the k-fold (k-1 for training & 1 for testing)_(0 for testing):  ');

nb_days =input('Please select the number of days (15, 21, 27, 30 or 59):  ','s');

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

% IF TRAINING, then specify the network architecture w/ cnn_init function
% ELSE (TESTING), provide the trained/preinitilized network NET
if(kfold ==0) % testing
    opts.numEpochs =  1;
    net = cnn_init_our_model(next_pred_pt);
else %training
    opts.numEpochs =  25; % 15 was 25
    if(isequal(opts.prediction_type,'r'))
        net = cnn_init_regression_convs_10wStr();  % cnn_init_regression() % cnn_init_regression_convs()
    else %if(isequal(opts.prediction_type,'c'))
        net = cnn_init_classification(opts);
    end
end
% The setup_data function loads the training and testing images into
% MatConvNet's imdb structure. You will be modifying the function.

% The commented out code can cache the image database so it isn't rebuilt
% with each run. I found it fast enough to rebuild and less likely to cause
% errors when you change the way images are preprocessed.
filename = strcat('traffic_images_H101_North_D7_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
%filename = strcat('traffic_images_I210_West_D7_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
%filename = strcat('traffic_images_I5_North_D11_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
%filename = strcat('traffic_images_65to70mph_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
%filename = strcat('traffic_images_60to70mph_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
%filename = strcat('traffic_images_I5_North_D7_',nb_days, 'days_',next_pred_pt, 'Pt'); 
%filename = strcat('traffic_images_I5_North_D11_',nb_days, 'days_',%next_pred_pt, 'Pt_16to19'); 
%filename = strcat('traffic_images_I5_North_D11_',nb_days, 'days_', next_pred_pt, 'Pt_L35_T17h_10wStr'); %'traffic_images_H101_North_D7_'
%filename = strcat('traffic_images_I210_West_D7_',nb_days, 'days_', next_pred_pt, 'Pt_L35_T08h40_10wStr'); %'traffic_images_H101_North_D7_'
%filename = strcat('traffic_images_SR1_North_',nb_days, 'days_', next_pred_pt, 'Pt_L329.55_T15h05_10wStr'); %'traffic_images_H101_North_D7_'
%traffic_images_SR1_North_1days_1Pt_L329.55_T15h05_10wStr
imdb_filename = strcat('imdb_',filename,'_', opts.prediction_type,'.mat'); %imdb_classification_D07-US101N.mat
if exist(imdb_filename, 'file')
    load(imdb_filename) ; %  imdb = load(imdb_filename) ;
else
  imdb = setup_data(opts, kfold, filename,next_pred_pt);
  %save(imdb_filename, '-struct', 'imdb') ;
end

%% compute batch size
%opts.batchSize iss the number of training images in each batch.
% if(kfold == 0) % use all testing dataset in one batch
%     opts.batchSize = length(imdb.images.labels) ; %150
% else
     opts.batchSize = 150 ; %150 for training %10000 for testing
% end

%% instead of 3 input channels (w. all 3 states: speed/occupancy/flow), 1 input channel (speed) 
% imdb.images.data = imdb.images.data(:,:,2,:); %1:occ / 2:speed / 3.flow in the old code
% imdb.images.data = imdb.images.data(:,:,2,:);%speed in new code
%% -------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
if(isequal(opts.prediction_type,'r'))
    opts.errorFunction = 'euclideanloss';
    [net, info] = cnn_train_r(net, imdb, @getBatch, opts, ...
    'val', find(imdb.images.set == 2)) ;
else % classification
    [net, info] = cnn_train_c(net, imdb, @getBatch, opts, ...
    'val', find(imdb.images.set == 2)) ;
end
[minm min_ind] =min(info.val.error(1,:));
fprintf('Lowest validation error is %f %d\n',minm*70, min_ind)
end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
%getBatch is called by cnn_train.

%'imdb' is the image database.
%'batch' is the indices of the images chosen for this batch.

%'im' is the height x width x channels x num_images stack of images. If
%opts.batchSize is 50 and image size is 64x64 and grayscale, im will be
%64x64x1x50.
%'labels' indicates the ground truth category of each image.

%This function is where you should 'jitter' data.
% --------------------------------------------------------------------
N = length(batch);
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% % Add jittering here before returning im
% for i = 1:N
% % using indexing to flip or mirror the selected values
% % roll = rand();
% % if roll>0.5
% %     im(:,:,:,i) = im(:,end:-1:1,:,i); % or im(:,:,:,ind) =  fliplr(im(:,:,:,ind));
% % end
% 
% %randomly rotate the training images for jittering
% roll = rand();
% if roll>0.5
%     im(:,:,:,i) = rotating(im(:,:,:,i)); 
% end
% 
% % randomly scale the training images for jittering
% roll = rand();
% if roll>0.5
%     im(:,:,:,i) = scaling2(im(:,:,:,i));
%     %imshow(im(:,:,:,i));
% end
% end

end


