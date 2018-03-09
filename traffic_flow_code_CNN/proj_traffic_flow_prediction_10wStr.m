function [net, info] = proj_traffic_flow_prediction_10wStr()

run matconvnet-1.0-beta16/matlab/vl_setupnn ;
%run('matconvnet-1.0-beta16', 'matlab', 'vl_setupnn.m') ;

% we don't need VLFeat for this project. % run(fullfile('vlfeat-0.9.20', 'toolbox', 'vl_setup.m'));

next_pred_pt = input('Please forecasting for which you wish to predict speed (1)for 5-min, (2)for 10-min, (3)...): ','s');
opts.learningRate = 0.01;% 0.01 for the first 20 epochs
opts.continue = false;

%GPU support is off by default.
% opts.gpus = [] ;

%select prediction task: classification (7 labels) or regression (1 label)
opts.prediction_type =input('Please select the prediction type: (c)classification / (r)regression  ','s');

% lambda=0 for L2 loss and lambda=1 for probabilistic loss function
opts.lambda =input('Please enter the loss (0)L2 loss, (1)P loss:  ');

freeway = input('Please select the freeway used for training: H101_North_D7 / I5_North_D7 / I5_South_D7 / I5_North_D11 / I450_North_D7 / I210_West_D7 ','s');

% kfold=0 for TESTING and kfold = 4 for TRAINING (3 for training & 1 for testing)
kfold =input('Please enter the k-fold (k-1 for training & 1 for testing)_(0 for testing):  ');
if kfold == 0
    test_freeway =  input('Please select the testing freeway: H101_North_D7 / I5_North_D7 / I5_South_D7 / I5_North_D11 / I450_North_D7 / I210_West_D7: ','s');
    %opts.expDir is where trained networks and plots are saved.
    opts.expDir = fullfile(strcat('result_data/data_',test_freeway,'_10by10_minP_R_1chan_WeigLossL',int2str(opts.lambda),'N_convs_', next_pred_pt,'pts')) ;
else % training
    opts.expDir = fullfile(strcat('result_data/data_',freeway,'_10by10_minP_R_1chan_WeigLossL',int2str(opts.lambda),'N_convs_', next_pred_pt,'pts')) ;
end
% select 30 days
nb_days =input('Please select the number of days (15, 21, 27, 30 or 59):  ','s');


% --------------------------------------------------------------------
%   Prepare data
% --------------------------------------------------------------------

% IF TRAINING, then specify the network architecture w/ cnn_init function
% ELSE (TESTING), provide the trained/preinitilized network NET
if(kfold ==0) % testing
    opts.numEpochs =  1;
    net = cnn_init_our_model(next_pred_pt, int2str(opts.lambda), freeway);
else %training
    opts.numEpochs =  25; % 15 was 25
    if(isequal(opts.prediction_type,'r'))
        net = cnn_init_regression_convs_10wStr();  % cnn_init_regression() % cnn_init_regression_convs()
    else %if(isequal(opts.prediction_type,'c'))
        net = cnn_init_classification(opts);
    end
end

%filename = strcat('traffic_images_I210_West_D7_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
%filename = strcat('traffic_images_I5_North_D11_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
%filename = strcat('traffic_images_65to70mph_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
%filename = strcat('traffic_images_I5_North_D7_',nb_days, 'days_',next_pred_pt, 'Pt'); 
%filename = strcat('traffic_images_I5_North_D11_',nb_days, 'days_', next_pred_pt, 'Pt_L35_T17h_10wStr'); %'traffic_images_H101_North_D7_'
%filename = strcat('traffic_images_I210_West_D7_',nb_days, 'days_', next_pred_pt, 'Pt_L35_T08h40_10wStr'); %'traffic_images_H101_North_D7_'
%filename = strcat('traffic_images_SR1_North_',nb_days, 'days_', next_pred_pt, 'Pt_L329.55_T15h05_10wStr'); %'traffic_images_H101_North_D7_'
% imdb_filename = strcat('imdb_',filename,'_', opts.prediction_type,'.mat'); %imdb_classification_D07-US101N.mat
% if exist(imdb_filename, 'file')
%     load(imdb_filename) ; %  imdb = load(imdb_filename) ;
% else
%   imdb = setup_data(opts, kfold, filename,next_pred_pt);
%   %%%save(imdb_filename, '-struct', 'imdb') ;
% end
if kfold ==0 % testing
    filename = strcat('traffic_images_',test_freeway ,'_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
else % training
    filename = strcat('traffic_images_',freeway ,'_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
end
    imdb = setup_data(opts, kfold, filename,next_pred_pt);


%% compute batch size
%opts.batchSize iss the number of training images in each batch.
% if(kfold == 0) % use all testing dataset in one batch
%     opts.batchSize = length(imdb.images.labels) ; %150
% else
     opts.batchSize = 150 ; %150 for training %10000 for testing
% end

%% -------------------------------------------------------------------
%    Train
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
fprintf('Lowest validation error is %f in epoch %d\n',minm*70, min_ind)
end

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
%getBatch is called by cnn_train.

%'imdb' is the image database.
%'batch' is the indices of the images chosen for this batch.

%This is where we 'jitter' data - no jittering is done for this project
N = length(batch);
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

end


