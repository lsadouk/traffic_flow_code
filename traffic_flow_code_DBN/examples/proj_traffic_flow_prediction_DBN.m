
nnbox_dir = '../';
addpath(fullfile(nnbox_dir, 'networks'));
addpath(fullfile(nnbox_dir, 'costfun'));
addpath(fullfile(nnbox_dir, 'utils'));

opts.prediction_type ='r'; % prediction type is regression 
kfold =input('Please enter the k-fold (k-1 for training & 1 for testing)_(0 for testing):  ');
lambda =input('Please enter the loss (0)L2 loss, (1)P loss:  ');
nb_days =input('Please select the number of days (15, 21, 27, 30 or 59):  ','s');
next_pred_pt =input('Enter prediction point:  ','s'); %next_pred_pt = '6';
if kfold == 0
    test_freeway =  input('Please select the testing freeway: H101_North_D7 / I5_North_D7 / I5_South_D7 / I5_North_D11 / I450_North_D7 / I210_West_D7: ','s');
end
freeway = input('Please select the input freeway used for training: H101_North_D7 / I5_North_D7 / I5_South_D7 / I5_North_D11 / I450_North_D7 / I210_West_D7: ','s');


%% Load Database --------------------------------------------------------------
image_size = 10*10; 
nb_labels = 1;
filename = strcat('traffic_images_',freeway ,'_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
%filename = strcat('traffic_images_H101_North_D7_',nb_days, 'days_',next_pred_pt, 'Pt'); 
%filename = strcat('traffic_images_I5_North_D11_',nb_days, 'days_', next_pred_pt, 'Pt_L35_T17h');  %at 5PM%
%filename = strcat('traffic_images_I270_West_D7_',nb_days, 'days_', next_pred_pt, 'Pt_L35_T08h40'); %at 8:40AM%

if kfold ==0 % testing
    test_filename = strcat('traffic_images_',test_freeway ,'_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
    imdb = setup_data(opts, kfold, test_filename,next_pred_pt);
else % training
    imdb = setup_data(opts, kfold, filename,next_pred_pt);
end
  %save(imdb_filename, '-struct', 'imdb') ;
% end

imdb.images.data = permute(imdb.images.data, [2 1 3 4]); % to 10x20x1xnb_inst 
imdb.images.data = reshape(imdb.images.data, size(imdb.images.data,1)*size(imdb.images.data,2), []); %to 200 x nb_inst

trainIndex = find(imdb.images.set==1);
testIndex = find(imdb.images.set==2);
trainX = imdb.images.data(:, trainIndex);
testX = imdb.images.data(:, testIndex);
trainY = imdb.images.labels(:, trainIndex);
testY = imdb.images.labels(:, testIndex);

nb_train = size(trainY,2);
nb_test = size(testY,2);


%% Setup network --------------------------------------------------------------
% Start with an empty multilayer network skeleton
net  = MultiLayerNet();

if kfold > 0
    model_uns_filename = strcat('./model/model_unsupervised_',filename, '_L', int2str(lambda),'.mat'); %L0 l2 loss
    if exist(model_uns_filename, 'file')
       load(model_uns_filename); %  imdb = load(imdb_filename) ;
    else
        % Setup first layer
        pretrainOpts = struct( ...
            'nEpochs', 15, ... %15  then 100 then 200???
            'momentum', 0.7, ...
            'lRate', 1e-3, ...
            'batchSz', 150, ...
            'dropout', 0.3, ...
            'displayEvery', 5);
        trainOpts.nEpochs = 15; %15  then 100 then 200???
        trainOpts = struct( ...
            'lRate', 5e-4, ...
            'batchSz', 150);
        rbm1 = RBM(image_size, 1000, pretrainOpts, trainOpts);

        % Add first layer
        net.add(rbm1);

        % Setup second layer
        rbm2 = RBM(1000, 1000, pretrainOpts, trainOpts);
        % Add second layer
        net.add(rbm2);

    %     % Setup Third layer
    %     rbm3 = RBM(500, 2000, pretrainOpts, trainOpts);
    %     % Add Third layer
    %     net.add(rbm3);


        %% Pretrain network -----------------------------------------------------------
        fprintf('Pretraining first two layers\n');
        net.pretrain(trainX); % note: MultilayerNet will pretrain layerwise
        save(strcat('./model/model_unsupervised_', filename,'_L', int2str(lambda)),'net');
    end
    %% Train ----------------------------------------------------------------------

    % Add fully connected layer above
    trainOpts = struct(...
        'lRate', 1e-3, ... % was 1e-3
        'nIter', 60, ... % was 150, then 300, then 400
        'batchSz', 150, ...
        'displayEvery', 1);
    per  = Perceptron(1000, nb_labels, trainOpts); % was 2000
    net.add(per);

    % Train in a supervized fashion
    fprintf('Fine-tuning\n');

    train(net, SquareCost(), trainX, trainY, trainOpts,lambda); 
    save(strcat('./model/model_supervised_', filename,'_L', int2str(lambda), '_ep60'),'net'); %save(strcat('./model/model_supervised_', filename,'_L', int2str(lambda),'_ep60'),'net');
else % if kfold =0, do testing
    model_s_filename = strcat('./model/model_supervised_',filename, '_L', int2str(lambda), '_ep60','.mat'); %L0 l2 loss
    if exist(model_s_filename, 'file')
       load(model_s_filename); % 
    else
       fprintf('Error, no supervised file found.\n');
    end
end % end if kfold

%% Results --------------------------------------------------------------------
disp('Classification error (testing):');
predictions =net.compute(testX);
error = 70*  sqrt(nansum((predictions - testY).^ 2) /nb_test); %  sum((predictions - labels).^ 2);
disp(error);

% disp('Showing first layer weights as filters (20 largest L2 norm)');
% weights = net.nets{1}.W;
% [~, order] = sort(sum(weights .^2), 'descend');
% colormap gray
% for i = 1:20
%     subplot(5, 4, i);
%     imagesc(reshape(weights(:, order(i)), 10, 10)); %imagesc(reshape(weights(:, order(i)), 28, 28));
%     axis image
%     axis off
% end

