opts.prediction_type ='r';
kfold =input('Please enter the k-fold (k-1 for training & 1 for testing)_(0 for testing):  ');
nb_days =input('Please select the number of days (15, 21, 27, 30 or 59):  ','s');
next_pred_pt =input('Enter prediction point:  ','s'); %next_pred_pt = '6';


%% Load Database --------------------------------------------------------------
image_size = 10*10; % was 20*10;
filename = strcat('traffic_images_H101_North_D7_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
imdb_filename = strcat('../traffic_flow_code/imdb_',filename,'_', opts.prediction_type,'.mat'); %imdb_classification_D07-US101N.mat
imdb = setup_data(opts, kfold, filename,next_pred_pt);

imdb.images.data = permute(imdb.images.data, [2 1 3 4]); % to 10x20x1xnb_inst 
imdb.images.data = reshape(imdb.images.data, size(imdb.images.data,1)*size(imdb.images.data,2), []); %to 100 x nb_inst
%imdb.images.data = imdb.images.data'; % to nb_inst x 100

nb_samples_SVM = 40000;
trainIndex = find(imdb.images.set==1);
testIndex = find(imdb.images.set==2);
trainIndex = trainIndex(1: nb_samples_SVM);
%trainIndex = trainIndex(1: floor(nb_samples_SVM*(kfold-1)/kfold));
%testIndex = testIndex(1: floor(nb_samples_SVM*(1)/kfold));
trainX = imdb.images.data(:, trainIndex); 
testX = imdb.images.data(:, testIndex); %154000 too many %size=100*nb_instances
trainY = imdb.images.labels(:, trainIndex); %size=1*nb_instances
testY = imdb.images.labels(:, testIndex); %50000 too many  size=1*nb_instances
%imshow(trainX(:,:,:,10));
nb_train = size(trainY,2);
nb_test = size(testY,2);

%% SVM
%SVMModel = 	fitrsvm(trainX,trainY', 'Standardize',true,'KernelFunction','RBF','KernelScale','auto');  
SVMModel = 	fitrsvm(trainX',trainY', 'Standardize',true,'KernelFunction','RBF','KernelScale','auto'); 
% Check that the model converged:
conv = SVMModel.ConvergenceInfo.Converged;
% Use the trained model to predict the response of given predictor data:
predictions = predict(SVMModel,testX'); % prediction's size = 5000*1 (nb_instances*1)
%% Display data
disp('Classification error (testing):');
error =   sqrt(nansum((predictions - testY').^ 2) /nb_test); %  sum((predictions - labels).^ 2);
disp(error);
