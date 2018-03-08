function imdb = setup_data(opts, kfold, filename,next_pred_pt)

%%
% 'traffic_images_27days_info.mat' has: (a) a matrix image (width, 
% height,#channels, #instances) i.e. (spacial_data, time_data, states,
% #frames/scenes). (Ex: 10x20x3x1777120)(b) labels/classes (in range [0,1])
% (decimals)                            (c) min_postmile (2)
%                                       (d) max_postmile (51)
%                                       (e) #days (27)

%% 1. Load scenes/frames/images
ScenesPath = strcat('../../traffic_flow_code/preprocessing/',next_pred_pt, '_point_estimate/'); %'data/preprocessed_data/';
%filename = 'traffic_images_H101_North_D7_59days'; % 'traffic_images_H101_South_D7_15days_realLabel for testing  
% 'traffic_images_H101_North_D7_27days_realLabel.mat'  H101_North_D7 for 
% training and testing
% after loading, we have variables 'image' and 'label'
load(fullfile( ScenesPath, strcat(filename, '.mat')), 'image', 'label'); 
image = image;
label = label;
%% 2. if prediction_type = 'classification', then convert label of each instance to a value {1,2,3,4,5,6,7}
% knowing that label = the speed at t+1 of range [0,1] (need to multiply by
% 70
%label= round(70*label, -1); % round to the neared 10 to get values {0,10,20,30,40,50,60,70}
%label = label ./10; %{0,1,2,3,4,5,6,7}
%label(label ==0) = 1;
if(isequal(opts.prediction_type,'c'))
    label = 70*label;
    label(label>= 1 & label<=10) = 1;
    label(label> 10 & label<=20) = 2;
    label(label> 20 & label<=30) = 3;
    label(label> 30 & label<=40) = 4;
    label(label> 40 & label<=50) = 5;
    label(label> 50 & label<=60) = 6;
    label(label> 60 & label<=70) = 7;
% else if(equals(opts.prediction_type,'r')) DO NOTHING 
end

if(kfold == 0) % NO TRAINING PHASE all images are for testing + no Shufflg
    trainLabel=[];
    testData=image;
    testLabel=label;
else % if(kfold ~= 0)
    %% 3. shuffle the dataset & divide dataset into kfold (traing/testg)
    randNdx=randperm(length(label));
    image=image(:,:,:,randNdx); % 10*20*3*177120
    label=label(1, randNdx); % 1*177120
    %kfold = 4; % 3fold for training (first 21days) & 1fold for testing (last 7 days)
    sizekmul =size(image,4)-mod(size(image,4),kfold);  % for 10-fold cross validation %177120
    trainData=image(:,:,:,1:sizekmul/kfold*(kfold-1)); %3/4 samples are for training
    trainLabel=label(:,1:sizekmul/kfold*(kfold-1)); %3/4 samples are for training (10*20*3*132840)
    testData=image(:,:,:,sizekmul/kfold*(kfold-1)+1:sizekmul);%1/4 samples are for training %44280
    testLabel=label(:,sizekmul/kfold*(kfold-1)+1:sizekmul);%1/4 samples are for training
    
    %% 4. balance the training dataset for classification only (not for regression)-----------------------
        % balancing number of samples in each class to give to the classifier
        % 1. find maximum number of samples in  classes
        %             Nb_Sample_perClass=max([length(rockClassNdx),length(rockflapClassNdx),length(flapClassNdx)]);
        % or: according to fahd's code total number of samples/
    if(isequal(opts.prediction_type,'c'))
        nb_classes = length(unique(trainLabel));
        Nb_Sample_perClass=floor(length(trainLabel)/nb_classes); %18977samples/clas
        % 2. randomly resampling two other class with less number of data
        % points to Nb_Sample_perClass
        balanced_data = zeros(size(trainData,1), size(trainData,2), size(trainData,3), Nb_Sample_perClass*nb_classes); %10*20*3*132839
        for i=1:nb_classes
            class_iNdx=(find(trainLabel==i));  %class1
            balanced_data(:,:,:,(i-1)*Nb_Sample_perClass+1 : i*Nb_Sample_perClass) = balance_trainingData(trainData, class_iNdx, Nb_Sample_perClass); %2649 %26500
            trainLabel(1,(i-1)*Nb_Sample_perClass+1 : i*Nb_Sample_perClass) = i;
        end 
        trainLabel = trainLabel(1,1:Nb_Sample_perClass*nb_classes); % from 1*132840 to 1*132839
        randNdx=randperm(length(trainLabel));
        trainData=balanced_data(:,:,:, randNdx);
        trainLabel=trainLabel(1,randNdx);
    end
end
% test_data: class1=40, c2=710, c3=2534, c4=2671, c5=3115, c6=7427, c7=27783
%% 5. put all data into final dataset 'imdb'
nb_train = length(trainLabel); %or size(trainLabel,2)  132839
nb_test = length(testLabel); %44280
nb_total = nb_train + nb_test; %177119
image_size = [size(testData,1) size(testData,2) size(testData,3)]; 
imdb.images.data   = zeros(image_size(1), image_size(2),image_size(3), nb_total, 'single');
imdb.images.labels = zeros(1, nb_total, 'single'); % 1*n
imdb.images.set    = zeros(1, nb_total, 'uint8');

if(kfold ~= 0) % NO TRAINING PHASE all images are for testing
    imdb.images.data(:,:,:,1:nb_train) = trainData;
    imdb.images.labels(1, 1:nb_train) = single(trainLabel);
    imdb.images.set(1, 1:nb_train) = 1;
end

imdb.images.data(:,:,:,nb_train+1:nb_train+nb_test) = testData;
imdb.images.labels(1, nb_train+1:nb_train+nb_test) = single(testLabel);
imdb.images.set(:, nb_train+1:nb_train+nb_test) = 2;

imdb_filename = strcat('imdb_',filename,'_',opts.prediction_type, '.mat');
save(imdb_filename ,'imdb');
end
