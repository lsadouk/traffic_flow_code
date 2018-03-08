 w_stride = 10;
 nb_days='30';
next_pred_pt = input('Please enter next prediction speed point 1(5min)/2(10min)/...): ', 's');

%filename1 = strcat('traffic_images_H101_South_D7_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
filename2 = strcat('traffic_images_I5_North_D7_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
filename3 = strcat('traffic_images_I5_South_D7_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
filename4 = strcat('traffic_images_I5_North_D11_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
filename5 = strcat('traffic_images_I210_West_D7_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 
filename6 = strcat('traffic_images_I450_North_D7_',nb_days, 'days_',next_pred_pt, 'Pt_10wStr'); 

ScenesPath = strcat(next_pred_pt, '_point_estimate/'); %'data/preprocessed_data/';

% load(fullfile( ScenesPath, strcat(filename1, '.mat')), 'image', 'label'); 
% image1= image;
% label1 = label;

load(fullfile( ScenesPath, strcat(filename2, '.mat')), 'image', 'label'); 
image2= image;
label2 = label;

load(fullfile( ScenesPath, strcat(filename3, '.mat')), 'image', 'label'); 
image3= image;
label3 = label;

load(fullfile( ScenesPath, strcat(filename4, '.mat')), 'image', 'label'); 
image4= image;
label4 = label;

load(fullfile( ScenesPath, strcat(filename5, '.mat')), 'image', 'label'); 
image5= image;
label5 = label;

load(fullfile( ScenesPath, strcat(filename6, '.mat')), 'image', 'label'); 
image6= image;
label6 = label;

%label = cat(2, label1, label2, label3, label4, label5, label6);
%image = cat(4, image1, image2, image3, image4, image5, image6);
label = cat(2, label2, label3, label4, label5, label6);
image = cat(4, image2, image3, image4, image5, image6);


ind_0to20 = find(label>0 & label<=0.28); %20/70 = 0.2857
ind_20to40 = find(label>0.28 & label<=0.57); %40/70 = 0.5714
ind_40to60 = find(label>0.57 & label<=0.86); %60/70 = 0.8571
ind_60to65 = find(label>0.86 & label<=0.9286); %65/70 = 0.9286
ind_65to70 = find(label>0.9286); %60/70 = 0.8571
%ind_60to70 = find(label>0.86); %& label<=1


image_data1 = image(:,:,:,ind_0to20);
image_data2= image(:,:,:,ind_20to40);
image_data3= image(:,:,:,ind_40to60);
%image_data4= image(:,:,:,ind_60to70);
image_data4= image(:,:,:,ind_60to65);
image_data5= image(:,:,:,ind_65to70);

label_data1 = label(:,ind_0to20); %pt1:41230 /pt4:40323 /pt3:   /pt4:  /pt5:  /pt6:  /pt8
label_data2 = label(:,ind_20to40); %149884  /pt4:147076 /pt3:   /pt4:  /pt5:  /pt6:  /pt8
label_data3 = label(:,ind_40to60); %361632  /pt4:357253 /pt3:   /pt4:  /pt5:  /pt6:  /pt8
%label_data4 = label(:,ind_60to70); %832379  /pt4:815759 /pt3:   /pt4:  /pt5:  /pt6:  /pt8
label_data4 = label(:,ind_60to65); %832379  /pt4:815759 /pt3:   /pt4:  /pt5:  /pt6:  /pt8
label_data5 = label(:,ind_65to70); %832379  /pt4:815759 /pt3:   /pt4:  /pt5:  /pt6:  /pt8

%freeway = {'0to20mph', '20to40mph','40to60mph','60to70mph'};
image = image_data1; label = label_data1;
filename = strcat('traffic_images_', '0to20mph', '_', nb_days, ...
    'days_',next_pred_pt, 'Pt_',int2str(w_stride) ,'wStr.mat'); %'traffic_images_H101_South_D7_15days.mat'
save(filename,'image', 'label', 'nb_days');

% shuffle and reduce the dimen
nb_instances = 100000;
randNdx=randperm(length(label_data2));
image_data2=image_data2(:,:,:,randNdx); % 10*20*3*177120
label_data2=label_data2(1, randNdx); % 1*177120
image_data2 = image_data2(:,:,:,1:nb_instances);
label_data2 = label_data2(:,1:nb_instances);
image = image_data2; label = label_data2;
filename = strcat('traffic_images_', '20to40mph', '_', nb_days, ...
    'days_',next_pred_pt, 'Pt_',int2str(w_stride) ,'wStr.mat'); %'traffic_images_H101_South_D7_15days.mat'
save(filename,'image', 'label');

randNdx=randperm(length(label_data3));
image_data3=image_data3(:,:,:,randNdx); % 10*20*3*177120
label_data3=label_data3(1, randNdx); % 1*177120
image_data3 = image_data3(:,:,:,1:nb_instances);
label_data3 = label_data3(:,1:nb_instances);
image = image_data3; label = label_data3;
filename = strcat('traffic_images_', '40to60mph', '_', nb_days, ...
    'days_',next_pred_pt, 'Pt_',int2str(w_stride) ,'wStr.mat'); %'traffic_images_H101_South_D7_15days.mat'
save(filename,'image', 'label', 'nb_days');

randNdx=randperm(length(label_data4));
image_data4=image_data4(:,:,:,randNdx); % 10*20*3*177120
label_data4=label_data4(1, randNdx); % 1*177120
image_data4 = image_data4(:,:,:,1:nb_instances);
label_data4 = label_data4(:,1:nb_instances);
image = image_data4; label = label_data4;
filename = strcat('traffic_images_', '60to65mph', '_', nb_days, ...
    'days_',next_pred_pt, 'Pt_',int2str(w_stride) ,'wStr.mat'); %'traffic_images_H101_South_D7_15days.mat'
save(filename,'image', 'label', 'nb_days');

randNdx=randperm(length(label_data5));
image_data5=image_data5(:,:,:,randNdx); % 10*20*3*177120
label_data5=label_data5(1, randNdx); % 1*177120
image_data5 = image_data5(:,:,:,1:nb_instances);
label_data5 = label_data5(:,1:nb_instances);
image = image_data5; label = label_data5;
filename = strcat('traffic_images_', '65to70mph', '_', nb_days, ...
    'days_',next_pred_pt, 'Pt_',int2str(w_stride) ,'wStr.mat'); %'traffic_images_H101_South_D7_15days.mat'
save(filename,'image', 'label', 'nb_days');




