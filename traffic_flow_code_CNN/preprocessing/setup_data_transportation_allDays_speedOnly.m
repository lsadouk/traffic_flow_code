%% code for traffic flow estimation
% Author : Lamyaa Sadouk

% Given states : speed / flow / occupancy   for time and absolute postmile 
% 5 min interval per time point
% variable mile interval per postmile point (i.e. detector) 
% --> convert to a fixed interval per postmile point
% then convert to multiple instances/frames/scenarios of 10x20 --10 spacial
% point (10 miles) and 20 time points (5*20 time intervals)

% For our study, we pick TRAJECTORY: California -> D07 (LA/Ventura) -> US101 highway
% for this TRAJECTORY, original Postmile range is 2.05 to 69.75
% but only 4 detectors in the last 18 miles (from postmile 52 to 69.75) ->
% so our study, updated original Postmile range is 2.05 to 52 (last
% detector is is 51.82)3


% convert to miles -> for miles where no detector is found, take the State
% of previous mile
starting_day = '09/01/2017';
ending_day = '09/30/2017'; %'10/30/2017'4

% I5-north/south-D7 for peak hours 7to9am, we delete 11 at the end and 77 sendors
freeway = input('Please select the freeway: H101_North_D7 / H101_South_D7 / I5_North_D7 / I5_South_D7 / I5_North_D11 / I450_North_D7 / I210_West_D7 ','s');
nb_days = daysact(starting_day, ending_day)+1;
%nb_days = 27; % 1. for D07 US101N(training and testing)
%nb_days = 15; % 2. for D07 US101S(testing only)
nb_VDS_given = input('Please enter the number of VDSs (82_D07-US101N & 93_D07-US101S & 90_D07-I5N &  99_D07-I5S & 101_D11-I5N & 98_I450_North_D7 & 91_I210_West_D7):  ');
%  nb_VDS_given = 82;  % 1. for D07 US101N(training and testing) % 82 sensors within the studied district freeway (H101 of D07) -4 (see above)
%  nb_VDS_given = 93;  % 2. for D07 US101S(training and testing) % 93 sensors within the studied district freeway (H101 of D07) -4 (see above)
nb_VDS_deleted_start= input('Please enter the number of VDSs you wish to delete at the beginning:  ');
%  nb_VDS_deleted_start = 4; % 1. for D07 US101N(training and testing)
%  nb_VDS_deleted_start = 3; % 1. for D07 I5N  D11-I5N(training and testing)
%  nb_VDS_deleted_start = 0; % 1. for D07 US101s( testing)  / I5S / I450_North_D7
nb_VDS_deleted_end= input('Please enter the number of VDSs you wish to delete at the end:  ');
%  nb_VDS_deleted_end = 0; % 1. for D07 US101N(training and testing) / I5N/I450_North_D7 
%  nb_VDS_deleted_end = 3; % 1. for D07 US101s( testing)  
%  nb_VDS_deleted_end = 2; % 1. for D07 I5S(training and testing)
next_pred_point = input('Please enter next prediction speed point 1(5min)/2(10min)/...): ');
image_count=1;

%do the following for each file
for day_count=1:nb_days  % 36 days (30 for training and 6 for testing)
    %% 1. read excel file for express way (highway) I-5 absolute postmile range of 0.27 to 100 
%data =xlsread(strcat('../data/original_data/pems_output_',freeway, '_red_7to9.xlsx'),day_count,'A:F'); 
%data =xlsread(strcat('../data/original_data/pems_output_',freeway, '_red_16to18.xlsx'),day_count,'A:F'); 
data =xlsread(strcat('../data/original_data/pems_output_',freeway, '_red.xlsx'),day_count,'A:F'); 
    time = data(:,1); 
    postmile = data(:,2); postmile = postmile(1+nb_VDS_deleted_start:nb_VDS_given-nb_VDS_deleted_end); % postmile contains postmiles repeated once only
    %Result: postmiles on the top of the list {69.75 63.75 59.70 59.58} are deteled 
    %occupancy = data(:,3); 
    speed = data(:,3); 
    %flow = data(:,5); 
    observation = data(:,4); 
    data_size= size(data,1);%14760


    %% create a matrix data_n of size data_size x 1 x 3 (speed, occupancy, flow)
    data_n = zeros(data_size, 1, 2);
    %data_n(:,:,1) = occupancy; 
    data_n(:,:,1) = speed; 
    %data_n(:,:,3) = flow; 
    data_n(:,:,2) = observation; 

    %% convert data_n to a matrix  of size data_size/nb_VDS_given x time x 3
    data_n = reshape(data_n, nb_VDS_given, [], 2);%82*180*4

    %% remove the deleted VDS  "nb_VDS_deleted"
    data_n = data_n(1+nb_VDS_deleted_start:end-nb_VDS_deleted_end, :,:); %78*180*4

    %% do interpolation column by column
    for column=1:size(data_n,2)
        index = find(data_n(:,column,2) < 70); % find index for which % observation is less than a threshold
        index_wout_st_end = index(index~=1 & index~=size(data_n,1));
        for index_count=1:size(index_wout_st_end)
            ind = index_wout_st_end(index_count);
            data_n(ind,column,:) = mean ( [data_n(ind-1,column,:), data_n(ind+1,column,:)] ); 
        end
        %end
    end

    %% create our final matrix data_n_oneMile w size nb_postmiles x time x 3
    % gather all detectors that are within range [min_postmile, min_postmile+1] at data_n_oneMile[1] 
    % within range [min_postmile+1, min_postmile+2] at data_n_oneMile[2] 
    % and so one

    %% setup dimension of n_data (with 1 mile for each line) 
    max_postmile = floor(max(postmile)); % get the max postmile (ending mile)
    min_postmile = floor(min(postmile)); % get the min postmile (starting mile)
    nb_postmiles = max_postmile - min_postmile+1 ; % overall distance of the studied district freeway
    i=1;
    data_n_oneMile = zeros(nb_postmiles, size(data_n,2), size(data_n,3)); %50*180*4
    for count_mile = min_postmile:max_postmile
        index = find(postmile >= count_mile  &  postmile < (count_mile+1));
        if(~isempty(index)) % if an index was found for that postmile range, 
            data_n_oneMile(i,:,1) = mean( data_n(index,:,1) ,1); % sum up over rows for occupancy 
            %data_n_oneMile(i,:,2) = mean( data_n(index,:,2) ,1); % sum up over rows for  speed
            %data_n_oneMile(i,:,3) = mean( data_n(index,:,3) ,1); % sum up over rows for flow
        else % if no index was found for that postmile range, 
            data_n_oneMile(i,:,1) = data_n_oneMile(i-1,:,1); % get occupancyof previous mile
            %data_n_oneMile(i,:,2) = data_n_oneMile(i-1,:,2); % get speed  of previous mile 
            %data_n_oneMile(i,:,3) = data_n_oneMile(i-1,:,3); % get flow of previous mile
        end
        i= i+1; % increment the mile point
    end

    %% so, we have our matrix data_n_oneMile of size 63*180*4 delete the observation (4th channel)
    data_n_oneMile = data_n_oneMile(:,:,1); %50*180*3

    %% normalization of our 3 states (speed, occupancy, flow)
    % 1.max_speed=73.2(~117.5km), min_speed=4.3(~6.4km)/speed_lim=70(~112.65km)
    % max_occupancy = 0.7316, min_occupancy = 0.0294
    % max_flow = 979.5, min_flow = 139

    % 1.for occupancy, nothing to do coz already normalized
    % 2.for speed,  convert speeds higher than 70 to 70
    speed_oneMile = data_n_oneMile(:,:,1);
    speed_oneMile(speed_oneMile >70) = 70;
    data_n_oneMile(:,:,1) = speed_oneMile ./70;
%     % 3. for flow, need a universal max and min (that we choose)to normalize
%     min_flow = 50; % divided by the # of lanes (2 to 4)
%     max_flow = 1200;
%     data_n_oneMile(:,:,3) = (data_n_oneMile(:,:,3) - min_flow)/ (max_flow-min_flow);

    %% now create a matrix frame that will contain our frame/image instances
    % frame/image size is  10 x 20 *3
    % #instances = (50-10+1)* (180-20+1-1) =6560 intances par day
    h_data = size(data_n_oneMile,1); % size of the overall space (i.e. 50)
    w_data = size(data_n_oneMile,2); % size of the overall time series (i.e. 180)
    h_stride = 10; % space interval
    w_stride = 10; % time interval
    if(day_count == 1) % if it is the 1st day, declare our matrix 'image' and vector 'label' 
        image = zeros(h_stride,w_stride,1, (h_data-h_stride+1)*(w_data-w_stride+1-next_pred_point)*nb_days  ); % size is 10x20x3x6560 [=(63-10+1)* (180-20+1-1)]
        label = zeros(1,  (h_data-h_stride+1)*(w_data-w_stride+1-next_pred_point)*nb_days ); %labels/ categories from 1 to 7
    end
    %6560 intances par day
    for row = 1:h_data-h_stride+1
        for column = 1:w_data-w_stride+1-next_pred_point % was 1:w_data-w_stride+1 (i.e. 180-20+1=161)
            %compute image
            image(:,:,:,image_count) = data_n_oneMile(row:row+h_stride-1, column:column+w_stride-1, :);
            %compute label/class from 1 to 7
            %label(1,image_count) = data_n_oneMile(row+h_stride/2-1-1, column+w_stride-1+next_pred_point, 1); % predictin the speed at location i+4
            label(1,image_count) = data_n_oneMile(row, column+w_stride-1+next_pred_point, 1); % predictin the speed at location i+4
            s = label(1,image_count) *70;
%             if(s >= 1 && s < 10)
%                     label(1,image_count) = 1;
%             elseif(s >= 10 && s < 20)
%                     label(1,image_count) = 2;
%             elseif(s >= 20 && s < 30)
%                     label(1,image_count) = 3;
%             elseif(s >= 30 && s < 40)
%                     label(1,image_count) = 4;
%             elseif(s >= 40 && s < 50)
%                     label(1,image_count) = 5;
%             elseif(s >= 50 && s < 60)
%                     label(1,image_count) = 6;
%             elseif(s >= 60 && s < 71)
%                     label(1,image_count) = 7;
%             end    
            image_count = image_count+1;
        end
    end
    %12 fold (10 for taining and  for testing)

    
end
% save images/frames and their corresponding labels with info about data
filename = strcat('traffic_images_', freeway, '_', int2str(nb_days), ...
    'days_',int2str(next_pred_point), 'Pt_',int2str(w_stride) ,'wStr.mat'); %'traffic_images_H101_South_D7_15days.mat'
save(filename,'image', 'label', 'freeway', 'nb_VDS_given', 'nb_VDS_deleted_start', 'nb_VDS_deleted_end', 'starting_day' ,'ending_day');

