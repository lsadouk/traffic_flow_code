next_pred_pt = '3';
%select prediction task: classification (7 labels) or regression (1 label)
%opts.prediction_type =input('Please select the prediction type: (c)classification / (r)regression  ','s');


%kfold = 0; % kfold=0 for TESTING and kfold = 4 for TRAINING (3 for training & 1 for testing)
%kfold =input('Please enter the k-fold (k-1 for training & 1 for testing)_(0 for testing):  ');

%nb_days =input('Please select the number of days (15, 21, 27, 30 or 59):  ','s');


  %Load sample data (with 7-day periodocity)
  
   data =xlsread('data/pems_output_ARIMA_I210W_D07_trainingTesting_5weekdays_red.xlsx',1,'A:B'); 
   %data =xlsread('data/pems_output_ARIMA_SR1_North_red_L329.55_T15h05_wStr10.xlsx',1,'A:B'); %contains 10 values for sample + 4 values for prediction(t+5,+10,+15,+20
   x = data(:,2);%x = data(:,2);
   T = length(x);
  %Estimate parameters
  initial_param = [0.7 0.1 0.001]; %  ? = 0.9, ? = 0.1, and ? = 0.001
  period = 288;
  [param] = fminsearch(@(param) holtwinters(param,period,x),initial_param);
  %Compute forecasts (fx)
  %[MAE,fx] = holtwinters(param,period,x);
  %data_test =xlsread('data/pems_output_ARIMA_SR1_North_red_L329.55_T15h05_wStr10.xlsx',1,'A:B'); %contains 10 values for sample + 4 values for prediction(t+5,+10,+15,+20
  y_test = data(:,2);%y_test = data(:,2);
  holtwinters_2(param,period,y_test) %param
  
  
  