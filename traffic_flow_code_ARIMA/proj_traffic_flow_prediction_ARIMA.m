%function [net, info] = proj_traffic_flow_prediction_ARIMA()
    
    
    %% load dataset and prepare data vector
    data =xlsread('data/pems_output_ARIMA_I210W_D07_training_21weekdays_red.xlsx',1,'A:B'); 
    %testing_day=1; % testing day at the end of data
    data = data(:,2);
    end_pt = length(data)-288*1; % was -288*2  // -288*3 is good
    y_train = data(1:end_pt); % was data(1:end_pt);
    %y_train = data(end_pt+1:end_pt+20);
    T = length(y_train);
    %% estimate data
    %Mdl = arima(2,1,3);
    Mdl = arima(3,1,2);
    %Mdl = arima('Constant',0,'D',1,'Seasonality',288, 'MALags',1,'SMALags',288);
    %efault regression model with  errors:SARMA: (1,1)x(2,1,1)4
    %(1,0,1) (0,1,1)288
    %Mdl = regARIMA('ARLags',1,'SARLags',[4, 8],    'Seasonality',4,'MALags',1,'SMALags',4,'Intercept',0);
    %Mdl = regARIMA('ARLags',1,'Seasonality',288,'MALags',1,'SMALags',288,'Intercept',0, 'SARLags',[288 288*2]); % WORSE
    %Mdl =regARIMA('ARLags',1,'Seasonality',288,'MALags',1,'SMALags',288,'Intercept',0); %GOOD
    %Mdl =regARIMA('ARLags',1,'Seasonality',288,'MALags',1,'Intercept',0); %GOOD
    % Declare the ARMA model with seasonal MA component to be estimated (using X13-CENSUS)
    
    EstMdl = estimate(Mdl,y_train);
    
    error = zeros(length(y_test-20),1);

%% Trigger the estimation of the model
%%estimate(model);
    %% Use the fitted model to generate MMSE forecasts and corresponding mean square errors over a 40min (8points) horizon
    %data =xlsread('data/pems_output_ARIMA_I210W_D07_testing_1weekday_red.xlsx',1,'A:B'); 
    %y_test = data(:,2);
    %horizon = length(y_test);
    y_test = data(end_pt+1:end); % the full last day (the 21st weekday) % 72=starting from 6am
    next_pred_pt = 5;% was 288 % k=1 = 5min (prediction time interval)
    i=1;
for   count = 21:1:length(y_test)-next_pred_pt
    [yF,yMSE] = forecast(EstMdl,next_pred_pt,'Y0',y_test(1 :count)); %, 'YF',y_test(1:8));
    %[yF,yMSE] = forecast(EstMdl,horizon,'Y0',y_train(end-288: end)); %
    %GOOD one for dseasonality
    %RMSE = sqrt(yMSE);
    error(i) = (yF(end) - y_test(count+next_pred_pt)).^2;
    disp(error(i));
    i = i+1;
end
RMSE = sqrt(sum(error)/(count-21));
disp(RMSE);
% figure
% %plot(y_test(T+1 :T+horizon),'Color',[.75,.75,.75])
% plot(y_test(1:end),'Color',[.75,.75,.75])
% hold on
% %h1 = plot(T:T+horizon-1,yF,'r','LineWidth',2); 
% h1 = plot(yF,'r','LineWidth',2); 
% %xlim([starting_pt-1,T+horizon-1])
% 
% title('Forecast and 95% Forecast Interval')
% %legend([h1,h2],'Forecast','95% Interval','Location','NorthWest')
% hold off
% %end
% 
% %plot(y_train)
