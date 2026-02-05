%% Start
close all              
clear                   
clc                     
data = xlsread('Data.xlsx');

seed = 42;              
rng(seed);             

nums = length(data); 
history = 12;        

for i = 1: nums - history
    res(i, :) = [data(i: i + history - 1)', data(i + history)];
end

% Normalization
X = res(:,1:end-1);
Y = res(:,end);
[x,psin] = mapminmax(X', 0, 1);

[y, psout] = mapminmax(Y', 0, 1);

%  Data splitting
num = size(res,1);
k = input('Whether to shuffle the sample (True: 1, False: 0)£º');
if k == 0
   state = 1:num;
else
   state = randperm(num);
end
ratio = 0.8;
train_num = floor(num*ratio);

x_train0 = x(:,state(1: train_num));
y_train = y(state(1: train_num))';

x_test0 = x(:,state(train_num+1: end));
y_test = y(state(train_num+1: end))';

for i = 1 : train_num
    x_train{i, 1} = x_train0(:,i);
end

for i = 1 : num-train_num
    x_test{i, 1}  = x_test0(:,i);
end

% Model creation
layers = [
    sequenceInputLayer(history)         
    gruLayer(4, 'OutputMode', 'last')  
    reluLayer                           
    fullyConnectedLayer(1)              
    regressionLayer];                   
 
% Config
options = trainingOptions('adam', ...      
    'MaxEpochs', 1200, ...                 
    'InitialLearnRate', 0.001, ...          
    'LearnRateSchedule', 'piecewise', ... 
    'LearnRateDropFactor', 0.1, ...        
    'LearnRateDropPeriod', 400, ...       
    'Shuffle', 'every-epoch', ...         
    'Plots', 'training-progress', ...     
    'Verbose',true);   

% Train
net = trainNetwork(x_train, y_train, layers, options);

re1 = predict(net, x_train);
re2 = predict(net, x_test );

Y_train = Y(state(1: train_num));
Y_test = Y(state(train_num+1:end));

pre1 = mapminmax('reverse', re1, psout);
pre2 = mapminmax('reverse', re2, psout);

error1 = sqrt(mean((pre1 - Y_train).^2));
error2 = sqrt(mean((pre2 - Y_test).^2));

R1 = 1 - norm(Y_train - pre1)^2 / norm(Y_train - mean(Y_train))^2;
R2 = 1 - norm(Y_test -  pre2)^2 / norm(Y_test -  mean(Y_test ))^2;

mae1 = mean(abs(Y_train - pre1 ));
mae2 = mean(abs(pre2 - Y_test ));

disp(['Train data R2£º', num2str(R1)])
disp(['Train data MAE£º', num2str(mae1)])
disp(['Train data RMSE£º', num2str(error1)])
disp(['Test data R2£º', num2str(R2)])
disp(['Test data MAE£º', num2str(mae2)])
disp(['Test data RMSE£º', num2str(error2)])

figure
plot(1: train_num, Y_train, 'r-^', 1: train_num, pre1, 'b-+', 'LineWidth', 1)
legend('True','Pre')
xlabel('Sample')
ylabel('Pre value')
title('Comparison of train data prediction Results')

figure
plot(1: num-train_num, Y_test, 'r-^', 1: num-train_num, pre2, 'b-+', 'LineWidth', 1)
legend('True','Pre')
xlabel('Sample')
ylabel('Pre value')
title('Comparison of test data prediction Results')

figure
plot((pre1 - Y_train )./Y_train, 'b-o', 'LineWidth', 1)
legend('Percentage error')
xlabel('Sample')
ylabel('Error')
title('Train data percentage error curve')

figure
plot((pre2 - Y_test )./Y_test, 'b-o', 'LineWidth', 1)
legend('Percentage error')
xlabel('Sample')
ylabel('Error')
title('Test data percentage error curve')

figure;
plotregression(Y_train, pre1, 'Train data', ...
               Y_test, pre2, 'Test data');
set(gcf,'Toolbar','figure');

future_num = 6;
inputnew{1,:} = y(89:100)';  
outputs = zeros(1,future_num); 
for i=1:future_num
    indata = inputnew{i};
    outputs(i) = predict(net,indata);
    inputnew{i+1,:} = [indata(2:end);outputs(i)];
end

realout = mapminmax('reverse',outputs',psout);

figure
plot(1:nums,data,'r-o','LineWidth',1)
hold on
plot(nums:nums+future_num,[data(end);realout],'b-*','LineWidth',1)
xlabel('Year')
ylabel('Pre value')
legend('History value','Prediction value')
grid on

lastValue = data(end);
predictedValues = [lastValue; realout];

years = (nums+1:nums+length(predictedValues))';

predictedTable = array2table([years, predictedValues],...
    'VariableNames', {'Year', 'PredictedValue'});

writetable(predictedTable, 'pre.csv');

disp('The predicted data has been saved to pre.csv');