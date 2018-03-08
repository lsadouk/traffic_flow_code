function balanced_data = balance_trainingData(trainData, classNdx, targetNumOfSamplesInEachClass )

% 2. randomly resampling the given class with less number of data
% points to targetNumOfSamplesInEachClass(18977)
balanced_data = zeros(size(trainData,1), size(trainData,2), size(trainData,3), targetNumOfSamplesInEachClass);
if 0<length(classNdx)&& length(classNdx)<targetNumOfSamplesInEachClass
    requiredSamplesLength=targetNumOfSamplesInEachClass-length(classNdx); %class1=123, requiredSamples=18854
    if requiredSamplesLength <= length(classNdx)
        requiredSamples = randsample(classNdx,requiredSamplesLength); %returns  k='requiredSamplesLength'  values sampled uniformly at random, without replacement, from the values in the vector classNdx
    else
        requiredSamples = datasample(classNdx,requiredSamplesLength); %returns k='requiredSamplesLength' observations sampled uniformly at random, with replacement, from the data in classNdx.
    end
    balanced_data(:,:,:,1:length(classNdx))=trainData(:,:,:,classNdx);
    balanced_data(:,:,:,length(classNdx)+1:end) = trainData(:,:,:,requiredSamples);
elseif 0<length(classNdx)&& length(classNdx)>targetNumOfSamplesInEachClass
    subsampleNdx = randsample(classNdx,targetNumOfSamplesInEachClass);
    balanced_data=trainData(:,:,:,subsampleNdx);
else
    balanced_data=trainData(:,:,:,classNdx);
end

%return balanced_data;
end

