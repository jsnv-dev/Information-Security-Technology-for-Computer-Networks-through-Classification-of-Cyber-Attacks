function Model = fuzzyLogic_training(dir,epoch_n,numMFs,inmftype, outmftype, i)
% This function is for the training of the Fuzzy Logic Model. 

% dir = the directory of the dataset to be used in the training.
% epoch_n = the number of the epochs in training.
% numMFs = the number of membership functions associated with each input.
% inmftype = is a string array in which each row specifies the membership function type associated with each input.
% membership function types: 'trimf' 'trapmf' 'gbellmf' 'gaussmf' 'gauss2mf' 'pimf' 'dsigmf' 'psigmf'
% outmftype = is a string that specifies the membership function type associated with the output.
% There can only be one output, because this is a Sugeno-type system. The output membership function type must be either linear or constant.
% i = is a vector corresponds to the columns of the features that will be using in the training.
% Model = the Fuzzy Logic Model.

load(dir) % Load the dataset file

TrainData = TrainData(:,i);

% TrainData = Data to be Train (rows = samples, columns = features)
% TrainClass = Class of train (rows = target output, 1 column)

dispOpt = zeros(1,1);
split_range = 2;
Model = train(TrainData,TrainClass,split_range,numMFs,inmftype,outmftype,dispOpt,epoch_n);

end

function Model=train(TrainData,TrainClass,split_range,numMFs,inmftype,outmftype,dispOpt,epoch_n)
  
    Model=struct('AnfisModel',{},'Reference',[],'splitrange',[]);
    iteration=1;
    while(1)
        if iteration==1
            [Model(iteration).AnfisModel,Model(iteration).Reference,Model(iteration).splitrange]=ANFIS.subtrain(TrainData,TrainClass,...
                split_range,inmftype,outmftype,numMFs,dispOpt,epoch_n );
        else
            [Model(iteration).AnfisModel,Model(iteration).Reference,Model(iteration).splitrange]=ANFIS.subtrain(Model(iteration-1).Reference,...
                TrainClass,split_range,inmftype,outmftype,numMFs,dispOpt,epoch_n );
        end
        if length(Model(iteration).splitrange)<3
            break
        end
        iteration=iteration+1;
    end
end

function [AnfisModel,Reference,splitrange]=subtrain(TrainData,TrainClass,split_range,mfType1,mfType2,numMFs,dispOpt,epoch_n )
    %% Split data for Better Classification
    lengthOfdata=size(TrainData,2);
    splitrange=zeros(100,1);
    tempVar=0;
    i=2;
    while(1)
        count=(lengthOfdata-tempVar);
        if count>split_range
            tempVar=tempVar+split_range;
            splitrange(i)=tempVar;
        elseif count<=split_range && count>0
            splitrange(i)=lengthOfdata;
            tempVar=lengthOfdata;
        else
            break;
        end
        i=i+1;
    end
    splitrange=splitrange(1:i-1);
    
    %% Anfis Train Model
    cycle=length(splitrange)-1;
    AnfisModel=cell(1,cycle);
    Reference=zeros(size(TrainData,1),cycle);
    for i=1:cycle
        dataRange=splitrange(i)+1:splitrange(i+1);
        fisdata=[TrainData(:,dataRange) TrainClass];
        fis1=genfis1(fisdata,numMFs,mfType1,mfType2);
        AnfisModel{i}=anfis(fisdata,fis1,epoch_n,dispOpt);
        Reference(:,i)=evalfis(TrainData(:,dataRange),AnfisModel{i});
    end
end
