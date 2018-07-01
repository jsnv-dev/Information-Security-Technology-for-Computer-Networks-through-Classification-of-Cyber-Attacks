function [net, Model]  = fuzzyNeuralNet_training(dir, trainFcn, hsize, valcheck, grad, epochN, epochF, numMFs, inmftype, outmftype, i)
    % This function is for the training of Fuzzy Neural Network.
    
    % dir = Directory of the training dataset.
    % trainFcn = training function of the Artificial Neural Network:
    % 'trainlm'		= Levenberg-Marquardt
    % 'trainbr'		= Bayesian Regularization
    % 'trainbfg'	= BFGS Quasi-Newton
    % 'trainrp'		= Resilient Backpropagation
    % 'trainscg'	= Scaled Conjugate Gradient
    % 'traincgb'	= Conjugate Gradient with Powell/Beale Restarts
    % 'traincgf'	= Fletcher-Powell Conjugate Gradient
    % 'traincgp'	= Polak-Ribiére Conjugate Gradient
    % 'trainoss'	= One Step Secant
    % 'traingdx'	= Variable Learning Rate Gradient Descent
    % 'traingdm'	= Gradient Descent with Momentum
    % 'traingd' 	= Gradient Descent
    % hsize = size of the hidden layers of the network
    % valcheck = maximum number of validation checks
    % grad = minumum value of gradient
    % epochN = maximum number of epoch for the training for the Artificial Neural Network part.
    % epochF = the number of the epochs in training for Fuzzy Logic part.
    % numMFs = the number of membership functions associated with each input.
    % inmftype = is a string array in which each row specifies the membership function type associated with each input.
    % membership function types: 'trimf' 'trapmf' 'gbellmf' 'gaussmf' 'gauss2mf' 'pimf' 'dsigmf' 'psigmf'
    % outmftype = is a string that specifies the membership function type associated with the output.
    % There can only be one output, because this is a Sugeno-type system. The output membership function type must be either linear or constant.
    % i = is a vector corresponds to the rows of the features that will be using in the training
    % net = the trained network of this function
    % Model = the Fuzzy Logic Model

    %   trainInputs - input data.
    %   trainTargets - target data.

    load(dir)

    trainInputs = trainInputs(i,:);

    x = trainInputs;
    t = trainTargets;

    % Create a Fitting Network
    hiddenLayerSize = hsize;
    net = fitnet(hiddenLayerSize,trainFcn);

    % Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};

    % Setup Division of Data for Training, Validation, Testing
    % For a list of all data division functions type: help nndivide
    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'sample';  % Divide up every sample
    net.divideParam.trainRatio = 100/100;
    net.divideParam.valRatio = 0/100;
    net.divideParam.testRatio = 0/100;

    % Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net.performFcn = 'mse';  % Mean Squared Error

    % Choose Plot Functions
    % For a list of all plot functions type: help nnplot
    net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        'plotregression', 'plotfit'};

    % CONFIGURE THE NETWORK
    % Set validation checks(VALIDATION CHECK represents the number of
    % successive iterations that the validation performance fails to decrease.)
    net.trainParam.max_fail = valcheck;

    % Set minimum gradient
    net.trainParam.min_grad = grad;

    % Set maximum number of epochs 
    net.trainParam.epochs = epochN;


    % INITIALIZE THE WEIGHTS AND BIASES
    net = init( net );

    % Train the Network
    trainnet = str2func(trainFcn);
    [net,tr] = trainnet(net,x,t);
    output = sim(net, trainInputs );
    TrainData = output.';
    dispOpt = zeros(1,1);
    split_range = 2;
    Model = train(TrainData,TrainClass,split_range,numMFs,inmftype,outmftype,dispOpt,epochF);
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
