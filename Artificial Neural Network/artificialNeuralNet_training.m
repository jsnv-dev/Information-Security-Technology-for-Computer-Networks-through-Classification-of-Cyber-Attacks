function net = artificialNeuralNet_training(dir, trainFcn, hsize, valcheck, grad, epoch, i)
    % This function is for the training of Artificial Neural Network.
    
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
    % epoch = maximum number of epoch for the training
    % i = is a vector corresponds to the rows of the features that will be using in the training
    % net = the trained network of this function

    load(dir)

    trainInputs = trainInputs(i,:);

    x = trainInputs;
    t = trainTargets;

    % Create a Network
    hiddenLayerSize = hsize;
    net = patternnet(hiddenLayerSize,trainFcn);

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
    net.divideParam.testRatio =0/100;

    % Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net.performFcn = 'mse';  % Mean Squared Error

    % Choose Plot Functions
    % For a list of all plot functions type: help nnplot
    net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        'plotregression', 'plotfit'};

    % CONFIGURE THE NETWORK
    % Set validation checks (VALIDATION CHECK represents the number of
    % successive iterations that the validation performance fails to decrease.)
    net.trainParam.max_fail = valcheck;

    % Set minimum gradient
    net.trainParam.min_grad = grad;

    % Set maximum number of epochs
    net.trainParam.epochs = epoch;


    % INITIALIZE THE WEIGHTS AND BIASES
    net = init( net );

    % Train the Network
    trainnet = str2func(trainFcn);
    [net,tr] = trainnet(net,x,t);

    % Data needed for inaaccuracy
    trainX = trainInputs( : , tr.trainInd );
    trainT = trainTargets( : , tr.trainInd );
    valX = trainInputs( : , tr.valInd );
    valT = trainTargets( : , tr.valInd );
    testX = trainInputs( : , tr.testInd );
    testT = trainTargets( : , tr.testInd );

    % Feed the data to the network
    trainY = net( trainX );
    valY = net( valX );
    testY = net( testX );

    % Concatenate target matrices and output matrices
    allT = horzcat( trainT , valT , testT );
    allY = horzcat( trainY , valY , testY );

    % Compute for the inaccuracy
    allC = confusion( allT , allY );

    % Initialize variables for retraining of the network
    allCmat = allC;
    netmat = [];
    i = 4;

        while i > 0 % retrain the network 4 more times
            % Retrain the network
            [net , trainRecord] = trainnet( net , trainInputs , trainTargets );
            trainX = trainInputs( : , tr.trainInd );
            trainT = trainTargets( : , tr.trainInd );
            valX = trainInputs( : , tr.valInd );
            valT = trainTargets( : , tr.valInd );
            testX = trainInputs( : , tr.testInd );
            testT = trainTargets( : , tr.testInd );

            % Feed the data to the network
            trainY = net( trainX );
            valY = net( valX );
            testY = net( testX );
            allT = horzcat( trainT , valT , testT );
            allY = horzcat( trainY , valY , testY );

            % Recompute for the inaccuracy
            allC = confusion( allT , allY );

            allCmat = horzcat(allCmat, allC);

            filename = ['net' num2str(i) '.mat'];
            save (filename);
            i = i - 1;
        end

    % Picks the trained network with lowest inaccuracy
    allCmat = fliplr(allCmat);
    [j,k]  = min(allCmat);
    filename2 = ['net' num2str(k) '.mat'];
    load (filename2);	

end