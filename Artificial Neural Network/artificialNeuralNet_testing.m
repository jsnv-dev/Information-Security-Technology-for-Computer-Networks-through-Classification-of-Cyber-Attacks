function output = artificialNeuralNet_testing(dirNet, testingInputs, i)
    % This function is for the testing of Artificial Neural Network.
    
    % dirNet = directory of the network trained.
    % testingInputs = testing data to be classified with the netwok.
    % i = is a vector corresponds to the rows of the features that were used in the training.
    % output = classification of the testingInputs. (1 = Normal, 2 = DoS, 3 = Probe, 4 = U2R, 5 = R2L)
    
    load(dirNet) % Load the network trained in the artificialNeuralNet_training function.(ANN.mat = trained network in this study)
    
    testingInputs = testingInputs(i,:);
    
    result = sim(net, testingInputs);
    output = vec2ind(result);
end