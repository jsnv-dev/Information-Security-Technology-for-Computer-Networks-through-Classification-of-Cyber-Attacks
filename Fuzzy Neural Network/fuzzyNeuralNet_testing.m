function output = fuzzyNeuralNet_testing(dir, testingInputs, i)
    % This function is for the testing of the Internet packets using the trained Fuzzy Logic Model. 

    % dir = directory of the model and network trained.
    % testingInputs = testing data to be classified with the netwok.
    % i = is a vector corresponds to the rows of the features that were used in the training.
    % output = classification of the TestData. (0 = Normal, 0.25 = DoS, 0.5 = Probe, 0.75 = U2R, 1 = R2L)

    load(dir) % Load the trained model and network (FNN.mat = trained model and network in this study)
    
    testingInputs = testingInputs(i,:);

    test_output = sim(net, testingInputs );
    test_data = test_output.';
    test_result = classify(Model,test_data);
    output = result(test_result);
    
end

function Result = classify(Model,TestData)
    % This function classify the test data to the corresponding class of trained data using the trained model.
    % Inputs
    % Model -Anfis Multi Model
    % Test Data
    % Result Class of Test Data
    for iteration = 1:length(Model)
        splitrange = Model(iteration).splitrange;
        if iteration==1
            Result=zeros(size(TestData,1),length(splitrange)-1);
            for i = 1:length(splitrange)-1
                range=splitrange(i)+1:splitrange(i+1);
                Result(:,i) = evalfis(TestData(:,range),Model(iteration).AnfisModel{i});
            end
        else
            ResultOld = Result;
            Result = zeros(size(TestData,1),length(splitrange)-1);
            for i = 1:length(splitrange)-1
                range = splitrange(i)+1:splitrange(i+1);
                Result(:,i) = evalfis(ResultOld(:,range),Model(iteration).AnfisModel{i});
            end
            clear ResultOld
        end
    end
end

function output = result(Result)
    % This function optimized the ouput data . Converts the data from which its value is nearer (0, 0.25, 0.5, 0.75, or 1).
    % Example: .8551 => 0.75
    
    Result = round(Result*100)/100;
    a = Result <= .125;
    b = Result <= .375;
    b = b - a;
    c = Result <= .625;
    c = c - b - a;
    d = Result <= .875;
    d = d - c - b - a;
    e = Result > .875;
    a = a.* 0;
    b = b.* 0.25;
    c = c.* 0.5;
    d = d.* 0.75;
    output = a + b + c + d + e;

end