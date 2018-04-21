clc;
clear; clc;
addpath(genpath('libsvm-3.22'));
addpath(genpath('liblinear-2.2'));
%% Given Data
train_image = double(loadMNISTImages('train-images.idx3-ubyte'));
train_label = double(loadMNISTLabels('train-labels.idx1-ubyte'));
test_image = double(loadMNISTImages('t10k-images.idx3-ubyte'));
test_label = double(loadMNISTLabels('t10k-labels.idx1-ubyte'));
% Retrieve dimension and sample number
[d,N] = size(train_image);
[td, tn] = size(test_image);
% 2. Create covariance matrix S
X_bar = mean(train_image, 2);
S = (train_image-repmat(X_bar, [1,N])) * (train_image-repmat(X_bar,[1,N]))' .* (1/N);
% 3. Singular Value Decomposition of S
[U, D, V] = svd(S);
%% linear SVM
disp('Run liblinear SVM');
for p = [40,80,200]    
    G = U(:, 1:p);
    X_tr = train_image' * G;
    X_te = test_image' * G;
    X_tr = sparse(X_tr);
    X_te = sparse(X_te);   
    for c = [0.01, 0.1, 1, 10]        
        % Use linear kernel for large data set
        arg1 = strcat({'-s 2 -c '}, num2str(c), ' -q');
        linear_model = train(train_label, X_tr, arg1{1});
        [linear_label, linear_accuracy, v] = predict(test_label, X_te, linear_model, '-q');        
        display(sprintf('Reduced dimension: %d Penalty Parameter: %f  Linear Accuracy: %f ', p,c,linear_accuracy(1)));       
    end
end
%% SVM with radial basis kernel for classification
disp('Run Radial Basis Kernel SVM');
for p = [40,80,200]    
    % Project the original images to lower dimension
    G = U(:, 1:p);
    X_tr = train_image' * G;
    X_te = test_image' * G;
    X_tr = sparse(X_tr);
    X_te = sparse(X_te);   
    for c = [0.01, 0.1, 1, 10]
        for gamma = [0.01, 0.1, 1]
            % In the function svmtrain t=2 for RBFN 
            arg2 = strcat({'-t 2 -c '}, num2str(c), {' -g '}, num2str(gamma), {' -h 0 -q'});
            nonlinear_model = svmtrain(train_label, X_tr, arg2{1});
            [nonlinear_label, nonlinear_accuracy, decision_values] = svmpredict(test_label, X_te, nonlinear_model, '-q');           
            display(sprintf('Reduced dimension: %d Penalty Parameter: %f Gamma Parameter: %f  Non-linear Accuracy: %f ', p,c,gamma,nonlinear_accuracy(1)));
        end
    end
end
% 
