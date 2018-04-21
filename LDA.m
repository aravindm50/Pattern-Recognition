clear all; 
clc;
%% Input Train and Test Data through loadMNISTImages and loadMNISTLabels functions
train_image = loadMNISTImages('train-images.idx3-ubyte');
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_image = loadMNISTImages('t10k-images.idx3-ubyte');
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');

%% Variable Initialization
counter = 0;
iterator = 0;
ratio = 0;
accuracy = double(zeros(10,1));
accuracy_mat = zeros(3,10);
%% Prepare data matrix

[d N] = size(train_image); % Train data size
[dt Nt] = size(test_image); % Test data size
Sw = zeros(d);   % Within class covariance
Sb = zeros(d);   % Between Class covariance
Mu = mean(train_image, 2);  % Get mean vector of train data

for classes = 0:9
    % LDA Step 2. Class specific covariance Si matrix of each category
    mask = (train_label ==  classes);
    x = train_image(:, mask);
    ni = size(x, 2);
    pi = ni / N;
    mu_i = mean(x, 2);
    Si = (1/ni) * (x - repmat(mu_i, [1,ni]))*(x - repmat(mu_i, [1,ni]))';
    % Sw within class covariance
    Sw = Sw + Si * pi;
    % Sb between class covariance
    Sb = Sb + pi * (mu_i - Mu) * (mu_i - Mu)';
end
%% Singular Value Decomposition of Scalar Objective Funnction Jw
Jw = pinv(Sw) * Sb;  
[U, D, V] = svd(Jw);

%% Task 1: 2D - Visualization

disp('1A: Visualisation of projected data in 2D and 3D plots');
R = 2; % Dimension
G2 = U(:, 1:R);
X2 = G2' * train_image;
data2d_fig = figure('Name', 'Visualisation of projected data in 2D');

for classes2 = 0:9
    
    mask = (train_label ==  classes2);
    a = X2(1,mask);
    b = X2(2,mask);
    c = train_label(mask);
    % 2D visualization in separate view
    subplot(2, 5, classes2+1)
    scatter(a',b','.');
    title(['Class = ', num2str(classes2)]);
end

%% Task 1b: 3D - Visualization
R = 3; % Dimension
G3 = U(:, 1:R);
X3 = G3' * train_image;
data3d_fig = figure('Name', 'Visualisation of projected data in 3D');
for classes3 = 0:9
    
    mask = (train_label ==  classes3);
    a = X3(1, mask);
    b = X3(2, mask);
    c = X3(3, mask);
    subplot(2,5,classes3+1);
    scatter3(a',b',c');
    title(['Number ', num2str(classes3)]);
end
%% Task 2: Classification
for p = [2, 3, 9]
    G = U(:, 1:p);
    Xtrain = G' * train_image;
    Xtest = G' * test_image;
    % Classify test data using Nearest Neighbor
    md = fitcknn(Xtrain',train_label,'NumNeighbors',5);
    label = predict(md,Xtest');
    count = 0;
    for i = 1:size(test_label,1)
        if (label(i) == test_label(i,1))
            count = count + 1;
        end
    end
    accuracy = (count/size(test_label,1));
    fprintf('Reduced Dimension = %f, Classification Accuracy = %f\r\n',p,accuracy*100);
    iterator = iterator + 1;
    accuracy_mat(iterator, :) = accuracy;
end

%% Task 3: Find maximum dimensionality via LDA

diag_vec = diag(D);
for value = 1:10
    fprintf('Eigen Value %d = %d\r\n',value,diag_vec(value))
end
disp('The objective function J(w) approaches to 0 on dimension 10');
