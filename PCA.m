clc;
clear all;
%% Load Train and Test Data
train_image = double(loadMNISTImages('train-images.idx3-ubyte'));
train_label = double(loadMNISTLabels('train-labels.idx1-ubyte'));
test_image = double(loadMNISTImages('t10k-images.idx3-ubyte'));
test_label = double(loadMNISTLabels('t10k-labels.idx1-ubyte'));
%% Dimensions of Data
[d,N] = size(train_image);
[td, tn] = size(test_image);
%% 2. Create covariance matrix S
X_bar = mean(train_image, 2);
S = (train_image-repmat(X_bar, [1,N])) * (train_image-repmat(X_bar,[1,N]))' .* (1/N);
%% 3. Singular Value Decomposition of S
%Projection matrix U
[U, D, V] = svd(S);
diag_vec = diag(D);
%% Task 1a: 2D - Visualization
disp('1a: Visualize projected data to 2D');
p = 2;
% Reduce dimension to 2
G2 = U(:, 1:p);
X2 = G2' * train_image;
data2fig = figure('Name', 'Visualisation of projected data in 2D');
%color = [ 0 0 0; 0 0 1; 0 1 0 ; 0 1 1 ; 1 0 0 ; 1 0 1 ; 1 1 0 ; 0.3 0.4 0.6;1 0.55 0; 0.5 0.5 0.5];
for number = 0:9
    mask = (train_label ==  number);
    a = X2(1,mask);
    b = X2(2,mask);
    c = train_label(mask);
    % Draw 2D visualization in separate view
    subplot(2,5,number+1);      
    scatter(a', b');
    title(['Number ' , num2str(number)]);
    % Draw 2D visualization in one graph
    plot_2d(number+1) = scatter(a', b',[], c,'.');
    hold on;
    title(['PCA 2D Visualization for number',num2str(number)]);
end
hold off;
% Plot eigen vectors of sample variance
vector2d_fig = figure('Name', '2D Eigen Plot');
eig1 = reshape(G2(:,1), [28,28]);
eig2 = reshape(G2(:,2), [28,28]);
subplot(1,2,1);
pcolor(eig1)
title('1st Eigen vectors of Sample Variance');
subplot(1,2,2);
pcolor(eig2);
title('2nd Eigen vectors of Sample Variance');
%% Task 1b: 3D - Visualization
p = 3; % Dimension
G3 = U(:, 1:p);
% Reconstruct train data matrix
X3 = G3' * train_image;
% Plot 3d figure
data3d_fig = figure('Name','Visualisation of projected data in 3D');
for number = 0:9
    mask = (train_label ==  number);
    a = X3(1,mask);
    b = X3(2,mask);
    c = X3(3,mask);
    color = train_label(mask);
    subplot(2,5,number+1);      
    scatter3(a', b', c', [],color, '.');
    title(['Number ' , num2str(number)]);    
end
%% Plot eigen vectors of sample variance
eigen3d_fig = figure('Name', '3D Eigen Plot');
eig1 = reshape(G3(:,1), [28,28]);
eig2 = reshape(G3(:,2), [28,28]);
eig3 = reshape(G3(:,3), [28,28]);
subplot(1,3,1);
pcolor(eig1);
title('1st Eigen vectors of Sample Variance');
subplot(1,3,2);
pcolor(eig2);
title('2nd Eigen vectors of Sample Variance');
subplot(1,3,3);
pcolor(eig3);
title('3rd Eigen vectors of Sample Variance');
figure
%% Task 2 : Dimensions of 40,80,200
p=[40 80 200];
for j = 1:3
    G = U(:,1:p(j));
    X_p = G' * train_image;
    X_t_p = G' * test_image;
    %% 5 Nearest Neighbours
    md = fitcknn(X_p',train_label,'NumNeighbors',10);
    label = predict(md,X_t_p');
    count = 0;
    for i = 1:size(test_label,1)
        if (label(i) == test_label(i,1))
            count = count + 1;
        end
    end
    accuracy(j) = (count/size(test_label,1))*100;
    ratio(j) = sum(diag_vec(1:p(j), 1)) / trace(D);
    fprintf('Accuracy of Dimension %d = %f , Energy Preserved = %f\r\n',p(j),accuracy(j),ratio(j)*100)
   
end
j = [40,80,200]
subplot(1,2,1)
plot(j,accuracy)
title('Accuracy with respect to dimension')
xlabel('Dimension');
ylabel('Accuracy')
subplot(1,2,2)
plot(j,ratio)
title('Energy Preserved with respect to dimension')
xlabel('Dimension');
ylabel('Energy Preserved')
%% 95% Energy Presevation
tr = trace(D);
sz = size(diag_vec, 1);
energy = 0; % current energy
index = 0;  % index where the eigen value preserves 95% total energy
for i = 1:sz
    energy = energy + diag_vec(i,1);
    if energy / tr >= 0.95
        index = i;
        break;
    end
end

%% Accuarcy for index
Gid = U(:,1:index);
X_ind = G' * train_image;
X_t_ind = G' * test_image;
md = fitcknn(X_ind',train_label,'NumNeighbors',5);
labelind = predict(md,X_t_ind');
countind = 0;
for i = 1:size(test_label,1)
    if (labelind(i) == test_label(i,1))
        countind = countind + 1;
    end
end
accuracy_ind = (countind/size(test_label,1))*100;
fprintf('Accuracy of Dimension %d = %f ,Energy Preserved = %f\r\n',index,accuracy_ind,(energy/tr)*100)



