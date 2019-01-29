%% PROJECT: POSTURE PRUNED DATA
clear all; clc; close all;
pd = xlsread('Postures_pruned.xlsx');  %Read data from Excel file
% Sorting data by posture number
data = sortrows(pd,1);

%% FEATURE VARIABLES
class =data(:,1);
user=data(:,2);
X0=data(:,3);
Y0=data(:,4);
Z0=data(:,5);
X1=data(:,6);
Y1=data(:,7);
Z1=data(:,8);
X2=data(:,9);
Y2=data(:,10);
Z2=data(:,11);
X3=data(:,12);
Y3=data(:,13);
Z3=data(:,14);
ft_mat = data(:,3:14);
% Number of observations of each posture
c1= categorical(class);
summary(c1)
label = data(:,1);  % class label vector
C = unique(label);

%% 2D SCATTER PLOT FOR ALL VARIABLES
%{
figure;
gplotmatrix(ft_mat);%2D Scatter matrix
title('2D Scatter plot -1')
xlabel('X-Axis')
ylabel('Y-Axis')
%}
%{
meas = data(:,3:14) 
ds = mat2dataset(meas);
label = data(:,1);
ds(:,:);
size(ds);
ds.Properties;
min(ds.meas1);
ds.Properties.VarNames = {'XO','Y0','Z0','X1','Y1','Z1','X2','Y2','Z2','X3','Y3','Z3'};
ds.Properties.Description = 'Postures Data';
ds.Properties;
ds.Label = nominal(label);

figure;
gplotmatrix(meas,meas,label);
title('2D Scatter plot')
xlabel('X-Axis')
ylabel('Y-Axis')
%}
%% PARALLEL COORDINATES PLOT
%labels = {'XO','Y0','Z0','X1','Y1','Z1','X2','Y2','Z2','X3','Y3','Z3'};
%figure;
%parallelcoords(meas,'Group',label,'Labels',labels);
%% 3D SCATTER PLOT MATRIX
%{
figure;
scatter3(X0(1:50), Y0(1:50), Z0(1:50),'b')
hold on
scatter3(X0(51:100), Y0(51:100),Z0(51:100),'r')
hold on
scatter3(Z0(101:150), Y0(101:150),Z0(101:150),'g')
title('3D Scatter Plot')
xlabel('X-cordinate')
ylabel('Y-cordinate')
zlabel('z-cordinate')
%}
%% CORELATION MATRIX
figure;
% corrplot(ft_mat)
%corelation=corrplot(ft_mat);
%imagesc(corelation)
%% FEATURE MATRIX VISUALIZATION
%{
Z = 10+ peaks;
surf(Z)
figure;
imagesc(Z)
title('Featur Matrix Visualization (a)')
figure;
imagesc(ft_mat)
title('Featur Matrix Visualization (b)')
%}
%% NORMATLIZATION OF FEATURE MATRIX
%{
N = normalize(ft_mat);
figure;
plot(N)
figure;
Z=zscore(ft_mat);
plot(Z)
%}
%% HISTOGRAMS OF FEATURES
%{
% (61679:77405) - class 1, (61679:77405) - class 2, (61679:77405) - class
% 3, (61679:77405)- class 4, 
data1=X0(61679:77405);
data2=Y0(61679:77405);
data3=Z0(61679:77405);
data4=X1(61679:77405);
data5=Y1(61679:77405);
data6=Z1(61679:77405);
data7=X2(61679:77405);
data8=Y2(61679:77405);
data9=Z2(61679:77405);
data10=X3(61679:77405);
data11=Y3(61679:77405);
data12=Z3(61679:77405);
histogram(data1)
title('Histogram for Class 5')
xlabel('features')
ylabel('Number of occurence')
hold on
histogram(data2)
hold on
histogram(data3)
holdon
histogram(data4)
hold on
histogram(data5)
hold on
histogram(data6)
hold on
histogram(data7)
hold on
histogram(data8)
hold on
histogram(data9)
hold on
histogram(data10)
hold on
histogram(data11)
hold on
histogram(data12)
%}
%% BOXPLOTS OF FEATURES
%{
figure;
boxplot(X0,label)
title('Boxplot of X0 Coordinate for 5 classes')
xlabel('X0 Coordinate')
figure;
boxplot(X1,label)
title('Boxplot of X1 Coordinate for 5 classes')
xlabel('X1 Coordinate')
figure;
boxplot(X2,label)
title('Boxplot of X2 Coordinate for 5 classes')
xlabel('X2 Coordinate')
figure;
boxplot(X3,label)
title('Boxplot of X3 Coordinate for 5 classes')
xlabel('X0 Coordinate')
figure;
boxplot(Y0,label)
title('Boxplot of Y0 Coordinate for 5 classes')
xlabel('Y0 Coordinate')
figure;
boxplot(Y1,label)
title('Boxplot of Y1 Coordinate for 5 classes')
xlabel('Y1 Coordinate')
figure;
boxplot(Y2,label)
title('Boxplot of Y2 Coordinate for 5 classes')
xlabel('Y2 Coordinate')
figure;
boxplot(Y3,label)
title('Boxplot of Y3 Coordinate for 5 classes')
xlabel('Y3 Coordinate')
figure;
boxplot(Z0,label)
title('Boxplot of Z0 Coordinate for 5 classes')
xlabel('Z0 Coordinate')
figure;
boxplot(Z1,label)
title('Boxplot of Z1 Coordinate for 5 classes')
xlabel('Z1 Coordinate')
figure;
boxplot(Z2,label)
title('Boxplot of Z2 Coordinate for 5 classes')
xlabel('Z2 Coordinate')
figure;
boxplot(Z3,label)
title('Boxplot of Z3 Coordinate for 5 classes')
xlabel('Z3 Coordinate')

%}
%% PCA
%{
clear all; clc; close all;
rng 'default'
pd = xlsread('Postures_userless.xlsx');  %Read data from Excel file
% Sorting data by posture number
data = sortrows(pd,1);
X= data(:,2:13);

% De-mean (MATLAB will de-mean inside of PCA, but I want the de-meaned values later)
X = X - mean(X); % Use X = bsxfun(@minus,X,mean(X)) if you have an older version of MATLAB
% Do the PCA
[coeff,score,latent,~,explained] = pca(X);
% Calculate eigenvalues and eigenvectors of the covariance matrix
covarianceMatrix = cov(X);
[V,D] = eig(covarianceMatrix);
% "coeff" are the principal component vectors. These are the eigenvectors of the covariance matrix. Compare ...
coeff
V
% Multiply the original data by the principal component vectors to get the projections of the original data on the
% principal component vector space. This is also the output "score". Compare ...
dataInPrincipalComponentSpace = X*coeff
score
% The columns of X*coeff are orthogonal to each other. This is shown with ...
corrcoef(dataInPrincipalComponentSpace)
% The variances of these vectors are the eigenvalues of the covariance matrix, and are also the output "latent". Compare
% these three outputs
var(dataInPrincipalComponentSpace)'
latent
sort(diag(D),'descend')

x_comp = 1; % Principal component for x-axis
y_comp = 2; % Principal component for y-axis
figure
hold on
for nc = 1:12
    h = plot([0 coeff(nc,x_comp)],[0 coeff(nc,y_comp)]);
    set(h,'Color','b')
end
coeff
%}



