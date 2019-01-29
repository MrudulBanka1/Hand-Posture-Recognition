
clear all; clc; close all; 

%-----1.Load Raw Data----------%
% Load feat and label from Postures_userless in to the workspace
data = xlsread('Postures_userless.xlsx');
feat = data(:,2:13); % feature matrix
label = data(:,1);  % class label vector
C = unique(label); %extract label information from label vector

%-----2. Prepare N-fold dataset for classification----------%
N = 5; % N-fold cross validation 
data_nfold = divide_nfold_data(feat, label, N); 

%-----3. Perform N-fold Cross-Validation using different Functions-----------------% 
ACC_SUM = [];
        acc_nfold = []; 
        confusion_nfold = zeros(5,5);
        for ifold = 1:N 
           %----prepare cross-validation training and testing dataset---% 
           idx_test = ifold; % index for testing fold
           idx_train = setdiff(1:N, ifold); % index for training folds
           Dtest = []; Ltest = []; % initialize testing data and label
           Dtrain = []; Ltrain = []; % initialize testing data and label

           %---construct the training and testing dataset for the ith fold cross validatoin
           for iC = 1:length(C) 
               cl = C(iC);   
               dtest = eval(['data_nfold.class',num2str(cl), '.fold', num2str(ifold)]);
               Dtest = [Dtest; dtest]; 
               Ltest = [Ltest; cl*ones(size(dtest,1), 1)]; 

               for itr = 1:length(idx_train)
                   idx = idx_train(itr); 
                   dtrain = eval(['data_nfold.class',num2str(cl), '.fold', num2str(idx)]);
                   Dtrain = [Dtrain; dtrain];
                   Ltrain = [Ltrain; cl*ones(size(dtrain,1), 1)]; 
               end  
           end
           %---------------------------------------------------------%

           %--------------classification-------------------------%  
           % KNN classification using the function myknn_v1
           % myknn_v1: KNN classification based on majority voting
           % Lpred = myBayesPredict(Dtrain, Ltrain, Dtest,3); 
           % Mdl= fitctree(Dtrain, Ltrain);
           % Discriminant analysis
           % Mdl = fitcdiscr(Dtrain, Ltrain)
           % Naive Bayes
           % Mdl = fitcnb(Dtrain, Ltrain);
           % KNN 
           % Mdl = fitcknn(Dtrain, Ltrain,'NumNeighbors',3,'Standardize',1);
           % SVM
           % Mdl = fitcecoc(Dtrain, Ltrain);
           Lpred= predict(Mdl,Dtest);

        %---------------------------------------------------------%

           %---Calculate Classification Accuracy-----%
           acc = sum(Lpred==Ltest)/length(Ltest);  
           acc_nfold(ifold, 1) = acc; 
           
           %---Obtain Confusion Matrix based on Lpred and Ltest-----%
           confusion_mat = confusionmat(Lpred,Ltest);
           
        end
 
acc_ave = mean(acc_nfold) % average of N folds of cross validations
ACC_SUM = [ACC_SUM; acc_ave];

%% OPTIMIZED KNN AND BAGGED TREES
% Bagged Trees
% [trainedClassifier, validationAccuracy] = trainClassifier(data);
% Optimised KNN
%[trainedClassifier, validationAccuracy] = trainClassifier_knn(data);
validationAccuracy
%% CONFUSION CHART, RECALL, PRECISION, F-score
%----------------------- Confusion matrix-------------------------%
[confMat,order] = confusionmat(Lpred,Ltest);
cm = confusionchart(confusion_mat);
% confmat(confMat)
%----------------------- Recall/Sensitivity-------------------------%
for i =1:size(confMat,1)
    recall(i)=confMat(i,i)/sum(confMat(i,:));
end
recall(isnan(recall))=[];
Recall=sum(recall)/size(confMat,1);
Sensitivity = Recall
%----------------------- Precision -------------------------%
for i =1:size(confMat,1)
    precision(i)=confMat(i,i)/sum(confMat(:,i));
end
Precision=sum(precision)/size(confMat,1)
%----------------------- F-score -------------------------%
F_score=2*Recall*Precision/(Precision+Recall)




