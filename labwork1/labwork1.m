clear
close all
clc

%Load data
%dataset 1
%Ex1
dataSet1 = load('Iris.txt');
x1=dataSet1(:, 1:4);

dataSet2 = load('wine.txt');
x2= dataSet2(:, 2:14);



% Find mean
mean_x1 = mean(x1)
mean_x2 = mean(x2)
% Find variance
var_x1 = var(x1)
var_x2 = var(x2)
% Find covariance
cov_x1 = cov(x1)
cov_x2 = cov(x2)
% Find correlation
corr_x1 = corr(x1)
corr_x2 = corr(x2)

% Find the most correlated couple of features
maxR1 = max(max(triu(x1,1)))
[row, col] = find(x1==maxR1, 1, 'first')

%Ex2
function [V,Xadj] = pcadm2 (matrix)
  matrixMean = mean(matrix);
  Xadj = matrix - matrixMean;
  CovXadj = cov(Xadj);
  %[U, D, pc] = svd(sigma);
  [V, D] = eig(CovXadj);
  [D, i] = sort(diag(D), 'descend');
  V = V(:,i);
  disp("Sorted D: "), disp(D)
  disp("Principle Component: "), disp(V)
  Y = transpose(V)*transpose(Xadj);

endfunction

function pcaCalculate (matrix)
  [V, Xadj]=pcadm2(matrix);
  [W, pc]= princomp(Xadj)
  pc = pc';
  W = W';
  group1 = pc(:, 1:50);
  group2 = pc(:, 51:100);
  group3 = pc(:, 101:150);
  figure(1)
  plot(group1(1,:), group1(2,:), 'b.', 'MarkerSize', 15, group2(1,:), group2(2,:), 'r.', 'MarkerSize', 15, group3(1,:), group3(2,:), 'g.', 'MarkerSize', 15); 
  legend('Setosa', 'Versicolor', 'Virginica');
  

endfunction

function pcaCalculate2 (matrix)
  [V, Xadj]=pcadm2(matrix);
  [W, pc]= princomp(Xadj)
  pc = pc';
  W = W';
  group1 = pc(:, 1:59);
  group2 = pc(:, 60:130);
  group3 = pc(:, 131:178);
  figure(2)
  plot(group1(1,:), group1(2,:), 'b.', 'MarkerSize', 15, group2(1,:), group2(2,:), 'r.', 'MarkerSize', 15, group3(1,:), group3(2,:), 'g.', 'MarkerSize', 15); 
  legend('Class1', 'Class2', 'Class3');


endfunction

% find principal components
pcadm2(x1);
[coeff] = princomp(x1);

pcadm2(x2)
[coeff2] = princomp(x2);
pcaCalculate2(x2);
%apply pca 
pcaCalculate(x1);


