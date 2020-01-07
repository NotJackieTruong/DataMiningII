clear
close all
clc

%Load data
%dataset 1
%Ex1
dataSet1 = load('Iris.txt');
x1=dataSet1(:, 1:4);
y1=dataSet1(:, 5);

mean_x1 = mean(x1)
var_x1 = var(x1)
cov_x1 = cov(x1)
corr_x1 = corr(x1)
maxR1 = max(max(triu(x1,1)))
[row, col] = find(x1==maxR1, 1, 'first')

%Ex2
function pcadm2 (matrix)
  matrixMean = mean(matrix)
  Xadj = matrix - matrixMean
  CovXadj = cov(Xadj)
  %[U, D, pc] = svd(CovXadj)
  [V, D] = eig(CovXadj)
  [D, i] = sort(diag(D), 'descend')
  V = V(:,i)
  disp("Sorted D: "), disp(D)
  disp("Decend V: "), disp(V)
%project(matrix, Y)
endfunction

function P = project(A, B)
  P= (dot(A,B)/norm(B)^2)*B
endfunction


pcadm2(x1)

[coeff] = princomp(x1)