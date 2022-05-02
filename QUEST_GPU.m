clc;
clear;
close all;
b = readmatrix('vectorInObs.txt');
r = readmatrix('vectorInRef.txt');

n = size(b);
n = n(1);

%q = QUEST_mat(b, r, n);

a_i = 1/n;

Z = zeros(1,3); % 1*3 matrix

% now lets turn b,r,Z into GPU objects
