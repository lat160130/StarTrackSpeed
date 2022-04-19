clc;
clear;

% we are reading in the matrices as column vectors
a3 = readmatrix('vectorInObs.txt');
b3 = readmatrix('vectorInRef.txt');
% x
% y
% z

% n = size(b);
% n = n(2);
% a_i = 1/n;


% B = zeros(3); % 3x3 matrix
% z = zeros(1,3); % 1*3 matrix
% for i = 1:n
%     mat = a_i* (b(:,i)' * r(:,i));
%     B = B + mat;
% 
%     matCr = a_i * cross(b(:,i), r(:,i));
%     z = z + matCr;
% 
% end % 

disp(a3'*b3);

a3 = a3(1:3,:);
b3 = b3(1:3,:);
disp(a3'*b3);

a2 = a3(1:2,:);
b2 = b3(1:2,:);
disp(a2'*b2);

a1 = a3(1,:);
b1 = b3(1,:);
disp(a1'*b1);

