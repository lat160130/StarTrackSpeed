
b = readmatrix('vectorInObs.txt');
r = readmatrix('vectorInRef.txt');

n = size(b);
n = n(1);

q = QUEST_mat(b, r, n);

a_i = 1/n;

test = b(1,:)' * r(1,:)