
b = readmatrix('vectorInObs.txt');
r = readmatrix('vectorInRef.txt');

n = size(b);
n = n(1);

%q = QUEST_mat(b, r, n);

a_i = 1/n;

B = zeros(3); % 3x3 matrix
z = zeros(1,3); % 1*3 matrix
for i = 1:n
    mat = a_i* (b(i,:)' * r(i,:));
    B = B + mat;

    matCr = a_i * cross(b(i,:), r(i,:));
    z = z + matCr;

end % 
    
    
S = B + B'; % 3x3 matrix
    
    


sigma = .5*trace(S);
kappa = trace(adjoint(S));
delta = det(S);


a = sigma^2 - kappa;
b_scalar = sigma^2 + z*z';




