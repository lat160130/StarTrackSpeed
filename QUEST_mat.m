function [q] = QUEST_mat(obsMat, refMat, n)

    % obsMat and refMat will be forced into an nx3 matrix, n is the number
    % of vectors, and each column is for the x,y,z component respectively
    % obsMat - b - body observation matrix, this matrix will hold all the vectors - W
    % refMat - r - inertial reference frame matrix - V
    % C*r_i = b_i - where C is the attitude matrix
    % Av = w

    % n = number of vectors

    a_i = 1/n; % this assumes all the vector pairs have equal weight, no reason to
    % believe otherwise.  This value will change if we have 
    
    % generate the attitude profile matrix B (not related to the body obs matrix)
    B = zeros(3); % 3x3 matrix
    z = zeros(1,3); % 1*3 matrix
    for i = 1:n
        mat = a_i* (obsMat(i,:)' * refMat(i,:));
        B = B + mat;

        matCr = a_i * cross(obsMat(i,:), refMat(i,:));
        z = z + matCr;

    end % 
    
    
    S = B + B';
    
    


    sig2  = .5*trace(S);
    kappa = trace(adjoint(S));

    q=1;

end