r = 128;
c = 3;
fileID = fopen('vectorInRef.txt','w');

matObs  = zeros(r,c);
matRef  = zeros(r,c);


for i = 1:r
    for j = 1:c
        matObs(i,j) = rand(1);
        matRef(i,j) = rand(1);  
    end
    
end
writematrix(matObs, 'vectorInObs.txt', 'Delimiter', ' ');
writematrix(matRef, 'vectorInRef.txt', 'Delimiter', ' ');

writematrix(matObs', 'vectorInObsCM.txt', 'Delimiter', ' ');
writematrix(matRef', 'vectorInRefCM.txt', 'Delimiter', ' ');