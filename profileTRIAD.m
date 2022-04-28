% % CPU
% command = "powershell -command Measure-Command { ./cpuTriad }";
% [status,cmdout] = system(command);
% status;
% cmdout;
% 
% % GPU
% command = "powershell -command Measure-Command { ./a }";
% [status,cmdout] = system(command);
% status;
% cmdout;


%[speedExDump, matches] = strsplit(cmdout, ':');
%speedExMili = str2double(C(12));

numRuns = 1000;
speedExVecCPU = zeros(1,numRuns);

for i = 1:numRuns
    % GPU
    command = "powershell -command Measure-Command { ./a }";
    [status,cmdout] = system(command);
    status;
    cmdout;
    [speedExDump, matches] = strsplit(cmdout, ':');
    speedExMili = str2double(speedExDump(12));
    speedExVecCPU(i) = speedExMili;

end % i = 1:numRuns
histogram(speedExVecCPU)
