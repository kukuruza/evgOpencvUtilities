%% degenerateMultiviewData - create a degenerate camera configuration
%  The camera centeres are on one line


clear all

%% paths for generated data, make data_dir empty to skip writing

data_dir = '../multiviewData/degenerate/';

pose2_name = 'pose2.txt';
pose3_name = 'pose3.txt';

matches123_clear_name = 'matches123-clear.txt';
matches123_noisy_name = 'matches123-noisy.txt';

randRot = 0;
randPoints = 0.1;


%% data

% translation 2
t2 = [1 0 0]';

% translation 3
t3 = [2 0 0]';

% rotation 2
rotVec = [0.5 0.5 0.5];
rotVal = 0.0;
q = qGetRotQuaternion(rotVal, rotVec);
R2 = qGetR(q);

dlmwrite([data_dir pose2_name], [R2 t2 + rand(3,1)*randRot; 0 0 0 1], ' ');

% rotation 3
rotVec = [-0.5 0.5 0.5];
rotVal = -0.0;
q = qGetRotQuaternion(rotVal, rotVec);
R3 = qGetR(q);

dlmwrite([data_dir pose3_name], [R3 t3 + rand(3,1) * randRot; 0 0 0 1], ' ');

% projection matrices
P1 = [eye(3) zeros(3,1)];
P2 = [R2, -R2 * t2];
P3 = [R3, -R3 * t3];



% N random 3D points, homogeneous coordinates
x = [0 0 1 1];

% invalid projection for the 3rd camera
N = 41;
X3 = [(-4 : 0.1 : 0)', zeros(N,1), ones(N,1)];

% projections for the 1st and 2nd camera
x1 = x * P1';
x2 = x * P2';
x1 = x1 / x1(3);
x2 = x2 / x2(3);
X1 = ones(N,1) * x1;
X2 = ones(N,1) * x2;

% write clear matches
dlmwrite([data_dir matches123_clear_name], [X1(:,1:2) X2(:,1:2) X3(:,1:2)], ' ');

% add noise
X1(:,1:2) = X1(:,1:2) + (rand(N,2) - 0.5) * randPoints;
X2(:,1:2) = X2(:,1:2) + (rand(N,2) - 0.5) * randPoints;
X3(:,1:2) = X3(:,1:2) + (rand(N,2) - 0.5) * randPoints;

% write with noise
dlmwrite([data_dir matches123_noisy_name], [X1(:,1:2) X2(:,1:2) X3(:,1:2)], ' ');

