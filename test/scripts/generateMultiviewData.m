%% paths for generated data, make data_dir empty to skip writing

data_dir = '../multiviewData/';

pose2_name = 'pose2.txt';
pose3_name = 'pose3.txt';

E12_name = 'E1to2.txt';
E13_name = 'E1to3.txt';

matches123_clear_name = 'matches123-clear.txt';
matches12_clear_name = 'matches12-clear.txt';
matches23_clear_name = 'matches23-clear.txt';
matches31_clear_name = 'matches31-clear.txt';
matches123_noisy_name = 'matches123-noisy.txt';
matches12_noisy_name = 'matches12-noisy.txt';
matches23_noisy_name = 'matches23-noisy.txt';
matches31_noisy_name = 'matches31-noisy.txt';

randRot = 0;
randPoints = 0.1;


%% data

% translation 2
t2 = [1 0 0]';

% translation 3
t3 = [-0.2 0.5 0.2]';

% rotation 2
rotVec = [0.5 0.5 0.5];
rotVal = 0.5;
q = qGetRotQuaternion(rotVal, rotVec);
R2 = qGetR(q);

dlmwrite([data_dir pose2_name], [R2 t2 + rand(3,1)*randRot; 0 0 0 1], ' ');

% rotation 3
rotVec = [-0.5 0.5 0.5];
rotVal = -0.5;
q = qGetRotQuaternion(rotVal, rotVec);
R3 = qGetR(q);

dlmwrite([data_dir pose3_name], [R3 t3 + rand(3,1) * randRot; 0 0 0 1], ' ');

% projection matrices
P1 = [eye(3) zeros(3,1)];
P2 = [R2, -R2 * t2];
P3 = [R3, -R3 * t3];



% N random 3D points, homogeneous coordinates
N = 40;
X = rand(N,3) * 2 - 1;
X(:,3) = X(:,3) + 3 * ones(N,1);
X = [X ones(N,1)];

% projections
X1 = X * P1';
X2 = X * P2';
X3 = X * P3';
for i = 1 : N
    X1(i,:) = X1(i,:) / X1(i,3);
    X2(i,:) = X2(i,:) / X2(i,3);
    X3(i,:) = X3(i,:) / X3(i,3);
end

% split data into matches between views: 1-2-3, 1-2, 2-3, 3-1
range123 = 1:10;
range12 = 11:20;
range23 = 21:30;
range31 = 31:40;

% write clear matches
dlmwrite([data_dir matches123_clear_name], [X1(range123,1:2) X2(range123,1:2) X3(range123,1:2)], ' ');
dlmwrite([data_dir matches12_clear_name], [X1(range12,1:2) X2(range12,1:2)], ' ');
dlmwrite([data_dir matches23_clear_name], [X2(range23,1:2) X3(range23,1:2)], ' ');
dlmwrite([data_dir matches31_clear_name], [X3(range31,1:2) X1(range31,1:2)], ' ');

% add noise
X1(:,1:2) = X1(:,1:2) + (rand(size(X,1),2) - 0.5) * randPoints;
X2(:,1:2) = X2(:,1:2) + (rand(size(X,1),2) - 0.5) * randPoints;
X3(:,1:2) = X3(:,1:2) + (rand(size(X,1),2) - 0.5) * randPoints;

% write with noise
dlmwrite([data_dir matches123_noisy_name], [X1(range123,1:2) X2(range123,1:2) X3(range123,1:2)], ' ');
dlmwrite([data_dir matches12_noisy_name], [X1(range12,1:2) X2(range12,1:2)], ' ');
dlmwrite([data_dir matches23_noisy_name], [X2(range23,1:2) X3(range23,1:2)], ' ');
dlmwrite([data_dir matches31_noisy_name], [X3(range31,1:2) X1(range31,1:2)], ' ');



%% compute from two E matrices

% essential matrices
E2 = skew(t2) * R2;
E3 = skew(t3) * R3;

% epipoles
[~, ~, V] = svd(E2);
epipole1in2 = V(:,3);
[~, ~, V] = svd(E3);
epipole1in3 = V(:,3);

% trifocal tensor
for i = 1 : 3
    T(:,:,i) = P2(:,i) * P3(:,4)' - P3(:,i) * P2(:,4)';
end
T_from_E = T;


%% compute from 6 correspondances

% filling entries of A
A = zeros(24, 27);
for n = 1 : 6
    x1 = X1(n,:);
    x2 = X2(n,:);
    x3 = X3(n,:);
    for i = 1 : 2
        for l = 1 : 2
            a = zeros(3,2,2);
            for k = 1 : 3
                a(3,3,k) = x1(k) * x2(i) * x3(l);
                a(i,3,k) = - x1(k) * x2(l);
                a(3,l,k) = - x1(k) * x1(i);
                a(i,l,k) = x1(k);
            end
            A((n-1)*4 + (i-1)*2 + l, :) = reshape(a, [1 27 1]);
        end
    end
end

% solving min|At|, |t| = 1
[~, S, V] = svd(A);
t = V(:,end);
T = reshape(t, [3,3,3]);

%% check that tensor works

% point in space and its projections
Y = [1, 0, 0, 0]';
y1 = P1 * Y;
y2 = P2 * Y;
y3 = P3 * Y;

% should be zeros(3)
skew(y2) * (T(:,:,1) * y1(1) + T(:,:,2) * y1(2) + T(:,:,3) * y1(3)) * skew(y3);
