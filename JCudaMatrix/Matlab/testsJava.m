% number from com.github.celldynamics.jcurandomwalk.IncidenceMatrixGeneratorTest.testIncidenceMatrix()
r = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45];
r = r+1;
c = [4, 0, 1, 0, 12, 0, 5, 1, 2, 1, 13, 1, 6, 2, 3, 2, 14, 2, 7, 3, 15, 3, 8, 4, 5, 4, 16, 4, 9, 5, 6, 5, 17, 5, 10, 6, 7, 6, 18, 6, 11, 7, 19, 7, 9, 8, 20, 8, 10, 9, 21, 9, 11, 10, 22, 10, 23, 11, 16, 12, 13, 12, 17, 13, 14, 13, 18, 14, 15, 14, 19, 15, 20, 16, 17, 16, 21, 17, 18, 17, 22, 18, 19, 18, 23, 19, 21, 20, 22, 21, 23, 22];
val=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
c = c + 1;
% A is the same as in full() from com.github.celldynamics.jcurandomwalk.IncidenceMatrixGeneratorTest.testIncidenceMatrix()
A = sparse(r,c,val);

%% com.github.celldynamics.jcurandomwalk.RandomWalkAlgorithmTest.testComputeLaplacean_1()
% Incidence as A (from default test stack com.github.celldynamics.jcurandomwalk.TestDataGenerators.getTestStack(int, int, int, String)
% Weights mocked - all 2.0
W = 2*ones(1,size(A,1));
ws = spdiags(W', 0, size(A,1), size(A,1));

L = A'*ws*A;
fL = full(L);

%% computeWeight com.github.celldynamics.jcurandomwalk.IncidenceMatrixGeneratorTest.testComputeWeight()
p1 = 0.4;
p2 = 0.9;
sigma_grad = 0.1;
mean_source = 0.6;
sigma_mean = 1e6;
exp(-0.5*( (p1-p2).^2 / sigma_grad^2 )  -0.5*( (p1-mean_source).^2 / sigma_mean^2 ))

%% computeSinkBox com.github.celldynamics.jcurandomwalk.IncidenceMatrixGeneratorTest.testComputeSinkBox()

R=4; %rows
C=5; %columns;
Z=3; %z-slices, layers
stack = reshape([0:R*C*Z-1],R,C,Z);
stack = permute(stack,[1,2,3]);

V = R*C*Z; % number of vertices
VerticesInLayer = R*C;
EdgesInLayer=R*(C-1)+C*(R-1);
E = Z*EdgesInLayer +(Z-1)*R*C; %number of edges in layers plus edges connecting layers

[A, W_] = generateAdjacency(stack, V, E);

Sink=[]; %we will use the faces of the bounding box as sink


    %%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%

%this is painfully inefficient, change!
for k=2:Z-1
    for i=1:R %left and right faces of image bounding box
        Sink(end+1)= i  + (k-1)*VerticesInLayer;
        Sink(end+1)= (C-1)*R+i + (k-1)*VerticesInLayer;
    end
    for i=2:C-1 %top and bottom faces, without the outer edges that have been incorporated already
        Sink(end+1)= (i-1)*R + 1 + (k-1)*VerticesInLayer;
        Sink(end+1)=  (i-1)*R + R   + (k-1)*VerticesInLayer;
    end
end


for k=[1,Z] %all vertices in bottom and top layer of image volume
    for i=1:R
        for j=1:C
            Sink(end+1)= (j-1)*R+i  + (k-1)*VerticesInLayer;
        end
    end
end

Sink-1
%% remove rows
c=[0,1,5,1,2,3,0,4,5]+1;
r=[0,0,0,1,2,3,4,4,5]+1;
v=[10,101,102,11,12,13,131,14,15];

s=sparse(r,c,v);
full(s)
s' % to see order like in Java
% (col, row) in java
sr = s;
sr([3],:)=[]
full(sr)

sc = s;
sc(:,[1,3])=[]
%% com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDeviceTest.testLuSolve()

a=round(10*rand(5,5))/10;

a = [0.9, 0.4, 0.1, 0.9, 0.1;0.1, 0.9, 0.9, 0.6, 0.2;0.4, 0.2, 0.6, 0.4, 0.1;0.3, 0.3, 0.5, 0.5, 0.2;0.8, 0.1, 0.1, 0.4, 0.2]
x=[1;2;3;4;5]
b = a*x
