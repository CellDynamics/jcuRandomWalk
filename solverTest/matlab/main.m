rowIndA = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/rowInd.txt') + 1;
colIndA = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/colInd.txt') + 1;
valA = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/val.txt');
b = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/b.txt');
xout = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/x_out.txt');

rowIndACOO = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/rowIndCOO.txt') + 1;
colIndACOO = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/colIndCOO.txt') + 1;
valACOO = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/valCOO.txt');
%%
all(colIndA==colIndACOO)
all(valA==valACOO)
colsA = max(colIndACOO);
rowsA = colsA;
nnzA = length(valACOO);

S = sparse(rowIndACOO, colIndACOO, valACOO, rowsA, colsA);

issparse(S)
% spy(S)
%%
xs = S\b;
%%
tic
x = gpuArray(S)\gpuArray(b);
toc

