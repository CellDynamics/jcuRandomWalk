rowIndA = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/rowInd.txt') + 1;
colIndA = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/colInd.txt') + 1;
valA = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/val.txt');
<<<<<<< Updated upstream
b = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/b.txt');
=======
>>>>>>> Stashed changes

rowIndACOO = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/rowIndCOO.txt') + 1;
colIndACOO = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/colIndCOO.txt') + 1;
valACOO = importdata('/home/baniuk/Documents/jcuRandomWalk/jcuRandomWalk/data/valCOO.txt');
%%
all(colIndA==colIndACOO)
all(valA==valACOO)
colsA = max(colIndACOO);
rowsA = colsA;
nnzA = length(valACOO);

<<<<<<< Updated upstream
S = sparse(rowIndACOO, colIndACOO, valACOO, rowsA, colsA);

issparse(S)
spy(S)
%%
tic
x = gpuArray(S)\gpuArray(b);
toc
=======
s = sparse(rowIndACOO,colIndACOO,valACOO,rowsA,colsA);
>>>>>>> Stashed changes
