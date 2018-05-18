
% test matrix - virtual matrix of indexes we assume to go through
a = reshape([0:4*3*3-1],3,4,3);
a = permute(a,[2,1,3]);

%% Retrieve r,c,z index from linear
% test of lin20ind and ind20lin
a % entries follow linear indexes

[v1, v2, v3] = lin20ind(a, size(a));
retndx = ind20lin(v1, v2, v3, size(a));

assert(all(all(all(ind20lin(v1, v2, v3, size(a))==a))))
%% Sparse

i = [1,2];
j = [1,2];
v = [1,-1];
s = sparse(i,j,v,3,3);
full(s)
%% generateAdjacency
nrows = 4;
ncols = 3;
nz = 2;

% test image - linear indexes row-ordered
stack = reshape(0:nrows*ncols*nz-1,ncols,nrows,nz);
stack = permute(stack,[2, 1, 3]);

V = nrows*ncols*nz; % number of vertices
VerticesInLayer = nrows*ncols;
EdgesInLayer = nrows*(ncols-1)+ncols*(nrows-1);
E = nz*EdgesInLayer +(nz-1)*nrows*ncols; %number of edges in layers plus edges connecting layers

[A, W] = generateAdjacency(stack, V, E);
fs = full(A);
assert(all(sum(abs(fs'))==2))

[yy,xx,zz]=meshgrid(1:nrows, 1:ncols, 1:nz);
yy = fliplr(yy);
G=graph(fs'*diag(W)*fs,'OmitselfLoops');
plot(G,'EdgeLabel',G.Edges.Weight,'XData',reshape(xx,1,[]),'YData',reshape(yy,1,[]), 'ZData', reshape(zz,1,[]))
grid on
figure % some edges will dissapear if weight is close to 0
G.Nodes.NodeColors = degree(G);
p=plot(G,'EdgeLabel',G.Edges.Weight);grid on;
layout(p,'subspace3')
p.NodeCData = G.Nodes.NodeColors;

figure
G=graph(fs'*fs,'OmitselfLoops');plot(G)
grid on
