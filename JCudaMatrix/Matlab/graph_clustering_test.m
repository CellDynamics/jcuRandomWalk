%% load test image of 5x5 size, scaled to 0-1 range
stack = double(imread('test_5x5_similar.tif'))/255;
nrows = 5;
ncols = 5;
nz = 1;
% get number of edges and vertices
V = nrows*ncols*nz; % number of vertices
EdgesInLayer = nrows*(ncols-1)+ncols*(nrows-1);
% number of edges in layers plus edges connecting layers
E = nz*EdgesInLayer +(nz-1)*nrows*ncols; 
% plot test image
figure(1)
imagesc(stack);axis square;colormap gray;
xticks(1:5); yticks(1:5)

%% generate incidence and weight matrix
sigmaGrad = 0.05;
clear fun
fun = @(stack, row, col, z, row1, col1, z1)computeWeightSimple(stack, row, col, z, row1, col1, z1, sigmaGrad);
[A, W] = generateAdjacency(stack, V, E, fun);

%% Visualise
L = A'*diag(W)*A;
[yy,xx,zz] = meshgrid(1:nrows, 1:ncols, 1:nz);
yy = fliplr(yy);
G = graph(L, 'OmitselfLoops');
figure(2)
plot(G,'EdgeLabel',G.Edges.Weight,'XData',reshape(xx,1,[]),'YData',reshape(yy,1,[]), 'ZData', reshape(zz,1,[]))
grid on
view(-90, -90);
figure(3) % some edges will dissapear if weight is close to 0
G.Nodes.NodeColors = degree(G);
p = plot(G,'EdgeLabel',G.Edges.Weight);grid on;
layout(p,'auto')
p.NodeCData = G.Nodes.NodeColors;

figure(4)
Ga = graph(A'*A,'OmitselfLoops');
Ga.Edges.Weight = W;
Ga.Nodes.NodeColors = degree(Ga);
p = plot(Ga,'EdgeLabel',Ga.Edges.Weight,'XData',reshape(xx,1,[]),'YData',reshape(yy,1,[]), 'ZData', reshape(zz,1,[]));
p.NodeCData = Ga.Nodes.NodeColors;
view(-90, -90);
grid on

% page rank
figure(6)
G = graph(L, 'OmitselfLoops');
p = plot(G,'EdgeLabel',G.Edges.Weight,'MarkerSize',5);
p.NodeCData = centrality(G,'PageRank');
layout(p,'auto')
colorbar
%% clustering
figure(7)
L = A'*spdiags(W, 0, E, E)*A;
[yy,xx,zz] = meshgrid(1:nrows, 1:ncols, 1:nz);
yy = fliplr(yy);
G = graph(L, 'OmitselfLoops');
p = plot(G,'EdgeLabel',G.Edges.Weight,'MarkerSize',5,'XData',reshape(xx,1,[]),'YData',reshape(yy,1,[]), 'ZData', reshape(zz,1,[]));
[V,D] = eigs(L, 2, 'smallestabs');
w = V(:,2);
highlight(p, find(w>=0),'NodeColor','r') % subgraph A
highlight(p, find(w<0),'NodeColor','b') % subgraph B
view(-90, -90);
%% Solve
Ws = spdiags(W, 0, E, E); % sparse diagonal
Sink = [1, 25];
Source = [9];
seeds = [Source, Sink];
Index_inside = [1:V]; % indexes of rows for unknown pixesl (not seed)
Index_inside(seeds) = [];

FORWARD = true;
L = A'*Ws*A;
phi_source = double(FORWARD);
phi_sink = double(~FORWARD);

% remove rows from L
L(seeds,:) = [];
% incorporate boundary conditions (seeds) into b
% L*phi_inside = b
b = zeros(V - length(seeds), 1)  - sum(L(:, Source), 2)*phi_source - sum(L(:, Sink), 2)*phi_sink;
% remove columns from L
L(:, seeds) = [];

% solve linear system L*phi_inside=b;
phi_inside = L\b;
% append boundary values to obtain full result vector
phif = zeros(1,V);
phif(Index_inside) = phi_inside;
phif(Source) = phi_source;
phif(Sink) = phi_sink;

% visualise
figure(5)
subplot(1,3,1)
imagesc(reshape(phif,5,5));axis square;xticks(1:5); yticks(1:5)

% backw
FORWARD = false;
L = A'*Ws*A;
phi_source = double(FORWARD);
phi_sink = double(~FORWARD);

% remove rows from L
L(seeds,:) = [];
% incorporate boundary conditions (seeds) into b
% L*phi_inside = b
b = zeros(V - length(seeds), 1)  - sum(L(:, Source), 2)*phi_source - sum(L(:, Sink), 2)*phi_sink;
% remove columns from L
L(:, seeds) = [];

% solve linear system L*phi_inside=b;
phi_inside = L\b;
% append boundary values to obtain full result vector
phib = zeros(1,V);
phib(Index_inside) = phi_inside;
phib(Source) = phi_source;
phib(Sink) = phi_sink;

% visualise
subplot(1,3,2)
imagesc(reshape(phib,5,5));axis square;xticks(1:5); yticks(1:5)

% final
subplot(1,3,3)
imagesc(reshape(phif>phib,5,5));axis square;xticks(1:5); yticks(1:5)