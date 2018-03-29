clear all; close all;

%Linear graph with 6 nodes, 5 edges

%incidence matrix A of size edges*vertices

%consider image with R rows, C columns
R=20; %rows
C=100; %columns;
V=R*C; % number of vertices

%it will have R*(C-1)+C*(R-1) edges
E=R*(C-1)+C*(R-1); %number of edges

%ExN sized adjacency matrix
% note that we consider no flux boundary conditions here!!!

A=zeros(E,V);

%start filling A with horizontal edges
count=1;
for i=1:R
    for j=1:C-1
        A(count, (i-1)*C+j  )=1; % right edges
        A(count, (i-1)*C+j+1  )=-1; % left edges
        count=count+1;
    end
end

for j=1:C
    for i=1:R-1
        A(count, (i-1)*C+j  )=1; % down edges
        A(count, i*C+j  )=-1; % up edges
        count=count+1;
    end
end

%weight (similarity) matrix with equal weights of each edge
w= ones(E,1); % all equal weights

w( floor((C-1)/2)+(C-1)*[0:R-1])=0.1; % block horizontal diffusion across the vertical midline

W = diag(w);

%graph Laplacian
Lap=A'*W*A;








%we are looking for the solution phi which solves Lap*phi = 0,
%employing the boundary conditions phi(Source)=1, phi(Sink)=0;


phi_source=1;
phi_sink=0.5; %usually set to zero

Centre= (V/2)-floor(3*C/4);
Source =[Centre,Centre+1,Centre+2]; %centre pixel and neighbours
Sink=[V,V-1,V-2]; %bottom right vertex and left neighbour

seeds=[Source,Sink];

%incorporate boundary conditions into b
Index_inside=[1:V];
Index_inside(seeds)=[];


b = zeros(V,1)  ;

b(Source)=sum(Lap(Source,Source),2)*phi_source;
b(Sink)=sum(Lap(Sink,Sink),2)*phi_sink;

v=[1:V];
j=v;
j(Source)=[];
k=v;
k(Sink)=[];

Lap(Source,j)=0; %in rows with Source nodes, set all non-Source entries to zero
Lap(Sink,k)=0;  %in rows with Sink nodes, set all non-Sink entries to zero
%make sure Sources have only outgoing and sinks only ingoing edges in adjacency matrix ??? 






% b = zeros(V,1)  ;
% b(Index_inside) =  - sum(Lap(Index_inside,Source),2)*phi_source  - sum(Lap(Index_inside,Sink),2)*phi_sink;
% 
% b(Source)=sum(Lap(Source,Source),2)*phi_source;
% b(Sink)=sum(Lap(Sink,Sink),2)*phi_sink;
% 
% v=[1:V];
% j=v;
% j(Source)=[];
% k=v;
% k(Sink)=[];
% 
% Lap(Source,j)=0;
% Lap(Index_inside,Source)=0;
% Lap(Sink,k)=0;
% Lap(Index_inside,Sink)=0;


% solve linear system L*phi_inside=b;
phi=Lap\b;


%%% Cholesky decomposition
%%%% A=LDL'
%%%%% A = LD^1/2  (LD^1/2)'
%%%solve Ly=b for y
%%%% solve  L'x=y for x

%%L=A'
 %%%Lap=A'DA
 
%  L=A'*W^1/2 ;
%  size(L)
%  Lap = L*L';
%%%solve Ly=b for y
%%%% solve  L'x=y for x

% y=L\b;
% phi=L'\y;




figure (1)
imagesc(reshape(phi,[C,R])')
daspect([1 1 1])

figure (2)
imagesc(Lap);

fid = fopen('Lap.txt','wt');
for i = 1:size(Lap,1)
    fprintf(fid,'%g\t',Lap(i,:));
    fprintf(fid,'\n');
end
fclose(fid)

fid = fopen('X.txt','wt');
for i = 1:size(phi,1)
    fprintf(fid,'%g\t',phi(i,:));
    fprintf(fid,'\n');
end
fclose(fid)

fid = fopen('B.txt','wt');
b(b==-0)=0;
for i = 1:size(b,1)
    
    fprintf(fid,'%g\t',b(i,:));
    fprintf(fid,'\n');
end
fclose(fid)




