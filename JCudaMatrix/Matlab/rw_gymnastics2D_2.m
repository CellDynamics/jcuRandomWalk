clear all; close all;

%Linear graph with 6 nodes, 5 edges

%incidence matrix A of size edges*vertices

%consider image with R rows, C columns
R=20; %rows
C=100; %columns;
V=R*C; % number of vertices

%it will have R*(C-1)+C*(R-1) edges
E=R*(C-1)+C*(R-1); %number of edges

Centre= (V/2)-floor(3*C/4);
Source =[Centre,Centre+1,Centre+2]; %centre pixel and neighbours
Sink=[V,V-1,V-2]; %bottom right vertex and left neighbour

% Source=[1];
% Sink=[5];


seeds=[Source,Sink];


%ExN sized adjacency matrix
% note that we consider no flux boundary conditions here!!!

A=zeros(E,V);

%start filling A with horizontal edges
count=1;
for i=1:R
    for j=1:C-1
        
        
        v=(i-1)*C+j ;

        A(count, v )=1; % right edges  

        v=(i-1)*C+j +1;

        A(count, v  )=-1;

        count=count+1;
        end
        
end

for j=1:C
    for i=1:R-1
       
        v= (i-1)*C+j ;
        
          % if(find (seeds==v)>0)
         A(count, v )=1; % down edges
          v= i*C+j ;
        A(count, v )=-1; % up edges
                count=count+1;
                
           %end
    end
end

%weight (similarity) matrix with equal weights of each edge
w= ones(E,1); % all equal weights

w( floor((C-1)/2)+(C-1)*[0:R-1])=0.1; % block horizontal diffusion across the vertical midline

W = diag(w);

%graph Laplacian
Lap=A'*W*A;

L=Lap;
%we are looking for the solution phi which solves Lap*phi = 0,
%employing the boundary conditions phi(Source)=1, phi(Sink)=0;


phi_source=1;
phi_sink=0.5; %usually set to zero

%incorporate boundary conditions into b
Index_inside=[1:V];
Index_inside(seeds)=[];


b = zeros(V,1)  ;
b(Source)=sum(Lap(Source,Source),2)*phi_source;
b(Sink)=sum(Lap(Sink,Sink),2)*phi_sink;

% b(Source)=sum(Lap(Source,Source),2)*phi_source  + sum(Lap(Source,Index_inside),2);
% b(Sink)=sum(Lap(Sink,Sink),2)*phi_sink + sum(Lap(Sink,Index_inside),2);
% b = zeros(V,1) - length(Source)*phi_source - length(Sink)*phi_sink ;


v=[1:V];
nonSource=v;
nonSource(Source)=[];
nonSink=v;
nonSink(Sink)=[];

Lap(Source,nonSource)=0; %in rows with Source nodes, set all non-Source entries to zero
Lap(Sink,nonSink)=0;  %in rows with Sink nodes, set all non-Sink entries to zero


% solve linear system L*phi_inside=b;
phi=Lap\b;
% phi(Source)=phi_source;
% phi(Sink)=phi_sink;


%%% Cholesky decomposition
%%%% A=LDL'
%%%%% A = LD^1/2  (LD^1/2)'
%%%solve Ly=b for y
%%%% solve  L'x=y for x

%%L=A'
 %%%Lap=A'DA
 
%  L=A'*W^1/2 ;
 
%  L(Source,nonSource)=0; %in rows with Source nodes, set all non-Source entries to zero
%  L(Sink,nonSink)=0;  %in rows with Sink nodes, set all non-Sink entries to zero
%  
 
 
%  size(L)
%  Lap = L*L';
%%%solve Ly=b for y
%%%% solve  L'x=y for x
% 
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



