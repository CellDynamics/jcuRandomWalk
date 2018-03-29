clear all; close all;

%Linear graph with 6 nodes, 5 edges

%incidence matrix A of size edges*vertices

%consider image with R rows, C columns
R=30; %rows
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
%employing the boundary conditions phi(1)=1, phi(V)=0;
%yields a reduced linear system L*phi_inside = b which applies to 
%values of nodes  [2:V-1] inside the domain

%set left and right boundary values
phi_source=1; %% top left corner of image

phi_sink=0; %% bottom right corner of image

%L will become our reduced Laplacian
L=Lap;

Centre= (V/2)-floor(3*C/4);
Source =[Centre,Centre+1]; %centre pixel and neighbour
Sink=[V,V-1]; %bottom right vertex and left neighbour

seeds=[Source,Sink];

Index_inside=[1:V];
Index_inside(seeds)=[];;

%remove rows with seeds from Laplacian
L(seeds,:)=[];

%incorporate boundary conditions into b

%subtract 1st and last column of L from both sides of the equation
%L*phi_inside = b
b = zeros(V-length(seeds),1)  - sum(L(:,Source),2)*phi_source - sum(L(:,Sink),2)*phi_sink;
L(:,[seeds])=[];

% solve linear system L*phi_inside=b;
phi_inside=L\b;

%append boundary values to obtain full result vector
phi = zeros(1,V);
phi(Index_inside) = phi_inside;
phi(Source) = phi_source;
phi(Sink) = phi_sink;


        
imagesc(reshape(phi,[C,R])')
daspect([1 1 1])


fid = fopen('L_reduced.txt','wt');
for i = 1:size(L,1)
    fprintf(fid,'%g\t',L(i,:));
    fprintf(fid,'\n');
end
fclose(fid)

fid = fopen('X_reduced.txt','wt');
for i = 1:size(phi,1)
    fprintf(fid,'%g\t',phi(i,:));
    fprintf(fid,'\n');
end
fclose(fid)

fid = fopen('B_reduced.txt','wt');
b(b==-0)=0;
for i = 1:size(b,1)
    
    fprintf(fid,'%g\t',b(i,:));
    fprintf(fid,'\n');
end
fclose(fid)

fid = fopen('Seeds_reduced.txt','wt');
for i = 1:size(Source,2)
     fprintf(fid,'%g\t%g',Source(i), phi_source);
    fprintf(fid,'\n');
end

for i = 1:size(Sink,2)
     fprintf(fid,'%g\t%g',Sink(i), phi_sink);
    fprintf(fid,'\n');
end
fclose(fid)



        
