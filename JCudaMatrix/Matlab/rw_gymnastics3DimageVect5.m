clear all; close all;

%reads in 3D image in Matlab format and outputs sparse Laplacian and b for
%solving the random walk problem Lx=b
%We consider no flux boundary conditions here.
%Sink is defined as all faces of the bounding box.

FORWARD=false
SolveInMATLAB=false;

stack =  load('../data/LifeActGFP_3D.mat');

stack=stack.LifeActGFP_3D;
%normalise intensities
stack=(stack-min(stack(:)));
stack=stack./max(stack(:));
sprintf("stack read in and normalised")

% view individual slices for checking
%  imagesc(squeeze(stack(:,:,47)))
% return

%consider image with R rows, C columns, Z layers
R=143; %rows
C=116; %columns;
Z=93; %z-slices, layers


%  R=21; %rows
%  C=31; %columns;
%  Z=41; %z-slices, layers

 % test stack
% stack=zeros(R,C,Z);
% stack([6:16],[11:21],[16:26])=1;


V=R*C*Z; % number of vertices
VerticesInLayer=R*C;
EdgesInLayer=R*(C-1)+C*(R-1);
E=Z*EdgesInLayer +(Z-1)*R*C; %number of edges in layers plus edges connecting layers

w= zeros(1,E); %weight matrix

sigma_grad=0.05;%0.3; %standard deviation for the intensity gradient weight
sigma_mean=1e6 ; %standard deviation for the difference from mean weight

if FORWARD
mean_source = 0.4; %TODO: change to reflect actual values of source pixels
else
    mean_source = 0.4;
end

%assemble incidence matrix A of size edges*vertices

VH=zeros(1,R*(C-1)); %left vertices of horizontal edges in layer z=1
VV=zeros(1,C*(R-1)); %top vertices of vertical edges in layer z=1
VZ=zeros(1,R*C); %vertices in layer z=1 that will connect to vertices one level above

count=1; %start with indices for left vertices of horizontal edges
for i=1:R
    for j=1:C-1
        VH(count)=(j-1)*R+i;
        count=count+1;
    end
end
COUNTH=1:length(VH);

count=1; %top vertices of vertical edges
for i=1:R-1
    for j=1:C
        VV(count)= (j-1)*R+i  ;
        count=count+1;
    end
end
COUNTV=1:length(VV);

count=1; %z-edges
for i=1:R
    for j=1:C
        VZ(count)=(j-1)*R+i;
        count=count+1;
    end
end
COUNTZ=1:length(VZ);





AI=[]; %will be filled with x-indices of sparse matrix A
AJ=[]; %will be filled with y-indices of sparse matrix A
AV=[]; %will be filled with actual values of sparse matrix A


for k=1:Z % we do this for each z-layer
    
    V1=VH  + (k-1)*VerticesInLayer; %indices of left vertices of horizontal edges in layer
    V2=V1+R; %indices of corresponding right vertices
    
    COUNT=COUNTH+(k-1)*length(COUNTH);
    AI=[AI,COUNT,COUNT];
    AJ=[AJ,V1,V2];
    AV=[AV,ones(1,length(COUNT)),-ones(1,length(COUNT))];
    
    w(COUNT)= exp(-0.5*( (stack(V1)-stack(V2)).^2 / sigma_grad^2 )  -0.5*( (stack(V1)-mean_source).^2 / sigma_mean^2  )        );
    
%     sprintf("horizontal edges, layer %d",k)
    
end


AHI=AI; %indices of horizontal edges
AHJ=AJ;
AHV=AV;

%continue similarly with vertical edges
AI=[];
AJ=[];
AV=[];

offset=Z*R*(C-1); %number of horizontal edges in all layers, used as offset
for k=1:Z
    
    V1=VV + (k-1)*VerticesInLayer;
    V2=V1+1;
    COUNT=COUNTV +(k-1)*length(COUNTV) + offset;
    AI=[AI,COUNT,COUNT];
    AJ=[AJ,V1,V2];
    AV=[AV,ones(1,length(COUNT)),-ones(1,length(COUNT))];
    
    w(COUNT)= exp(-0.5*( (stack(V1)-stack(V2)).^2 / sigma_grad^2 )   -0.5*( (stack(V1)-mean_source).^2 / sigma_mean^2 )    );
    
%     sprintf("vertical edges, layer %d",k)
    
end

AVI=AI;
AVJ=AJ;
AVV=AV;


%last not least we put in z-edges
AI=[];
AJ=[];
AV=[];

offset=Z*EdgesInLayer; %number of all horizontal and vertical edges in all layers

for k=1:Z-1
    V1=VZ + (k-1)*VerticesInLayer;
    V2=V1 + VerticesInLayer;
    COUNT=COUNTZ+(k-1)*length(COUNTZ) + offset;
    
    AI=[AI,COUNT,COUNT];
    AJ=[AJ,V1,V2];
    AV=[AV,ones(1,length(COUNT)),-ones(1,length(COUNT))];
    
    w(COUNT)= exp(-0.5*( (stack(V1)-stack(V2)).^2 / sigma_grad^2 )    -0.5*( (stack(V1)-mean_source).^2 / sigma_mean^2)     );
    
%     sprintf("up/down edges, layer %d",k)
    
end

AZI=AI;
AZJ=AJ;
AZV=AV;

%assemble all edges and construct sparse matrix
AI=[AHI,AVI,AZI];
AJ=[AHJ,AVJ,AZJ];
AV=[AHV,AVV,AZV];

A=sparse(AI,AJ,AV,E,V,2*E);


sprintf(" assembled adjacency matrix")

% nonsense
% for i=1:V
%     w(i)=w(i)* exp(-0.5*( (stack(i)-mean_source).^2 / sigma_mean^2 )      );
%     
% end


%create sparse weight matrix
W = spdiags(w', 0, E, E);
sprintf("W matrix set up")

%compute graph Laplacian
L=sparse(V,V);
L=A'*W*A;

sprintf("L computed")

%we are looking for the solution phi which solves Lap*phi = 0,
%employing the boundary conditions phi(Source)=1, phi(Sink)=0;
%yields a reduced linear system L*phi_inside = b

%set values for source to 1 and sink to 0
if FORWARD
phi_source=1;
phi_sink=0;
else
phi_source=0;
phi_sink=1;    
end

%L will be reduced when incorporating the boundary contitions (seeds)

%for now, we use the centre of the image volume as source
% Centre= ( round (Z/2)-1)*VerticesInLayer+R*(round(C/2)-1) +round(R/2) ; %round( (Z/2-1)*VerticesInLayer+R*(C/2-1) +R/2 );
Source =find(stack>0.5)'; %find(stack>0.5)'; 
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

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%5

seeds=[Source,Sink];

Index_inside=[1:V];
Index_inside(seeds)=[];

%remove rows with seeds from Laplacian
% % [LI,LJ,LV] = find(L);
% % LI(seeds)=[];
% % A=sparse(AI,AJ,AV,E,V,2*E);


L(seeds,:)=[];

%incorporate boundary conditions (seeds) into b

%L*phi_inside = b
b = zeros(V-length(seeds),1)  - sum(L(:,Source),2)*phi_source - sum(L(:,Sink),2)*phi_sink;

L(:,[seeds])=[];



if SolveInMATLAB %false: do not solve system in Matlab
    % solve linear system L*phi_inside=b;
    phi_inside=L\b;
    
    %append boundary values to obtain full result vector
    phi = zeros(1,V);
    phi(Index_inside) = phi_inside;
    phi(Source) = phi_source;
    phi(Sink) = phi_sink;
    
    % visualise small datasets directly, otherwise use View3D script
    if false
        stack = reshape(phi,[C,R,Z]);
        
        [X,Y,Z] = meshgrid([1:R],[1:C],[1:Z]);
        
        xslice = [1,6,12];
        yslice = [];
        zslice = 6;
        slice(X,Y,Z,stack,xslice,yslice,zslice)
        
        daspect([1 1 1]);
    end
    
end

% output Laplacian and vector b

[i,j,val] = find(L);
data_dump = [j,i,val];
if FORWARD
fid = fopen('../data/L_3D_FW.txt','wt');
else
fid = fopen('../data/L_3D_BW.txt','wt');    
end

fprintf( fid,'%d %d %f\n', transpose(data_dump) );
fclose(fid);

if SolveInMATLAB
    fid = fopen('../data/X_3D.txt','wt');
    for i = 1:size(phi,1)
        fprintf(fid,'%f\t',phi(i,:));
        fprintf(fid,'\n');
    end
    fclose(fid);
end


if FORWARD
fid = fopen('../data/B_3D_FW.txt','wt');
else
fid = fopen('../data/B_3D_BW.txt','wt');    
end

b(b==-0)=0;
for i = 1:size(b,1)
    
    fprintf(fid,'%f\t',b(i,:));
    fprintf(fid,'\n');
end
fclose(fid);

if FORWARD
fid = fopen('../data/Seeds_3D_FW.txt','wt');
else
fid = fopen('../data/Seeds_3D_BW.txt','wt');    
end
for i = 1:size(Source,2)
    fprintf(fid,'%f\t%f',Source(i), phi_source);
    fprintf(fid,'\n');
end

for i = 1:size(Sink,2)
    fprintf(fid,'%f\t%f',Sink(i), phi_sink);
    fprintf(fid,'\n');
end
fclose(fid);




"DONE"




