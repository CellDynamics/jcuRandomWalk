clear all; close all;

%reads in 3D image in Matlab format and outputs sparse Laplacian and b for
%solving the random walk problem Lx=b
%We consider no flux boundary conditions here.
%Sink is defined as all faces of the bounding box.
tic
FORWARD = true
SecondPass = false
SolveInMATLAB = false;

if SecondPass
    stack =  load ('../data/stackSink.mat');
    stackSink=stack.stackSink;
    % imagesc(squeeze(stackSink(73,:,:)))
    % return
end

stack =  load('../data/LifeActGFP_3D.mat');

stack=stack.LifeActGFP_3D;
%normalise intensities
stack=medfilt3(stack);
stack=(stack-min(stack(:)));
stack=stack.^0.5; %%TTTTT
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


V = R*C*Z; % number of vertices
VerticesInLayer = R*C;
EdgesInLayer=R*(C-1)+C*(R-1);
E = Z*EdgesInLayer +(Z-1)*R*C; %number of edges in layers plus edges connecting layers

SOURCEMIN=0.7;

if FORWARD
    mean_source = 0.6; %TODO: change to reflect actual values of source pixels
else
    mean_source = 0.6;
end

% Note about compatibility with previous wersion:
% I use row ordering whereas previous version is column-wise.
% To get the same A there shoud be dimensions R and C swaped
% My procedure uses olso different edge order
[A, W_] = generateAdjacency(stack, V, E);

sprintf(" assembled adjacency matrix")

%create sparse weight matrix
W = spdiags(W_, 0, E, E);
sprintf("W matrix set up")
toc

%compute graph Laplacian
% L=sparse(V,V);
L=A'*W*A; % output is square, order should not matter

sprintf("L computed")
toc
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
Source =find(stack>SOURCEMIN)'; %find(stack>0.5)'; !! Potential incompatibility - I have row order

if SecondPass
    Sink = find(stackSink==1)';   
else
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
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
"Sourcebox defined"
toc


seeds=[Source,Sink];

Index_inside=[1:V];
Index_inside(seeds)=[];

%remove rows with seeds from Laplacian
% [LI,LJ,LV] = find(L);
% LI(seeds)=[];
% LJ(seeds)=[];
% LV(seeds)=[];
% L=sparse(AI,AJ,AV,E,V,2*E);


 L(seeds,:)=[];
 "L rows removed"
 toc

%incorporate boundary conditions (seeds) into b

%L*phi_inside = b
b = zeros(V-length(seeds),1)  - sum(L(:,Source),2)*phi_source - sum(L(:,Sink),2)*phi_sink;

L(:,seeds)=[];

 "L columns removed"
 toc


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




