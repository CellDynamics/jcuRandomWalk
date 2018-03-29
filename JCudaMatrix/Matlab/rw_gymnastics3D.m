clear all; close all;

SPYES=true;
%incidence matrix A of size edges*vertices

%consider image with R rows, C columns
R=25; %rows
C=25; %columns;
Z=25; %z-slices, layers
V=R*C*Z; % number of vertices

%it will have R*(C-1)+C*(R-1) edges in each layer
VerticesInLayer=R*C;

EdgesInLayer=R*(C-1)+C*(R-1);
E=Z*EdgesInLayer +(Z-1)*R*C; %number of edges in layers plus edges connecting layers

%ExV sized adjacency matrix
% note that we consider no flux boundary conditions here!!!

A=sparse(E,V);

%start filling A with horizontal edges
count=1;
for k=1:Z-1
for i=1:R
for j=1:C-1
    A(count, (i-1)*C+j  + (k-1)*VerticesInLayer)=1; % right edges
    A(count, (i-1)*C+j+1 + (k-1)*VerticesInLayer)=-1; % left edges
    count=count+1;
end
end
    
for j=1:C
    for i=1:R-1
    A(count, (i-1)*C+j + (k-1)*VerticesInLayer )=1; % down edges
    A(count, i*C+j  + (k-1)*VerticesInLayer)=-1; % up edges
    count=count+1;
end
end

end



for k=1:Z-1
    
    for i=1:R
for j=1:C
    A(count, (i-1)*C+j  + (k-1)*VerticesInLayer)=1; % to layer above edges
    A(count, (i-1)*C+j + k*VerticesInLayer)=-1; % to layer down edges
    count=count+1;
end
end

end
 

if SPYES
    w= ones(1,E); % all equal weights
W = spdiags(w', 0, E, E);     
%graph Laplacian   

L=sparse(V-4,V-4);
else
%weight (similarity) matrix with equal weights of each edge        
w= ones(E,1); % all equal weights

% w( floor((C-1)/2)+(C-1)*[0:R-1])=0.1; % block horizontal diffusion across the vertical midline

W = diag(w);
   
% 
end

L=A'*W*A;


%we are looking for the solution phi which solves Lap*phi = 0,
%employing the boundary conditions phi(1)=1, phi(V)=0;
%yields a reduced linear system L*phi_inside = b which applies to 
%values of nodes  [2:V-1] inside the domain

%set left and right boundary values
phi_source=1; %% top left corner of image

phi_sink=0; %% bottom right corner of image

%L will become our reduced Laplacian


Centre= round(C*(R/2)+(C/2) + (Z/2-1)*R*C);
Source =[Centre,Centre+1]; %centre pixel and neighbour
Sink=[V,V-1]; %bottom right vertex and left neighbour

seeds=[Source,Sink];

Index_inside=[1:V];
Index_inside(seeds)=[];;

%remove rows with seeds from Laplacian
L(seeds,:)=[];

%incorporate boundary conditions into b


%L*phi_inside = b
b = zeros(V-length(seeds),1)  - sum(L(:,Source),2)*phi_source - sum(L(:,Sink),2)*phi_sink;
L(:,[seeds])=[];



if false
% solve linear system L*phi_inside=b;
phi_inside=L\b;

%append boundary values to obtain full result vector
phi = zeros(1,V);
phi(Index_inside) = phi_inside;
phi(Source) = phi_source;
phi(Sink) = phi_sink;


stack = reshape(phi,[C,R,Z]);
        
[X,Y,Z] = meshgrid([1:R],[1:C],[1:Z]);

xslice = [1,6,12];   
yslice = [];
zslice = 6;
slice(X,Y,Z,stack,xslice,yslice,zslice)

daspect([1 1 1])

end


if SPYES
% pw=java.io.PrintWriter(java.io.FileWriter('../data/L_3D.txt'));
% line=num2str(0:size(L,2)-1);
% %disp(line);
% %pw.println(line);
% for index=1:length(L)
%     %disp(index);
%     line=num2str(full(L(index,:)));
%     pw.println(line);
% end
% pw.flush();
% pw.close();

[i,j,val] = find(L);
data_dump = [j,i,val];
fid = fopen('../data/L_3D.txt','wt');
fprintf( fid,'%d %d %f\n', transpose(data_dump) );
fclose(fid);


% fid = fopen('../data/L_3D.txt','wt');
% for i = 1:size(L,1)
%     line=full(L(i, :));
%     if(mod(i,5)==0)
%         i
%     end
%     for j= 1:size(L,2)
%     fprintf(fid,'%g\t',line(j));
%     
%     end
%     fprintf(fid,'\n');
% end
% fclose(fid)
else
fid = fopen('../data/L_3D.txt','wt');
for i = 1:size(L,1)
    fprintf(fid,'%g\t',L(i,:));
    fprintf(fid,'\n');
end
fclose(fid)

end


if SPYES==false
fid = fopen('../data/X_3D.txt','wt');
for i = 1:size(phi,1)
    fprintf(fid,'%g\t',phi(i,:));
    fprintf(fid,'\n');
end
fclose(fid)
end

fid = fopen('../data/B_3D.txt','wt');
b(b==-0)=0;
for i = 1:size(b,1)
    
    fprintf(fid,'%g\t',b(i,:));
    fprintf(fid,'\n');
end
fclose(fid)

fid = fopen('../data/Seeds_3D.txt','wt');
for i = 1:size(Source,2)
     fprintf(fid,'%g\t%g',Source(i), phi_source);
    fprintf(fid,'\n');
end

for i = 1:size(Sink,2)
     fprintf(fid,'%g\t%g',Sink(i), phi_sink);
    fprintf(fid,'\n');
end
fclose(fid)



        
