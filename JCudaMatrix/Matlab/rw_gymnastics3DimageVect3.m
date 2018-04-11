clear all; close all;


stack =  load('../data/LifeActGFP_3D.mat');

stack=stack.LifeActGFP_3D;
%normalise intensities
stack=(stack-min(stack(:)));
stack=stack./max(stack(:));


% imagesc(squeeze(stack(10,:,:)))




sprintf("stack read in and normalised")

SPYES=true;
%incidence matrix A of size edges*vertices

%consider image with R rows, C columns
R=143; %rows
C=116; %columns;
Z=93; %z-slices, layers

% R=12; %rows
% C=12; %columns;
% Z=12; %z-slices, layers

%  R=21; %rows
%  C=31; %columns;
%  Z=41; %z-slices, layers
% 
% stack=zeros(R,C,Z);
% stack([6:16],[11:21],[16:26])=1;
stack=permute(stack, [2 1 3]);

%output format then C R Z

V=R*C*Z; % number of vertices

%it will have R*(C-1)+C*(R-1) edges in each layer
VerticesInLayer=R*C;

EdgesInLayer=R*(C-1)+C*(R-1);
E=Z*EdgesInLayer +(Z-1)*R*C; %number of edges in layers plus edges connecting layers

%ExV sized adjacency matrix
% note that we consider no flux boundary conditions here!!!

% A=zeros(E,V);
% sprintf("created A")
w= zeros(1,E); %
%
% sprintf("created w")
sigma=0.1; %was 0.3
drawnow('update')
%start filling A with horizontal edges


VH=zeros(1,R*(C-1));
VV=zeros(1,C*(R-1));
VZ=zeros(1,R*C);
count=1;





for i=1:R
    for j=1:C-1
        VH(count)= (i-1)*C+j  ;
        count=count+1;
    end
end
COUNTH=1:length(VH);
count=1;
for j=1:C
    for i=1:R-1
        VV(count)=(i-1)*C+j;
        count=count+1;
    end
end
count=1;
COUNTV=1:length(VV);
for i=1:R
    for j=1:C
        VZ(count)=(i-1)*C+j;
        count=count+1;
    end
end
COUNTZ=1:length(VZ);


CC=length(COUNTH);


if true
    
    
%     Asize=size(A);
    
    
    AI=[];
    AJ=[];
    AV=[];
    
    for k=1:Z
        
        
        
        V1=VH  + (k-1)*VerticesInLayer;
        V2=V1+1;
        
        COUNT=COUNTH+(k-1)*CC;
        AI=[AI,COUNT,COUNT];
        AJ=[AJ,V1,V2];
        AV=[AV,ones(1,length(COUNT)),-ones(1,length(COUNT))];
        
%             ind= sub2ind(Asize,COUNT,V1);
%             A(ind )=1; % right edges
%                 ind= sub2ind(Asize,COUNT,V2);
%             A(ind )=-1; % left edges
        
        w(COUNT)= exp(-0.5*( (stack(V1)-stack(V2)).^2 / sigma^2 ));
        
        sprintf("horizontal edges, layer %d",k)
        
    end
    % count=count+1;
    %   size(A)
    % A
    %   return
    %
    % size(AI)
    % size(AJ)
    % size(AV)

    
    AHI=AI;
    AHJ=AJ;
    AHV=AV;
    
    
    AI=[];
    AJ=[];
    AV=[];
    
    
    CC=Z*R*(C-1);
    
    for k=1:Z
        
        V1=VV + (k-1)*VerticesInLayer;
        V2=V1+C;
        COUNT=COUNTV+CC +(k-1)*length(COUNTV);
        AI=[AI,COUNT,COUNT];
        AJ=[AJ,V1,V2];
        AV=[AV,ones(1,length(COUNT)),-ones(1,length(COUNT))];
        
%                  ind= sub2ind(Asize,COUNT,V1);
%             A(ind  )=1; % down edges
%              ind= sub2ind(Asize,COUNT,V2);
%             A(ind)=-1; % up edges
        
        w(COUNT)= exp(-0.5*( (stack(V1)-stack(V2)).^2 / sigma^2 ));
        
        
        sprintf("vertical edges, layer %d",k)
        
    end
    
    
    AVI=AI;
    AVJ=AJ;
    AVV=AV;
    
    
    AI=[];
    AJ=[];
    AV=[];
    
    
 
    
    
    CC=Z*R*(C-1) + Z*(R-1)*C;
    
    for k=1:Z-1
        
        
        V1=VZ + (k-1)*VerticesInLayer;
        V2=V1 + VerticesInLayer;
        COUNT=COUNTZ+CC +(k-1)*length(COUNTZ);
        
        AI=[AI,COUNT,COUNT];
        AJ=[AJ,V1,V2];
        AV=[AV,ones(1,length(COUNT)),-ones(1,length(COUNT))];
        
%              ind= sub2ind(Asize,COUNT,V1);
%             A(ind)=1; % to layer above edges
%              ind= sub2ind(Asize,COUNT,V2);
%             A(ind)=-1; % to layer down edges
        
        w(COUNT)= exp(-0.5*( (stack(V1)-stack(V2)).^2 / sigma^2 ));
        
        
        sprintf("up/down edges, layer %d",k)
        
        
    end
    
    
    
    AZI=AI;
    AZJ=AJ;
    AZV=AV;
    
    AI=[AHI,AVI,AZI];
    AJ=[AHJ,AVJ,AZJ];
    AV=[AHV,AVV,AZV];
    
    A=sparse(AI,AJ,AV,E,V,2*E);
    
    
end



  drawnow('update')

sprintf(" edges done")
%  w= ones(1,E);

"TTT"
 W = spdiags(w', 0, E, E);

 
 % w= ones(E,1);
% W = diag(w);

%graph Laplacian

sprintf("W matrix done")
%L=sparse(V-4,V-4);
L=sparse(V,V);


L=A'*W*A;


sprintf("L computed")

%we are looking for the solution phi which solves Lap*phi = 0,
%employing the boundary conditions phi(1)=1, phi(V)=0;
%yields a reduced linear system L*phi_inside = b which applies to
%values of nodes  [2:V-1] inside the domain

%set left and right boundary values
phi_source=1; %% top left corner of image

phi_sink=0; %% bottom right corner of image

%L will become our reduced Laplacian


Centre= round(V/2)%round(C*(R/2)+(C/2) + (Z/2-1)*R*C);
Source =[Centre,Centre+1]; %centre pixel and neighbour
% Sink=[V,V-1]; %bottom right vertex and left neighbour
  Sink=[]; %bottom right vertex and left neighbour


%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

for k=2:Z-1
for i=1:R

    Sink(end+1)=(i-1)*C+1  + (k-1)*VerticesInLayer; 
    Sink(end+1)= (i-1)*C+C + (k-1)*VerticesInLayer; 
    

end
    
for j=2:C-1
    
    Sink(end+1)= j + (k-1)*VerticesInLayer; 
    Sink(end+1)= (R-1)*C+j  + (k-1)*VerticesInLayer; 
    

end

end



for k=[1,Z]
    
    for i=1:R
for j=1:C
   Sink(end+1)= (i-1)*C+j  + (k-1)*VerticesInLayer; % to layer above edges
   
end
end

end








%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%5
size(Sink)
[Uttt, Ittt] = unique(Sink, 'first');
x = 1:length(Sink);
x(Ittt) = [];
size(Sink)
find(Sink==1526099)
"TTTT1"
%return





seeds=[Source,Sink];

Index_inside=[1:V];
Index_inside(seeds)=[];;

size(L)
%remove rows with seeds from Laplacian
L(seeds,:)=[];

size(L)


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
    fprintf(fid,'%f\t',L(i,:));
    fprintf(fid,'\n');
end
fclose(fid)

end


if SPYES==false
fid = fopen('../data/X_3D.txt','wt');
for i = 1:size(phi,1)
    fprintf(fid,'%f\t',phi(i,:));
    fprintf(fid,'\n');
end
fclose(fid)
end

fid = fopen('../data/B_3D.txt','wt');
b(b==-0)=0;
for i = 1:size(b,1)
    
    fprintf(fid,'%f\t',b(i,:));
    fprintf(fid,'\n');
end
fclose(fid)

fid = fopen('../data/Seeds_3D.txt','wt');
for i = 1:size(Source,2)
     fprintf(fid,'%f\t%f',Source(i), phi_source);
    fprintf(fid,'\n');
end

for i = 1:size(Sink,2)
     fprintf(fid,'%f\t%f',Sink(i), phi_sink);
    fprintf(fid,'\n');
end
fclose(fid)


% 
% if SPYES
%     % pw=java.io.PrintWriter(java.io.FileWriter('../data/L_3D.txt'));
%     % line=num2str(0:size(L,2)-1);
%     % %disp(line);
%     % %pw.println(line);
%     % for index=1:length(L)
%     %     %disp(index);
%     %     line=num2str(full(L(index,:)));
%     %     pw.println(line);
%     % end
%     % pw.flush();
%     % pw.close();
%     
%     [i,j,val] = find(L);
%     data_dump = [j,i,val];
%     fid = fopen('../data/L_3D.txt','wt');
%     fprintf( fid,'%d %d %f\n', transpose(data_dump) );
%     fclose(fid);
%     
%     
%     % fid = fopen('../data/L_3D.txt','wt');
%     % for i = 1:size(L,1)
%     %     line=full(L(i, :));
%     %     if(mod(i,5)==0)
%     %         i
%     %     end
%     %     for j= 1:size(L,2)
%     %     fprintf(fid,'%g\t',line(j));
%     %
%     %     end
%     %     fprintf(fid,'\n');
%     % end
%     % fclose(fid)
% else
%     fid = fopen('../data/L_3D.txt','wt');
%     for i = 1:size(L,1)
%         fprintf(fid,'%g\t',L(i,:));
%         fprintf(fid,'\n');
%     end
%     fclose(fid)
%     
% end
% 
% 
% if SPYES==false
%     fid = fopen('../data/X_3D.txt','wt');
%     for i = 1:size(phi,1)
%         fprintf(fid,'%g\t',phi(i,:));
%         fprintf(fid,'\n');
%     end
%     fclose(fid)
% end
% 
% fid = fopen('../data/B_3D.txt','wt');
% b(b==-0)=0;
% for i = 1:size(b,1)
%     
%     fprintf(fid,'%g\t',b(i,:));
%     fprintf(fid,'\n');
% end
% fclose(fid)
% 
% fid = fopen('../data/Seeds_3D.txt','wt');
% for i = 1:size(Source,2)
%     fprintf(fid,'%g\t%g',Source(i), phi_source);
%     fprintf(fid,'\n');
% end
% 
% for i = 1:size(Sink,2)
%     fprintf(fid,'%g\t%g',Sink(i), phi_sink);
%     fprintf(fid,'\n');
% end
% fclose(fid)

"DONE"




