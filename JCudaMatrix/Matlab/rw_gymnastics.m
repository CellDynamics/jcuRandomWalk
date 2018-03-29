clear all; close all;

%Linear graph with 6 nodes, 5 edges

%incidence matrix A of size edges*vertices

A = [  +1  -1   0   0   0  0
            0 +1 -1   0   0  0
            0   0 +1 -1   0  0
            0   0   0 +1 -1  0
            0   0   0   0 +1 -1  ];
        
        
[E,V] = size(A); % get number of edges, E, and vertices,V
        
  
%weight (similarity) matrix with equal weights of each edge        
W = [    1   0   0   0   0
            0   1   0   0   0
            0   0   1   0   0
            0   0   0   1   0
            0   0   0   0   1  ];
 
 %weight matrix with unequal weights             
 W1 = [ 1   0   0   0   0
            0  .9   0   0   0
            0   0  0.1  0   0
            0   0   0  .9   0
            0   0   0   0   1  ];       
         

        
%graph Laplacian   
Lap=A'*W1*A

%we are looking for the solution phi which solves Lap*phi = 0,
%employing the boundary conditions phi(1)=1, phi(V)=0;
%yields a reduced linear system L*phi_inside = b which applies to 
%values of nodes  [2:V-1] inside the domain

%set left and right boundary values
phi_1=1; phi_V=0;

%L will become our reduced Laplacian
L=Lap;

%remove first and last row from Laplacian
L([1,end],:)=[];

%incorporate boundary conditions into b

%subtract 1st and last column of L from both sides of the equation
%L*phi_inside = b
b = zeros(V-2,1)  - L(:,1)*phi_1 - L(:,end)*phi_V
L(:,[1,end])=[]

% solve linear system L*phi_inside=b;
phi_inside=L\b;

%append boundary values to obtain full result vector
phi = [phi_1 , phi_inside', phi_V];



%plot phi
plot(phi);       
        
        
        
%  %graph Laplacian   
% L=A'*W*A;
% 
% 
% %set boundary conditions
% %clamp first vertex to 1 (probability to be foreground), last vertex to zero
% b_1=1; b_end=0;
% 
% %create b, right hand side of Lx=b 
% b = zeros(V,1) - L(:,1)*b_1 - L(:,end)*b_end;
% 
% %remove first and last column from Laplacian as they have been shifted to
% %the rhs
% L(:,[1,end])=[];
% 
% %take care of fluxes over boundaries
% f=[-1;zeros(V-2,1);1];
% L(:,end+1)=f;
% 
% %solve linear system Lx=b, returns values in inner domain;
% result=L\b; 
% 
% %incorporate boundary values in result, remove flux
% result=[b_1 , result(1:end-1)', b_end];
% 
% %plot result
% plot([1:V],result);       
%         
        
        




%  return;

% clear all; close all;
% 
% %using sparse matrices
% V=101; %number of vertices
% E=V-1; %number of edges
% e = ones(E,1);
% A = spdiags([e -e], 0:1, E, V);
% 
% %compute a weight function which drops in the centre of the graph
% sigma=E/10; %weights drop over a region of 2*sigma vertices
% minW=0.2 %minimum weight (similarity)
% w=1-(1-minW)*exp(-0.5*( ([1:E]-E/2).^2 / sigma^2 ));
% 
% %w=ones(1,E); %for isotropic case
% 
% W = spdiags(w', 0, E, E);
% 
% 
% %graph Laplacian   
% L=A'*W*A;
% 
% %set left and right boundary values
% phi_1=1; phi_V=0;
% 
% %remove first and last row from Laplacian
% L([1,end],:)=[];
% 
% %incorporate boundary conditions into b
% 
% %subtract 1st and last column of L from both sides of the equation
% b=zeros(V-2,1)  - L(:,1)*phi_1 - L(:,end)*phi_V;
% L(:,[1,end])=[];
% 
% % solve linear system L*phi=b;
% phi=L\b;
% 
% %append boundary values to obtain full result vector
% phi=[phi_1 , phi', phi_V];
% 
% %plot phi
% plot(phi);       