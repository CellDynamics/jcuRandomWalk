clear all; close all;


%Linear graph with 6 nodes, 5 edges

%incidence matrix A of size edges*vertices

A = [  +1  -1   0   0   0  0
            0 +1 -1   0   0  0
            0   0 +1 -1   0  0
            0   0   0 +1 -1  0
            0   0   0   0 +1 -1  ];
        
        
[E,V] = size(A); % get number of edges,E, and vertices,V
        
  
%weight matrix with equal weights        
W = [    1   0   0   0   0
            0   1   0   0   0
            0   0   1   0   0
            0   0   0   1   0
            0   0   0   0   1  ];
 
 %weight matrix with unequal weights, individual weights are a measure for similarity             
 W1 = [ 1   0   0   0   0
            0  .9   0   0   0
            0   0  .2   0   0
            0   0   0  .9   0
            0   0   0   0   1  ];       
         
%graph Laplacian   
L=A'*W*A;


%set boundary conditions
%clamp first vertex to 1 (probability to be foreground), last vertex to zero
b_1=1; b_end=0;

%create b, right hand side of Lx=b 
b = zeros(V,1) - L(:,1)*b_1 - L(:,end)*b_end;

%remove first and last column from Laplacian as they have been shifted to
%the rhs
L(:,[1,end])=[];

%solve linear system Lx=b, returns values in inner domain;
result=L\b; 

%incorporate boundary values in result
result=[b_1 , result', b_end];

%plot result
plot(1:V,result);





return;

clear all; close all;

%using sparse matrices
V=101; %number of vertices
E=V-1; %number of edges
sigma=E/10; %weights drop over a region of 2*sigma vertices
minW=0.2
e = ones(E,1);
A = spdiags([e -e], 0:1, E, V);
e = ones(E,1);
w=1-(1-minW)*exp(-0.5*( ([1:E]-E/2).^2 / sigma^2 ));
%w=ones(1,E); %check for isotropic case
W = spdiags(w', 0, E, E);

    
L=A'*W*A;
%clamp first vertex to 1, last vertex to zero
b1=1;
b2=0;
%remove first and last row from Laplacian (boundary points), as we are only solving in the
%inner region
L([1,end],:)=[];
L(1,1)=0;
L(end,end)=0;
b=zeros(V-2,1);
b(1)=1; %boundary conditions
result=L\b; % solve linear system Lx=b;
result(1)=b1;
result(end)=b2;
plot(1:V,result);

