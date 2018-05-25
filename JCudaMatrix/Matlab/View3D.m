close all;		 clear all;
stack =  load('../data/Result.txt','ascii');
R=143;
C=116;
Z=93;

%          R=21;
%          C=31;
%          Z=41;

stack=reshape(stack,R,C,Z);
stackSink=stack;
stackSink(stackSink==0)=-1;
stackSink(stackSink~=-1)=0;
stackSink(stackSink==-1)=1;
 save '../data/stackSink.mat' stackSink;
%  
% stack(stack>0.75)=1;
% stack(stack<=0.75)=0;

% mystery why R and C needs to be swapped
stack=permute(stack, [2 1 3]);
% save 'RWsegmentedPgt075.mat'  stack;
% size(stack)

outputFileName = 'img_segmented.tif'
for K=1:length(stack(1, 1, :))
   imwrite(stack(:, :, K), outputFileName, 'WriteMode', 'append',  'Compression','none');
end

[Xmesh,Ymesh,Zmesh] = meshgrid([1:R],[1:C],[1:Z]);


xslice = [round(R/2)];
yslice = [];
zslice = round(Z/2);
figure(88)
h = slice(Xmesh,Ymesh,Zmesh,stack,xslice,yslice,zslice);
daspect([1 1 1])   ;
for i=1:2
    h(i).EdgeColor = 'none';
end
%imagesc(squeeze(stack(10,:,:)))


limits = [NaN NaN NaN NaN NaN NaN];
[x, y, z, stack] = subvolume(stack, limits);           % extract a subset of the volume data

[fo,vo] = isosurface(x,y,z,stack,.75);               % isosurface for the outside of the volume
[fe,ve,ce] = isocaps(x,y,z,stack,.75);               % isocaps for the end caps of the volume

figure (99)
p1 = patch('Faces', fo, 'Vertices', vo);       % draw the outside of the volume
p1.FaceColor = 'green';
p1.EdgeColor = 'none';

p2 = patch('Faces', fe, 'Vertices', ve, ...    % draw the end caps of the volume
    'FaceVertexCData', ce);
p2.FaceColor = 'interp';
p2.EdgeColor = 'none';

%view(-40,24)
daspect([1 1 1])                             % set the axes aspect ratio
colormap(gray(100))
box on

camlight(40,40)                                % create two lights
%camlight(-20,-10)
camlight(0,-90)
lighting gouraud

%%%%%%%%%%%%%%%%%%%%

stack =  load('../data/LifeActGFP_3D.mat');

stack=stack.LifeActGFP_3D;
stack=(stack-min(stack(:)));
stack=255.*stack./max(stack(:));

stack=permute(stack, [2 1 3]);

% outputFileName = 'img_original.tif'
% for K=1:length(stack(1, 1, :))
%    imwrite(stack(:, :, K), outputFileName, 'WriteMode', 'append',  'Compression','none');
% end

size(stack)

figure(77)
h1 = slice(Xmesh,Ymesh,Zmesh,stack,xslice,yslice,zslice);
daspect([1 1 1])   ;
for i=1:2
    h1(i).EdgeColor = 'none';
end


%%%%%%%%%%%%%%%%%%%%%%%%
% 
% hlink = linkprop([h1(1),h1(2),p1,p2,h(1),h(2)],{'CameraPosition','CameraUpVector'}); 
% rotate3d on