		stack =  load('../data/Result.txt','ascii');
         R=143;
         C=116;
         Z=93;
%                   R=21;
%          C=31;
%          Z=41;
         %attention, we swapped original R and C
         stack=reshape(stack,C,R,Z);

		 [X,Y,Z] = meshgrid([1:R],[1:C],[1:Z]);
		 xslice = [1,round(R/2),R];   
		 yslice = [];
		 zslice = round(R/2);
		 slice(X,Y,Z,stack,xslice,yslice,zslice);
         %imagesc(squeeze(stack(10,:,:)))