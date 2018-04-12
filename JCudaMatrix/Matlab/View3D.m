		stack =  load('../data/Result.txt','ascii');
         R=143;
         C=116;
         Z=93;
         
         R=21;
         C=31;
         Z=41;

         stack=reshape(stack,R,C,Z);
         
         % mystery why R and C needs to be swapped
         stack=permute(stack, [2 1 3]);
        
         size(stack)

		 [Xmesh,Ymesh,Zmesh] = meshgrid([1:R],[1:C],[1:Z]);

         
		 xslice = [1,round(R/2),R];   
		 yslice = [];
		 zslice = round(Z/2);
		 slice(Xmesh,Ymesh,Zmesh,stack,xslice,yslice,zslice);
         %imagesc(squeeze(stack(10,:,:)))