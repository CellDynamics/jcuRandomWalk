		stack =  load('../data/Result.txt','ascii')
         R=25;
         C=25;
         Z=25;
         stack=reshape(stack,R,C,Z);
		 [X,Y,Z] = meshgrid([1:R],[1:C],[1:Z]);
		 xslice = [1,R/2,R];   
		 yslice = [];
		 zslice = R/2;
		 slice(X,Y,Z,stack,xslice,yslice,zslice)