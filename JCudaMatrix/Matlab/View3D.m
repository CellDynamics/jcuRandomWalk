		 load('../data/Result.txt','ascii')
         R=12
         C=12
         Z=12;
		 [X,Y,Z] = meshgrid([1:R],[1:C],[1:Z]);
		 xslice = [1,6,12];   
		 yslice = [];
		 zslice = 6;
		 slice(X,Y,Z,stack,xslice,yslice,zslice)