import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ONE;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE;
import static jcuda.jcusparse.cusparseSolvePolicy.CUSPARSE_SOLVE_POLICY_NO_LEVEL;
import static jcuda.jcusparse.cusparseSolvePolicy.CUSPARSE_SOLVE_POLICY_USE_LEVEL;
import static jcuda.jcusparse.cusparseFillMode.CUSPARSE_FILL_MODE_LOWER;
import static jcuda.jcusparse.cusparseFillMode.CUSPARSE_FILL_MODE_UPPER;
import static jcuda.jcusparse.cusparseDiagType.CUSPARSE_DIAG_TYPE_UNIT;
import static jcuda.jcusparse.cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT;
import static jcuda.jcusparse.cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST;
import static jcuda.jcusparse.cusparsePointerMode.CUSPARSE_POINTER_MODE_DEVICE;

import static jcuda.jcublas.cublasPointerMode.CUBLAS_POINTER_MODE_DEVICE;


import jcuda.jcublas.*;
import jcuda.jcublas.JCublas2.*;

import static jcuda.jcusparse.cusparseStatus.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import jcuda.*;
import jcuda.jcusparse.*;
import jcuda.runtime.JCuda;



public class csrSparseMatrix {


	// Variable declarations
	private cusparseMatDescr descrA = new cusparseMatDescr();

	private int cooRowIndex_host[];
	private int cooColIndex_host[];
	private float cooVal_host[];

	private Pointer cooRowIndex_gpuPtr = new Pointer();
	private Pointer cooColIndex_gpuPtr = new Pointer();
	private Pointer cooVal_gpuPtr = new Pointer();

	private Pointer csrRowIndex_gpuPtr = new Pointer();
	
	private Pointer iLUcooRowIndex_gpuPtr = new Pointer();
	private Pointer iLUcooColIndex_gpuPtr = new Pointer();
	private Pointer iLUcooVal_gpuPtr = new Pointer();

	private Pointer iLUcsrRowIndex_gpuPtr = new Pointer();
	
	
	

	private int nnz;
	private int m;
	private int n;
	cusparseHandle handle;

	public csrSparseMatrix(cusparseHandle handle, float[][] e, int m, int n){
		//m: rows, n: columns, nnz: number of non zero elements

		this.m=m; this.n=n; this.handle=handle;

		System.out.println("in csrSparseMatrix constructor");

		JCusparse.setExceptionsEnabled(true);
		JCuda.setExceptionsEnabled(true);


		{ //count nnz elements
			nnz=0;
			for(int i=0;i<m;i++)
				for(int j=0;j<n;j++)
					if (e[i][j]!=0)
						++nnz;
		}

		cooRowIndex_host = new int[nnz];
		cooColIndex_host = new int[nnz];
		cooVal_host      = new float[nnz];

		{
			int count=0;
			float v;
			for(int i=0;i<m;i++)
				for(int j=0;j<n;j++)
					if ((v=e[i][j])!=0){
						cooRowIndex_host[count]=i;
						cooColIndex_host[count]=j;
						cooVal_host[count++]=  v;
					}	
		}
;



		// Allocate GPU memory and copy the matrix and vectors into it
		cudaMalloc(cooRowIndex_gpuPtr, nnz*Sizeof.INT);
		cudaMalloc(cooColIndex_gpuPtr, nnz*Sizeof.INT);
		cudaMalloc(cooVal_gpuPtr,      nnz*Sizeof.FLOAT);

		cudaMemcpy(cooRowIndex_gpuPtr, Pointer.to(cooRowIndex_host), nnz*Sizeof.INT,          cudaMemcpyHostToDevice);
		cudaMemcpy(cooColIndex_gpuPtr, Pointer.to(cooColIndex_host), nnz*Sizeof.INT,          cudaMemcpyHostToDevice);
		cudaMemcpy(cooVal_gpuPtr,      Pointer.to(cooVal_host),      nnz*Sizeof.FLOAT,        cudaMemcpyHostToDevice);




		// Create and set up matrix descriptor
		cusparseCreateMatDescr(descrA);
		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

		// Exercise conversion routines (convert matrix from COO 2 CSR format)
		cudaMalloc(csrRowIndex_gpuPtr, (m+1)*Sizeof.INT); 
		cusparseXcoo2csr(handle, cooRowIndex_gpuPtr, nnz, m,
				csrRowIndex_gpuPtr, CUSPARSE_INDEX_BASE_ZERO);

	}

	public float [] LUsolve(denseVector cu_V){
/////////////////////
	/////////////////////
		// Suppose that A is m x m sparse matrix represented by CSR format, 
		// Assumption:
		// - handle is already created by cusparseCreate(),
		// - (d_csrRowPtr, d_csrColInd, d_csrVal) is CSR of A on device memory,
		// - d_x is right hand side vector on device memory,
		// - d_y is solution vector on device memory.
		// - d_z is intermediate result on device memory.
		float result_host[] = new float[m]; //array to hold results
		Pointer d_csrColInd = cooColIndex_gpuPtr;
		Pointer d_csrVal=cooVal_gpuPtr;
		Pointer d_csrRowPtr=csrRowIndex_gpuPtr;
		Pointer  cu_Y = new  Pointer();
		Pointer  cu_Z = new  Pointer();
		cudaMalloc(cu_Y, m*Sizeof.FLOAT);
		cudaMalloc(cu_Z, m*Sizeof.FLOAT);
		
		
		cusparseMatDescr descr_iLU=new cusparseMatDescr();;
		cusparseMatDescr descr_L=new cusparseMatDescr();;
		cusparseMatDescr descr_U=new cusparseMatDescr();;
		csrilu02Info info_iLU =  new csrilu02Info();
		csrsv2Info  info_L = new csrsv2Info();
		csrsv2Info  info_U= new csrsv2Info();
		int[] pBufferSize_iLU=new int[1];
		int[] pBufferSize_L=new int[1];
		int[] pBufferSize_U=new int[1];
		int pBufferSize;
		Pointer pBuffer = new Pointer();

		
		int policy_iLU = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
		 int policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
		 int policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
		int trans_L  = CUSPARSE_OPERATION_NON_TRANSPOSE;
		 int trans_U  = CUSPARSE_OPERATION_NON_TRANSPOSE;

		// step 1: create a descriptor which contains
		// - matrix M is base-1
		// - matrix L is base-1
		// - matrix L is lower triangular
		// - matrix L has unit diagonal 
		// - matrix U is base-1
		// - matrix U is upper triangular
		// - matrix U has non-unit diagonal
		cusparseCreateMatDescr(descr_iLU);
		cusparseSetMatIndexBase(descr_iLU, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatType(descr_iLU, CUSPARSE_MATRIX_TYPE_GENERAL);

		cusparseCreateMatDescr(descr_L);
		cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
		cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

		cusparseCreateMatDescr(descr_U);
		cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
		cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

		// step 2: create a empty info structure
		// we need one info for csrilu02 and two info's for csrsv2
		cusparseCreateCsrilu02Info(info_iLU);
		cusparseCreateCsrsv2Info(info_L);
		cusparseCreateCsrsv2Info(info_U);


		cudaMalloc(iLUcsrRowIndex_gpuPtr, (m+1)*Sizeof.INT); 
		cudaMalloc(iLUcooColIndex_gpuPtr, nnz*Sizeof.INT);
		cudaMalloc(iLUcooVal_gpuPtr,      nnz*Sizeof.FLOAT);
		cudaMemcpy(iLUcsrRowIndex_gpuPtr, d_csrRowPtr, (m+1)*Sizeof.INT,          cudaMemcpyDeviceToDevice);
		cudaMemcpy(iLUcooColIndex_gpuPtr, d_csrColInd, nnz*Sizeof.INT,          cudaMemcpyDeviceToDevice);
		cudaMemcpy(iLUcooVal_gpuPtr,   d_csrVal   ,      nnz*Sizeof.FLOAT,        cudaMemcpyDeviceToDevice);


		
		
		// step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
		cusparseScsrilu02_bufferSize(handle, m, nnz,
		    descr_iLU, d_csrVal, d_csrRowPtr, d_csrColInd, info_iLU, pBufferSize_iLU);
		cusparseScsrsv2_bufferSize(handle, trans_L, m, nnz, 
		    descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L, pBufferSize_L);
		cusparseScsrsv2_bufferSize(handle, trans_U, m, nnz, 
		    descr_U, d_csrVal, d_csrRowPtr, d_csrColInd, info_U, pBufferSize_U);

		pBufferSize = Math.max(pBufferSize_iLU[0], Math.max(pBufferSize_L[0], pBufferSize_U[0]));
		System.out.println("in csrSparseMatrix.LUsolve(),buffersize = "+ pBufferSize);

		// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
		cudaMalloc(pBuffer, pBufferSize);

		// step 4: perform analysis of incomplete Cholesky on M
//		         perform analysis of triangular solve on L
//		         perform analysis of triangular solve on U 
		// The lower(upper) triangular part of M has the same sparsity pattern as L(U), 
		// we can do analysis of csrilu0 and csrsv2 simultaneously.

		
		//JCuda.cudaDeviceSynchronize();   
		
		cusparseScsrilu02_analysis(handle, m, nnz, descr_iLU,
		    d_csrVal, d_csrRowPtr, d_csrColInd, info_iLU, 
		    policy_iLU, pBuffer);
		Pointer structural_zero = new Pointer();
		cudaMalloc(structural_zero, Sizeof.INT);

		int[] mode = new int[1];
		//default mode seems to be host
//		cusparseGetPointerMode(handle,mode);
//		System.out.printf("pointer mode %d \n", mode[0]);
		cusparseSetPointerMode(handle,CUSPARSE_POINTER_MODE_DEVICE);
//		cusparseGetPointerMode(handle,mode);
//		System.out.printf("pointer mode %d \n", mode[0]);
		
		System.out.println("Here TTT ");
		
		
		int status = cusparseXcsrilu02_zeroPivot(handle, info_iLU, structural_zero);
		if (CUSPARSE_STATUS_ZERO_PIVOT == status){
			int [] sz =new int[1];
			cudaMemcpy(Pointer.to(sz), structural_zero, Sizeof.INT, cudaMemcpyDeviceToHost); //copy results back
			System.out.printf("A(%d,%d) is missing\n", sz[0], sz[0]);
		}

		cusparseSetPointerMode(handle,CUSPARSE_POINTER_MODE_HOST);

		System.out.println("Here TTT1 ");
		
		cusparseScsrsv2_analysis(handle, trans_L, m, nnz, descr_L, 
		    d_csrVal, d_csrRowPtr, d_csrColInd,
		    info_L, policy_L, pBuffer);

		cusparseScsrsv2_analysis(handle, trans_U, m, nnz, descr_U, 
		    d_csrVal, d_csrRowPtr, d_csrColInd,
		    info_U, policy_U, pBuffer);

		// step 5: M = L * U
		cusparseScsrilu02(handle, m, nnz, descr_iLU,
				iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, info_iLU, policy_iLU, pBuffer);

		cusparseSetPointerMode(handle,CUSPARSE_POINTER_MODE_DEVICE);		
		Pointer numerical_zero=new Pointer();
		cudaMalloc(numerical_zero, Sizeof.INT);
		
		status = cusparseXcsrilu02_zeroPivot(handle, info_iLU, numerical_zero);
		
		if (CUSPARSE_STATUS_ZERO_PIVOT == status){
			int [] nz =new int[1];
			cudaMemcpy(Pointer.to(nz), numerical_zero, Sizeof.INT, cudaMemcpyDeviceToHost); //copy results back	
		
			System.out.printf("U(%d,%d) is zero\n", nz[0], nz[0]);
		}
		
		cusparseSetPointerMode(handle,CUSPARSE_POINTER_MODE_HOST);
		 
		
		float[] alpha_host={1.f};
	
		
		
		// step 6: solve L*z = x
		cusparseScsrsv2_solve(handle, trans_L, m, nnz, Pointer.to(alpha_host), descr_L,
				iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, info_L,
		   cu_V.getPtr(), cu_Z, policy_L, pBuffer);

		// step 7: solve U*y = z
		cusparseScsrsv2_solve(handle, trans_U, m, nnz, Pointer.to(alpha_host), descr_U,
				iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, info_U,
		   cu_Z, cu_Y, policy_U, pBuffer);
		
//		// step 6: solve L*z = x
//		cusparseScsrsv2_solve(handle, trans_L, m, nnz, Pointer.to(alpha_host), descr_L,
//		   d_csrVal, d_csrRowPtr, d_csrColInd, info_L,
//		   cu_V.getPtr(), cu_Z, policy_L, pBuffer);
//
//		// step 7: solve U*y = z
//		cusparseScsrsv2_solve(handle, trans_U, m, nnz, Pointer.to(alpha_host), descr_U,
//		   d_csrVal, d_csrRowPtr, d_csrColInd, info_U,
//		   cu_Z, cu_Y, policy_U, pBuffer);
		
		
		cudaMemcpy(Pointer.to(result_host), cu_Y, m*Sizeof.FLOAT, cudaMemcpyDeviceToHost); //copy results back

		
//
//		// step 6: free resources
//		cudaFree(pBuffer);
//		cusparseDestroyMatDescr(descr_M);
//		cusparseDestroyMatDescr(descr_L);
//		cusparseDestroyMatDescr(descr_U);
//		cusparseDestroyCsrilu02Info(info_M);
//		cusparseDestroyCsrsv2Info(info_L);
//		cusparseDestroyCsrsv2Info(info_U);
//		cusparseDestroy(handle);
	
		//CG routine
		if (true){
		
///////////////////////////////
		/////////////////////////////
//		/ * **** BiCGStab Code **** * /
//		/ * ASSUMPTIONS :
//		1 . The CUSPARSE and CUBLAS l i b r a r i e s have been i n i t i a l i z e d .
//		2 . The a p p r o p r i a t e memory has been a l l o c a t e d and s e t t o z e r o .
//		3 . The m a t r i x A ( valA , csrRowPtrA , c s r C o l I n d A ) and t h e i n c o m p l e t e −
//		LU l o w e r L ( valL , csrRowPtrL , c s r C o l I n d L ) and upper U ( valU ,
//		csrRowPtrU , c s r C o l I n d U ) t r i a n g u l a r f a c t o r s have been
//		computed and a r e p r e s e n t i n t h e d e v i c e (GPU) memory . * /
//		
	
			
			Pointer r=new Pointer();
			cudaMalloc(r, m*Sizeof.FLOAT);
			
			Pointer t=new Pointer();
			cudaMalloc(t, m*Sizeof.FLOAT);
			
			Pointer ph=new Pointer();
			cudaMalloc(ph, m*Sizeof.FLOAT);
			
			Pointer p=new Pointer();
			cudaMalloc(p, m*Sizeof.FLOAT);
			
			Pointer rw=new Pointer();
			cudaMalloc(rw, m*Sizeof.FLOAT);
			
			Pointer q=new Pointer();
			cudaMalloc(q, m*Sizeof.FLOAT);
			
			Pointer s=new Pointer();
			cudaMalloc(s, m*Sizeof.FLOAT);
			
			float[] zero_host={0.f};	
			float nrmr0;
			float nrmr;
			int maxit = 50;
			float rho=1.f;
			float rhop;
			float beta=0.1f;
			float alpha=1.f;
			float omega=1.f;
			float temp,temp2;	
			int[] ttt=new int[1];
			float[] tttf=new float[1];
			
			//https://www.cfd-online.com/Wiki/Sample_code_for_BiCGSTAB_-_Fortran_90
			
		//create the info and analyse the lower and upper triangular factors
		cusparseSolveAnalysisInfo infoL = new cusparseSolveAnalysisInfo();
		cusparseCreateSolveAnalysisInfo (infoL ) ;
		cusparseSolveAnalysisInfo infoU = new cusparseSolveAnalysisInfo();
		cusparseCreateSolveAnalysisInfo (infoU ) ;
		
		System.out.println("Here I am AAA");
		
		cusparseScsrsv_analysis(handle , CUSPARSE_OPERATION_NON_TRANSPOSE,
		n , nnz, descr_L , iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoL ) ;
		cusparseScsrsv_analysis(handle , CUSPARSE_OPERATION_NON_TRANSPOSE,
		n , nnz, descr_U , iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr , infoU ) ;
		//c u s p a r s e D c s r s v _ a n a l y s i s ( handle , C U S P A R S E _ O P E R A T IO N _ N O N _ T R A N S P O S E ,
		//n , descrU , valU , csrRowPtrU , csrColIndU , infoU ) ;
		
		// 1 : compute i n i t i a l r e s i d u a l r = b − A x0 ( u s i n g i n i t i a l g u e s s i n x )
		//x=cu_Y
		
		cusparseScsrmv ( handle , CUSPARSE_OPERATION_NON_TRANSPOSE , n , n , nnz,Pointer.to(alpha_host),descrA , d_csrVal, d_csrRowPtr, d_csrColInd  , cu_Y , Pointer.to(zero_host) , r ) ;
		
		

		
		
		
		cublasHandle cublashandle = new cublasHandle();
		
		jcuda.jcublas.JCublas2.cublasCreate(cublashandle);

		tttf[0]=-1.0f;
		jcuda.jcublas.JCublas2.cublasSscal (cublashandle, n ,  Pointer.to(tttf), r , 1 ) ;
		//f=cu_V.getPtr()
		tttf[0]=1.0f;
		jcuda.jcublas.JCublas2.cublasSaxpy (cublashandle, n , Pointer.to(tttf) , cu_V.getPtr(), 1 , r , 1 ) ;
		
//		jcuda.jcublas.JCublas.cublasSscal ( n , -1.0f , r , 1 ) ;
//		//f=cu_V.getPtr()
//		jcuda.jcublas.JCublas.cublasSaxpy ( n , 1.0f , cu_V.getPtr(), 1 , r , 1 ) ;
//		// 2 : S e t p=r and \ t i l d e { r}=r

		


		
		jcuda.jcublas.JCublas2.cublasScopy (cublashandle, n , r , 1 , p , 1 ) ;
		jcuda.jcublas.JCublas2.cublasScopy ( cublashandle,n , r , 1 , rw , 1 ) ;
		jcuda.jcublas.JCublas2.cublasSnrm2 (cublashandle, n , r , 1 ,Pointer.to(tttf)) ;
		nrmr0=tttf[0];
		// 3 : r e p e a t u n t i l c o n v e r g e n c e ( based on max . i t . and r e l a t i v e r e s i d u a l )


		
		
		for ( int i =0; i<maxit ; i++){
		// 4 : \ rho = \ t i l d e { r }ˆ{T} r
		rhop= rho ;
		
		
		
		jcuda.jcublas.JCublas2.cublasSdot (cublashandle, n , rw , 1 , r , 1, Pointer.to(ttt) ) ;
		rho=ttt[0];
		//rho = jcuda.jcublas.JCublas.cublasSdot ( n , rw , 1 , r , 1 ) ;
		

		
		
		
		
		
		if ( i > 0) {
		// 1 2 : \ b e t a = ( \ r h o { i } / \ r h o { i −1}) ( \ a l p h a / \omega )
		beta= ( rho / rhop ) * ( alpha / omega ) ;
		// 1 3 : p = r + \ b e t a ( p − \omega v )
		
		tttf[0]=-omega;
		jcuda.jcublas.JCublas2.cublasSaxpy (cublashandle, n ,Pointer.to(tttf) , q , 1 , p , 1 ) ;
		tttf[0]=beta;
		jcuda.jcublas.JCublas2.cublasSscal ( cublashandle,n , Pointer.to(tttf) , p , 1 ) ;
		tttf[0]=1.0f;
		jcuda.jcublas.JCublas2.cublasSaxpy ( cublashandle,n , Pointer.to(tttf) , r , 1 , p , 1 ) ;
		}
		

		
		
//		// 1 5 : M \ hat {p} = p ( s p a r s e l o w e r and upper t r i a n g u l a r s o l v e s )
		cusparseScsrsv_solve( handle , CUSPARSE_OPERATION_NON_TRANSPOSE ,
		n , Pointer.to(alpha_host) , descr_L ,iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr,
		infoL , p , t ) ;
		

		
		
		cusparseScsrsv_solve ( handle , CUSPARSE_OPERATION_NON_TRANSPOSE ,
		n , Pointer.to(alpha_host) , descr_U , iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr,
		infoU , t , ph ) ;
		

		
		
//		// 1 6 : q = A \ hat {p} ( s p a r s e matrix−v e c t o r m u l t i p l i c a t i o n )
		cusparseScsrmv ( handle , CUSPARSE_OPERATION_NON_TRANSPOSE , n , n ,nnz, Pointer.to(alpha_host) ,
		descrA , d_csrVal, d_csrRowPtr, d_csrColInd, ph , Pointer.to(zero_host) , q ) ;

		
		//		// 1 7 : \ a l p h a = \ r h o { i } / ( \ t i l d e { r }ˆ{T} q )

		jcuda.jcublas.JCublas2.cublasSdot (cublashandle, n , rw , 1 , q , 1, Pointer.to(ttt) ) ;
temp=ttt[0];		
		//temp = jcuda.jcublas.JCublas.cublasSdot ( n , rw , 1 , q , 1 ) ;
		alpha= rho / temp ;
//		// 1 8 : s = r − \ a l p h a q		
//		
		tttf[0]=-alpha;
		jcuda.jcublas.JCublas2.cublasSaxpy ( cublashandle,n ,Pointer.to(tttf), q , 1 , r , 1 ) ;
//		// 1 9 : x = x + \ a l p h a \ hat {p}
		tttf[0]=alpha;
		jcuda.jcublas.JCublas2.cublasSaxpy (cublashandle, n , Pointer.to(tttf) , ph , 1 , cu_Y , 1 ) ;
//		// 2 0 : c h e c k f o r c o n v e r g e n c e
		 jcuda.jcublas.JCublas2.cublasSnrm2 ( cublashandle,n , r , 1 , Pointer.to(tttf) ) ;
		 nrmr=tttf[0];
		 
		 System.out.println("Here nrmr first time "  +nrmr);
		 
		 
		 
		float tol=1e-3f;

		if ( nrmr / nrmr0 < tol ) {
		break ;
		}
//		// 2 3 : M \ hat { s } = r ( s p a r s e l o w e r and upper t r i a n g u l a r s o l v e s )
		cusparseScsrsv_solve( handle , CUSPARSE_OPERATION_NON_TRANSPOSE ,
		n , Pointer.to(alpha_host) , descr_L , iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr,
		infoL , r , t ) ;
	


		
		cusparseScsrsv_solve( handle , CUSPARSE_OPERATION_NON_TRANSPOSE ,
		n , Pointer.to(alpha_host) , descr_U, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr,
		infoU , t , s ) ;
		
		


//		// 2 4 : t = A \ hat { s } ( s p a r s e matrix−v e c t o r m u l t i p l i c a t i o n )

		cusparseScsrmv ( handle , CUSPARSE_OPERATION_NON_TRANSPOSE , n , n ,nnz, Pointer.to(alpha_host) ,
		descrA , d_csrVal, d_csrRowPtr, d_csrColInd, s , Pointer.to(zero_host) , t ) ;
		
		
//		// 2 5 : \omega = ( t ˆ{T} s ) / ( t ˆ{T} t )
		
		//JCuda.cudaDeviceSynchronize(); 
		
		//jcuda.jcublas.JCublas2.cublasSetPointerMode(cublashandle,CUBLAS_POINTER_MODE_DEVICE);
		

		jcuda.jcublas.JCublas2.cublasSdot (cublashandle, n , t , 1 , r , 1, Pointer.to(ttt) ) ;
temp=ttt[0];
		jcuda.jcublas.JCublas2.cublasSdot (cublashandle, n , t , 1 , t , 1, Pointer.to(ttt) ) ;

		temp2=ttt[0];
		
		//temp = jcuda.jcublas.JCublas.cublasSdot ( n , t , 1 , r , 1 ) ;
		// temp2= jcuda.jcublas.JCublas.cublasSdot ( n , t , 1 , t , 1 ) ;
		omega= temp / temp2 ;
//		// 2 6 : x = x + \omega \ hat { s }
		
		tttf[0]=omega;
		jcuda.jcublas.JCublas2.cublasSaxpy (cublashandle, n , Pointer.to(tttf) , s , 1 , cu_Y , 1 ) ;
		
		
		////!!!!!!!!  cublasSdot to compute temp and temp 2 causes problems
		System.out.println("Here temp "  +temp + " temp2 "+temp2);
//		cudaMemcpy(Pointer.to(result_host), t, 100*Sizeof.FLOAT, cudaMemcpyDeviceToHost); //copy results back
//		for(int ii=0;ii<100;ii++)
//			System.out.println("Here t "+ii +"  "+result_host[ii]);
		
		
//		// 2 7 : r = s − \omega t
		tttf[0]=-omega;
		jcuda.jcublas.JCublas2.cublasSaxpy ( cublashandle,n ,Pointer.to(tttf) , t , 1 , r , 1 ) ;
//		// c h e c k f o r c o n v e r g e n c e
		
		jcuda.jcublas.JCublas2.cublasSnrm2 (cublashandle ,n , r , 1, Pointer.to(tttf)) ;
		nrmr=tttf[0];
		System.out.println("Here nrmr second time "  +nrmr);
		cudaMemcpy(Pointer.to(result_host), r, 100*Sizeof.FLOAT, cudaMemcpyDeviceToHost); //copy results back
		for(int ii=0;ii<100;ii++)
			System.out.println("Here r  "+ii +"  "+result_host[ii]);
		
		if ( nrmr / nrmr0 < tol ) {
		break ;
		}
	
		 System.out.println("Here0 nrmr: "+nrmr +" nrmr0: "+nrmr0+ " alpha: "+alpha+ " beta: "+beta+ " rho: "+rho+ " temp: "+temp2+ " temp: "+temp2+ " omega: "+omega);

		//System.out.println("Here nrmr "+nrmr +" nrmr0 "+nrmr0);
		
		
		
		}
		
//		// d e s t r o y t h e a n a l y s i s i n f o ( f o r l o w e r and upper t r i a n g u l a r f a c t o r s )
//		c u s p a r s e D e s t r o y S o l v e A n a l y s i s I n f o ( infoL ) ;
//		c u s p a r s e D e s t r o y S o l v e A n a l y s i s I n f o ( infoU ) ;
//		
//		
		//jcuda.JCublas.cusparseDestroy(cublashandle);
		/////////////////////////////
		//////////////////////////////
		

		
		
		
		} //CG routine
		///needs changing
		
		cudaMemcpy(Pointer.to(result_host), cu_Y, m*Sizeof.FLOAT, cudaMemcpyDeviceToHost); //copy results back

		return result_host;
	///////////////////
	
	///////////////////
}
	
	
	public float[] mldivide(denseVector  cu_V){

		//print_coo();

		float result_host[] = new float[n]; //array to hold results
		Pointer result_gpuPtr = new Pointer();
		cudaMalloc(result_gpuPtr, n*Sizeof.FLOAT); //allocate memory for results

		float[] alpha_host={1.f};


		cusparseSolveAnalysisInfo info = new cusparseSolveAnalysisInfo();
		cusparseCreateSolveAnalysisInfo(info);

		int cusparseStatus;

		cusparseStatus = cusparseScsrsm_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nnz, descrA, cooVal_gpuPtr, csrRowIndex_gpuPtr, cooColIndex_gpuPtr, info); //perform analysis phase of solving

		if(cusparseStatus != CUSPARSE_STATUS_SUCCESS)
			System.out.println("in csrSparseMatrix.mldivide(), problem with cusparseDcsrsm_analysis, cusparseStatus_t = "+ cusparseStatus);

		cusparseStatus = cusparseScsrsm_solve( handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, 1, Pointer.to(alpha_host), descrA, cooVal_gpuPtr, csrRowIndex_gpuPtr, cooColIndex_gpuPtr, info, cu_V.getPtr(), n, result_gpuPtr, n);

		if(cusparseStatus != CUSPARSE_STATUS_SUCCESS)
			System.out.println("in csrSparseMatrix.mldivide(), problem with cusparseScsrsm_solve, cusparseStatus_t = "+ cusparseStatus);


		cudaMemcpy(Pointer.to(result_host), result_gpuPtr, n*Sizeof.FLOAT, cudaMemcpyDeviceToHost); //copy results back


		JCuda.cudaDeviceSynchronize();   

		return result_host;
	}	

	public void free(){
		cudaFree(csrRowIndex_gpuPtr);
		cudaFree(cooRowIndex_gpuPtr);
		cudaFree(cooColIndex_gpuPtr);
		cudaFree(cooVal_gpuPtr);  
	}
	
	public void print_coo(){

		System.out.printf(" csrSparseMatrix in COO format:\n");
		for (int i=0; i<nnz; i++)
		{
			System.out.printf("cooRowInded_host[%d]=%d  ",i,cooRowIndex_host[i]);
			System.out.printf("cooColInded_host[%d]=%d  ",i,cooColIndex_host[i]);
			System.out.printf("cooVal_host[%d]=%f     \n",i,cooVal_host[i]);
		}



	}

	
	
	


}
