import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
//import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE;
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


		// Initialize JCusparse library
		// TODO: move outside this class to avoid intitialising it for different instances
		cusparseCreate(handle);

		// Create and set up matrix descriptor
		cusparseCreateMatDescr(descrA);
		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

		// Exercise conversion routines (convert matrix from COO 2 CSR format)
		cudaMalloc(csrRowIndex_gpuPtr, (m+1)*Sizeof.INT); 
		cusparseXcoo2csr(handle, cooRowIndex_gpuPtr, nnz, m,
				csrRowIndex_gpuPtr, CUSPARSE_INDEX_BASE_ZERO);

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
