package storage;

import utils.Constants.matrixFormat;
import utils.Operations;
import utils.Conversions;
import jcuda.*;
import jcuda.jcusparse.*;
import jcuda.runtime.JCuda;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import static jcuda.runtime.cudaMemcpyKind.*;
import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.*;
import static jcuda.runtime.JCuda.*;
import storage.DenseMatrix;
import storage.DenseMatrixGPU;
import storage.DenseVector;
import storage.DenseVectorGPU;
import storage.SparseMatrix;
import storage.SparseMatrixGPU;
import utils.Constants.matrixFormat;
import static jcuda.jcusparse.cusparseStatus.*;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparseSolvePolicy.CUSPARSE_SOLVE_POLICY_NO_LEVEL;
import static jcuda.jcusparse.cusparseSolvePolicy.CUSPARSE_SOLVE_POLICY_USE_LEVEL;
import static jcuda.jcusparse.cusparseFillMode.CUSPARSE_FILL_MODE_LOWER;
import static jcuda.jcusparse.cusparseFillMode.CUSPARSE_FILL_MODE_UPPER;
import static jcuda.jcusparse.cusparseDiagType.CUSPARSE_DIAG_TYPE_UNIT;
import static jcuda.jcusparse.cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT;
import static jcuda.jcusparse.cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST;
import static jcuda.jcusparse.cusparsePointerMode.CUSPARSE_POINTER_MODE_DEVICE;

import jcuda.jcublas.*;
import static jcuda.jcublas.JCublas2.*;

public class GPURW {
	public static void main(String[] args) {
		
		int[] CRowIndCSR = {0,1,2,3,4,5};
		int[] CColIndCSR = {0,1,2,3,4};
		double[] CValCSR = {1,1,1,1,1};
		
		int[] ARowIndCSR = {0,2,4,6,8,10};
		int[] AColIndCSR = {0,1,1,2,2,3,3,4,4,5};
		double[] AValCSR = {1,-1,1,-1,1,-1,1,-1,1,-1};
		
		int[] A_tRowIndCSR = {0,1,3,5,7,9,10};
		int[] A_tColIndCSR = {0,0,1,1,2,2,3,3,4,4};
		double[] A_tValCSR = {1,-1,1,-1,1,-1,1,-1,1,-1};
		
		SparseMatrixGPU C = new SparseMatrixGPU(CRowIndCSR, CColIndCSR, CValCSR, 5, 5, 5, matrixFormat.MATRIX_FORMAT_CSR);
		System.out.println("C");
		System.out.println(C.toString());
		SparseMatrixGPU A = new SparseMatrixGPU(ARowIndCSR, AColIndCSR, AValCSR, 5, 6, 10, matrixFormat.MATRIX_FORMAT_CSR);
		System.out.println("A");
		System.out.println(A.toString());
		SparseMatrixGPU A_t = new SparseMatrixGPU(A_tRowIndCSR, A_tColIndCSR, A_tValCSR, 6, 5, 10, matrixFormat.MATRIX_FORMAT_CSR);
		System.out.println("A_t");
		System.out.println(A_t.toString());
		
		cusparseHandle handle = new cusparseHandle();
		cusparseCreate(handle);
		
		SparseMatrixGPU A_tC = Operations.Multiply(handle, A_t, C);
		System.out.println("A_tC");
		System.out.println(A_tC.toString());
		SparseMatrixGPU Lap = Operations.Multiply(handle, A_tC, A);
		System.out.println("Lap");
		System.out.println(Lap.toString());
		
		
		
	}
}
