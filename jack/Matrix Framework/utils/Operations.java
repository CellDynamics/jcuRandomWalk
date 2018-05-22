package utils;

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

public class Operations {
	//TODO
	public static DenseMatrix TransposeDense(DenseMatrix m) {
		double data[][] = m.GetData();
		double dataTranspose[][] = new double[m.GetM()][m.GetN()];
		for (int i = 0; i < m.GetM(); i++) {
			for (int j = 0; j < m.GetN(); j++) {
				dataTranspose[i][j] = data[j][i];
			}
		}
		return new DenseMatrix(dataTranspose);
	}
	
	public static DenseMatrixGPU TransposeDenseGPU(DenseMatrixGPU m) {
		return Conversions.DMtoDMGPU(TransposeDense(Conversions.DMGPUtoDM(m)));
	}
	
	public static SparseMatrix TransposeSparse(SparseMatrix m) {
		return Conversions.DMtoSM(TransposeDense(Conversions.SMtoDM(m)));
	}
	
	public static SparseMatrixGPU TransposeSparseGPU(SparseMatrixGPU m) {
		return Conversions.DMtoSMGPU(TransposeDense(Conversions.SMGPUtoDM(m)));
	}
	
	public static SparseMatrixGPU Multiply(cusparseHandle handle, SparseMatrixGPU m1, SparseMatrixGPU m2) {
		
		if (m1.GetN() != m2.GetM()) {
			System.out.println("DIMENSIONS DON'T MATCH!");
			return null;
		}
		
		int m,n,k;
		m = m1.GetM();
		n = m2.GetN();
		k = m1.GetN();
		int[] nnzOut = new int[1];
		int n_t = CUSPARSE_OPERATION_NON_TRANSPOSE;
		
		cusparseMatDescr descrOut = new cusparseMatDescr();
		cusparseCreateMatDescr(descrOut);
		cusparseSetMatType(descrOut, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrOut, CUSPARSE_INDEX_BASE_ZERO);
		
		Pointer nnzOutPtr = new Pointer();
		Pointer rowIndOutPtr = new Pointer();
		Pointer colIndOutPtr = new Pointer();
		Pointer valOutPtr = new Pointer();
		
		cudaMalloc(nnzOutPtr, Sizeof.INT);
		cudaMalloc(rowIndOutPtr, (m + 1)*Sizeof.INT);
		
		cusparseXcsrgemmNnz(handle, n_t, n_t, m, n, k, m1.GetDescr(), m1.GetNnz(), m1.GetRowIndPtr(), m1.GetColIndPtr(), m2.GetDescr(), m2.GetNnz(), m2.GetRowIndPtr(), m2.GetColIndPtr(), descrOut, rowIndOutPtr, nnzOutPtr);
		JCuda.cudaDeviceSynchronize();
		
		cudaMemcpy(Pointer.to(nnzOut), nnzOutPtr, Sizeof.INT, cudaMemcpyDeviceToHost);
		
		cudaMalloc(colIndOutPtr, nnzOut[0]*Sizeof.INT);
		cudaMalloc(valOutPtr, nnzOut[0]*Sizeof.DOUBLE);
		
		cusparseDcsrgemm(handle, n_t, n_t, m, n, k, m1.GetDescr(), m1.GetNnz(), m1.GetValPtr(), m1.GetRowIndPtr(), m1.GetColIndPtr(), m2.GetDescr(), m2.GetNnz(), m2.GetValPtr(), m2.GetRowIndPtr(), m2.GetColIndPtr(), descrOut, valOutPtr, rowIndOutPtr, colIndOutPtr);
		JCuda.cudaDeviceSynchronize();		
		
		return new SparseMatrixGPU(descrOut, rowIndOutPtr, colIndOutPtr, valOutPtr, m, n, nnzOut[0], matrixFormat.MATRIX_FORMAT_CSR);
	}
	
	public double[] LuSolve(cusparseHandle handle, DenseVectorGPU b_gpuPtr, SparseMatrixGPU X) {
		return LuSolve(handle, b_gpuPtr, X, false);
	}

	public double[] LuSolve(cusparseHandle handle, DenseVectorGPU b_gpuPtr, SparseMatrixGPU X, boolean iLuBiCGStabSolve) {

		// some useful constants
		double[] one_host = { 1.f };
		double[] zero_host = { 0.f };
		double[] minus_one_host = { -1.f };

		// iLU part adapted from nvidia cusparse documentation

		// Suppose that A is m x m sparse matrix represented by CSR format,
		// Assumption
		// - handle is already created by cusparseCreate(),
		// - (AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, AcooVal_gpuPtr) is CSR
		// of A on device memory,
		// - b_gpuPtr is right hand side vector on device memory,
		// - x_gpuPtr is solution vector on device memory.
		// - z_gpuPtr is intermediate result on device memory.

		// setup solution vector and intermediate vector
		Pointer x_gpuPtr = new Pointer();
		Pointer z_gpuPtr = new Pointer();
		cudaMalloc(x_gpuPtr, X.GetM() * Sizeof.DOUBLE);
		cudaMalloc(z_gpuPtr, X.GetM() * Sizeof.DOUBLE);

		// setting up pointers for the sparse iLU matrix, which contains L and U
		// Nvidia's original example overwrites matrix A, which is not ideal
		// when later using iLuBiCGStabSolve
		Pointer iLUcooColIndex_gpuPtr = new Pointer();
		Pointer iLUcooVal_gpuPtr = new Pointer();
		Pointer iLUcsrRowIndex_gpuPtr = new Pointer();

		// step 1: create descriptors/policies/operation modi for iLU, L, and U
		cusparseMatDescr descr_iLU = new cusparseMatDescr();
		cusparseCreateMatDescr(descr_iLU);
		cusparseSetMatIndexBase(descr_iLU, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatType(descr_iLU, CUSPARSE_MATRIX_TYPE_GENERAL);
		int policy_iLU = CUSPARSE_SOLVE_POLICY_NO_LEVEL;

		cusparseMatDescr descr_L = new cusparseMatDescr();
		;
		cusparseCreateMatDescr(descr_L);
		cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
		cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);
		int policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
		int trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;

		cusparseMatDescr descr_U = new cusparseMatDescr();
		;
		cusparseCreateMatDescr(descr_U);
		cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
		cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);
		int policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
		int trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

		// step 2: create a empty info structure
		// we need one info for csrilu02 and two info's for csrsv2
		csrilu02Info info_iLU = new csrilu02Info();
		csrsv2Info info_L = new csrsv2Info();
		csrsv2Info info_U = new csrsv2Info();
		cusparseCreateCsrilu02Info(info_iLU);
		cusparseCreateCsrsv2Info(info_L);
		cusparseCreateCsrsv2Info(info_U);

		// copy matrix A into iLU
		cudaMalloc(iLUcsrRowIndex_gpuPtr, (X.GetM() + 1) * Sizeof.INT);
		cudaMalloc(iLUcooColIndex_gpuPtr, X.GetNnz() * Sizeof.INT);
		cudaMalloc(iLUcooVal_gpuPtr, X.GetNnz() * Sizeof.DOUBLE);
		cudaMemcpy(iLUcsrRowIndex_gpuPtr, X.GetRowIndPtr(), (X.GetM() + 1)
				* Sizeof.INT, cudaMemcpyDeviceToDevice);
		cudaMemcpy(iLUcooColIndex_gpuPtr, X.GetColIndPtr(),
				X.GetNnz() * Sizeof.INT, cudaMemcpyDeviceToDevice);
		cudaMemcpy(iLUcooVal_gpuPtr, X.GetValPtr(), X.GetNnz() * Sizeof.DOUBLE,
				cudaMemcpyDeviceToDevice);

		// set up buffer
		int[] pBufferSize_iLU = new int[1];
		int[] pBufferSize_L = new int[1];
		int[] pBufferSize_U = new int[1];
		int pBufferSize;
		Pointer pBuffer = new Pointer();

		// step 3: query how much memory used in csrilu02 and csrsv2, and
		// allocate the buffer
		cusparseScsrilu02_bufferSize(handle, X.GetM(), X.GetNnz(), descr_iLU, X.GetValPtr(),
				X.GetRowIndPtr(), X.GetColIndPtr(), info_iLU,
				pBufferSize_iLU);
		cusparseScsrsv2_bufferSize(handle, trans_L, X.GetM(), X.GetNnz(), descr_L,
				X.GetValPtr(), X.GetRowIndPtr(), X.GetColIndPtr(),
				info_L, pBufferSize_L);
		cusparseScsrsv2_bufferSize(handle, trans_U, X.GetM(), X.GetNnz(), descr_U,
				X.GetValPtr(), X.GetRowIndPtr(), X.GetColIndPtr(),
				info_U, pBufferSize_U);

		pBufferSize = Math.max(pBufferSize_iLU[0],
				Math.max(pBufferSize_L[0], pBufferSize_U[0]));
		// System.out.println("in csrSparseMatrix.LuSolve(),buffersize = "+
		// pBufferSize);

		// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
		cudaMalloc(pBuffer, pBufferSize);

		// step 4: perform analysis of incomplete Cholesky on M
		// perform analysis of triangular solve on L
		// perform analysis of triangular solve on U
		// The lower(upper) triangular part of M has the same sparsity pattern
		// as L(U),
		// we can do analysis of csrilu0 and csrsv2 simultaneously.

		cusparseScsrilu02_analysis(handle, X.GetM(), X.GetNnz(), descr_iLU, X.GetValPtr(),
				X.GetRowIndPtr(), X.GetColIndPtr(), info_iLU, policy_iLU,
				pBuffer);

		Pointer structural_zero = new Pointer();
		cudaMalloc(structural_zero, Sizeof.INT);

		// int[] cusparsePointerMode = new int[1];
		// default mode seems to be HOST
		// cusparseGetPointerMode(handle, cusparsePointerMode);
		// System.out.printf("Cusparse pointer mode %d \n",
		// cusparsePointerMode[0]);
		// we need to switch to DEVICE before using cusparseXcsrilu02_zeroPivot,
		// for obscure reasons, and switch back to HOST afterwards
		cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
		if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseXcsrilu02_zeroPivot(handle,
				info_iLU, structural_zero)) {
			int[] sz = new int[1];
			cudaMemcpy(Pointer.to(sz), structural_zero, Sizeof.INT,
					cudaMemcpyDeviceToHost); // copy results back
			System.out.printf("A(%d,%d) is missing\n", sz[0], sz[0]);
		}
		cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

		cusparseScsrsv2_analysis(handle, trans_L, X.GetM(), X.GetNnz(), descr_L,
				X.GetValPtr(), X.GetRowIndPtr(), X.GetColIndPtr(),
				info_L, policy_L, pBuffer);

		cusparseScsrsv2_analysis(handle, trans_U, X.GetM(), X.GetNnz(), descr_U,
				X.GetValPtr(), X.GetRowIndPtr(), X.GetColIndPtr(),
				info_U, policy_U, pBuffer);

		// step 5: M = L * U
		cusparseScsrilu02(handle, X.GetM(), X.GetNnz(), descr_iLU, iLUcooVal_gpuPtr,
				iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, info_iLU,
				policy_iLU, pBuffer);

		Pointer numerical_zero = new Pointer();
		cudaMalloc(numerical_zero, Sizeof.INT);

		// same trick of switching modes needed here
		cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
		if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseXcsrilu02_zeroPivot(handle,
				info_iLU, numerical_zero)) {
			int[] nz = new int[1];
			cudaMemcpy(Pointer.to(nz), numerical_zero, Sizeof.INT,
					cudaMemcpyDeviceToHost); // copy results back
			System.out.printf("U(%d,%d) is zero\n", nz[0], nz[0]);
		}
		cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

		// step 6: solve L*z = x
		cusparseScsrsv2_solve(handle, trans_L, X.GetM(), X.GetNnz(), Pointer.to(one_host),
				descr_L, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
				iLUcooColIndex_gpuPtr, info_L, b_gpuPtr.GetPtr(), z_gpuPtr,
				policy_L, pBuffer);

		// step 7: solve U*y = z
		cusparseScsrsv2_solve(handle, trans_U, X.GetM(), X.GetNnz(), Pointer.to(one_host),
				descr_U, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
				iLUcooColIndex_gpuPtr, info_U, z_gpuPtr, x_gpuPtr, policy_U,
				pBuffer);

		// CG routine
		if (iLuBiCGStabSolve) {

			// see paper by Nvidia using cublas1
			// http://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html
			// this was also useful:
			// //https://www.cfd-online.com/Wiki/Sample_code_for_BiCGSTAB_-_Fortran_90

			// we make extensive use of JCublas2
			cublasHandle cublashandle = new cublasHandle();
			jcuda.jcublas.JCublas2.cublasCreate(cublashandle);

			// /*****BiCGStabCode*****/
			// /*ASSUMPTIONS:
			// 1.The CUSPARSE and CUBLAS libraries have been initialized.
			// 2.The appropriate memory has been allocated and set to zero.
			// 3.The matrixA (valA, csrRowPtrA, csrColIndA) and the incomplete−
			// LUlowerL (valL, csrRowPtrL, csrColIndL) and upperU (valU,
			// csrRowPtrU, csrColIndU) triangular factors have been
			// computed and are present in the device (GPU) memory.*/
			//

			// the above requirements are met

			// we create a number of pointers according to the method, and
			// subsequently allocate memory
			// TODO: rename these according to _gpuPtr scheme
			Pointer p = new Pointer();
			Pointer ph = new Pointer();
			Pointer q = new Pointer();
			Pointer r = new Pointer();
			Pointer rw = new Pointer();
			Pointer s = new Pointer();
			Pointer t = new Pointer();

			cudaMalloc(p, X.GetM() * Sizeof.DOUBLE);
			cudaMalloc(ph, X.GetM() * Sizeof.DOUBLE);
			cudaMalloc(q, X.GetM() * Sizeof.DOUBLE);
			cudaMalloc(r, X.GetM() * Sizeof.DOUBLE);
			cudaMalloc(rw, X.GetM() * Sizeof.DOUBLE);
			cudaMalloc(s, X.GetM() * Sizeof.DOUBLE);
			cudaMalloc(t, X.GetM() * Sizeof.DOUBLE);

			// BiCGStab parameters (all on host)
			double[] nrmr0 = new double[1];
			double[] nrmr = new double[1];

			double[] rho = { 1.f };
			double[] rhop = new double[1];
			double[] alpha = { 1.f };
			double[] beta = { 0.1f };
			double[] omega = { 1.f };
			double[] temp = new double[1];
			double[] temp2 = new double[1];

			double[] double_host = new double[1]; // used as helper variable to
												// pass doubles

			// BiCGStab numerical parameters
			int maxit = 1000; // maximum number of iterations
			double tol = 1e-3f; // tolerance nrmr / nrmr0[0], which is size of
								// current errors divided by initial error

			// create the info and analyse the lower and upper triangular
			// factors
			cusparseSolveAnalysisInfo infoL = new cusparseSolveAnalysisInfo();
			cusparseCreateSolveAnalysisInfo(infoL);
			cusparseSolveAnalysisInfo infoU = new cusparseSolveAnalysisInfo();
			cusparseCreateSolveAnalysisInfo(infoU);

			cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					X.GetN(), X.GetNnz(), descr_L, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
					iLUcooColIndex_gpuPtr, infoL);
			cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					X.GetN(), X.GetNnz(), descr_U, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
					iLUcooColIndex_gpuPtr, infoU);

			// 1 : compute initial residual r = b − A x0 ( using initial guess in
			// x )
			cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, X.GetN(), X.GetN(), X.GetNnz(),
					Pointer.to(one_host), X.GetDescr(), X.GetValPtr(),
					X.GetRowIndPtr(), X.GetColIndPtr(), x_gpuPtr,
					Pointer.to(zero_host), r);
			cublasSscal(cublashandle, X.GetN(), Pointer.to(minus_one_host), r, 1);
			cublasSaxpy(cublashandle, X.GetN(), Pointer.to(one_host),
					b_gpuPtr.GetPtr(), 1, r, 1);

			// 2 : Set p=r and \tilde{r}=r
			cublasScopy(cublashandle, X.GetN(), r, 1, p, 1);
			cublasScopy(cublashandle, X.GetN(), r, 1, rw, 1);
			cublasSnrm2(cublashandle, X.GetN(), r, 1, Pointer.to(nrmr0));

			// 3 : repeat until convergence (based on maximum number of
			// iterations and relative residual)

			for (int i = 0; i < maxit; i++) {

				System.out.println("Iteration " + i);

				// 4 : \rho = \tilde{ r }ˆ{T} r
				rhop[0] = rho[0];

				cublasSdot(cublashandle, X.GetN(), rw, 1, r, 1, Pointer.to(rho));

				if (i > 0) {
					// 1 2 : \beta = (\rho{ i } / \rho { i − 1}) ( \alpha /
					// \omega )
					beta[0] = (rho[0] / rhop[0]) * (alpha[0] / omega[0]);

					// 1 3 : p = r + \beta ( p − \omega v )

					double_host[0] = -omega[0];
					cublasSaxpy(cublashandle, X.GetN(), Pointer.to(double_host), q, 1,
							p, 1);
					cublasSscal(cublashandle, X.GetN(), Pointer.to(beta), p, 1);
					cublasSaxpy(cublashandle, X.GetN(), Pointer.to(one_host), r, 1, p,
							1);
				}

				// 1 5 : A \ hat{p} = p ( sparse lower and upper triangular
				// solves )
				cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						X.GetN(), Pointer.to(one_host), descr_L, iLUcooVal_gpuPtr,
						iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoL, p,
						t);

				cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						X.GetN(), Pointer.to(one_host), descr_U, iLUcooVal_gpuPtr,
						iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoU, t,
						ph);

				// 1 6 : q = A \ hat{p} ( sparse matrix−vector multiplication )
				cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, X.GetN(), X.GetN(),
						X.GetNnz(), Pointer.to(one_host), X.GetDescr(), X.GetValPtr(),
						X.GetRowIndPtr(), X.GetColIndPtr(), ph,
						Pointer.to(zero_host), q);

				// 1 7 : \alpha = \rho_{ i } / ( \tilde{ r }ˆ{T} q )

				jcuda.jcublas.JCublas2.cublasSdot(cublashandle, X.GetN(), rw, 1, q, 1,
						Pointer.to(temp));

				alpha[0] = rho[0] / temp[0];

				// 1 8 : s = r − \alpha q

				double_host[0] = -alpha[0];
				cublasSaxpy(cublashandle, X.GetN(), Pointer.to(double_host), q, 1, r, 1);

				// 1 9 : x = x + \alpha \ hat{p};

				cublasSaxpy(cublashandle, X.GetN(), Pointer.to(alpha), ph, 1,
						x_gpuPtr, 1);

				// 2 0 : check for convergence

				cublasSnrm2(cublashandle, X.GetN(), r, 1, Pointer.to(nrmr));

				if (nrmr[0] / nrmr0[0] < tol) {
					break;
				}
				// 2 3 : M \ hat{ s } = r ( sparse lower and upper triangular
				// solves )

				cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						X.GetN(), Pointer.to(one_host), descr_L, iLUcooVal_gpuPtr,
						iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoL, r,
						t);

				cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						X.GetN(), Pointer.to(one_host), descr_U, iLUcooVal_gpuPtr,
						iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoU, t,
						s);

				// 2 4 : t = A \ hat{ s } ( sparse matrix−vector multiplication
				// )

				cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, X.GetN(), X.GetN(),
						X.GetNnz(), Pointer.to(one_host), X.GetDescr(), X.GetValPtr(),
						X.GetRowIndPtr(), X.GetColIndPtr(), s,
						Pointer.to(zero_host), t);

				// 2 5 : \omega = ( tˆ{T} s ) / ( tˆ{T} t )

				cublasSdot(cublashandle, X.GetN(), t, 1, r, 1, Pointer.to(temp));
				cublasSdot(cublashandle, X.GetN(), t, 1, t, 1, Pointer.to(temp2));

				omega[0] = temp[0] / temp2[0];

				// 2 6 : x = x + \omega \ hat{ s }

				cublasSaxpy(cublashandle, X.GetN(), Pointer.to(omega), s, 1, x_gpuPtr,
						1);

				// cudaMemcpy(Pointer.to(result_host), t, 100*Sizeof.double,
				// cudaMemcpyDeviceToHost); //copy results back
				// for(int ii=0;ii<100;ii++)
				// System.out.println("Here t "+ii +"  "+result_host[ii]);

				// 2 7 : r = s − \omega t

				double_host[0] = -omega[0];
				cublasSaxpy(cublashandle, X.GetN(), Pointer.to(double_host), t, 1, r, 1);

				// check for convergence

				cublasSnrm2(cublashandle, X.GetN(), r, 1, Pointer.to(nrmr));

				if (nrmr[0] / nrmr0[0] < tol) {
					break;
				}

				System.out.println("nrmr: " + nrmr[0] + " nrmr0: " + nrmr0[0]
						+ " alpha: " + alpha[0] + " beta: " + beta[0]
						+ " rho: " + rho[0] + " temp: " + temp[0] + " temp2: "
						+ temp2[0] + " omega: " + omega[0]);

			}

			cudaFree(p);
			cudaFree(ph);
			cudaFree(q);
			cudaFree(r);
			cudaFree(rw);
			cudaFree(s);
			cudaFree(t);

			cusparseDestroySolveAnalysisInfo(infoL);
			cusparseDestroySolveAnalysisInfo(infoU);

			cublasDestroy(cublashandle);

		} // CG routine
			// /needs changing
		double result_host[] = new double[X.GetM()]; // array to hold results

		cudaMemcpy(Pointer.to(result_host), x_gpuPtr, X.GetM() * Sizeof.DOUBLE,
				cudaMemcpyDeviceToHost); // copy results back

		cudaFree(x_gpuPtr);
		cudaFree(z_gpuPtr);
		cudaFree(iLUcooColIndex_gpuPtr);
		cudaFree(iLUcooVal_gpuPtr);
		cudaFree(iLUcsrRowIndex_gpuPtr);
		cudaFree(pBuffer);
		cudaFree(structural_zero);
		cudaFree(numerical_zero);

		cusparseDestroyMatDescr(descr_iLU);
		cusparseDestroyMatDescr(descr_L);
		cusparseDestroyMatDescr(descr_U);
		cusparseDestroyCsrilu02Info(info_iLU);
		cusparseDestroyCsrsv2Info(info_L);
		cusparseDestroyCsrsv2Info(info_U);

		return result_host;

	}
	
	
	
}
