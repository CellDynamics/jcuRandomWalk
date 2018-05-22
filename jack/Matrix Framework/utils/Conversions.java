package utils;

import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.jcusparse.cusparseSolveAnalysisInfo;
import jcuda.runtime.JCuda;
import jcuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.*;
import static jcuda.runtime.JCuda.*;

import storage.*;
import utils.Constants.matrixFormat;

public class Conversions {
	//TODO
	//Matrix Format Conversions
	public static SparseMatrix SMtoCOO(cusparseHandle handle, SparseMatrix m) {
		if (m.GetFormat().equals(matrixFormat.MATRIX_FORMAT_COO)) return m;
		return SMGPUtoSM(SMGPUtoCOO(handle, SMtoSMGPU(m)));
	}
	
	public static SparseMatrix SMtoCSR(cusparseHandle handle, SparseMatrix m) {
		if (m.GetFormat().equals(matrixFormat.MATRIX_FORMAT_CSR)) return m;
		return SMGPUtoSM(SMGPUtoCSR(handle, SMtoSMGPU(m)));
	}
	
	public static SparseMatrixGPU SMGPUtoCOO(cusparseHandle handle, SparseMatrixGPU m) {
		if (m.GetFormat().equals(matrixFormat.MATRIX_FORMAT_COO)) return m;
		Pointer rowIndPtr = new Pointer();
		cudaMalloc(rowIndPtr, m.GetNnz()*Sizeof.INT);
		cusparseXcsr2coo(handle, m.GetRowIndPtr(), m.GetNnz(), m.GetM(), rowIndPtr, CUSPARSE_INDEX_BASE_ZERO);
		return new SparseMatrixGPU(rowIndPtr, m.GetColIndPtr(), m.GetValPtr(), m.GetM(), m.GetN(), m.GetNnz(), matrixFormat.MATRIX_FORMAT_COO);
	}
	
	public static SparseMatrixGPU SMGPUtoCSR(cusparseHandle handle, SparseMatrixGPU m) {
		if (m.GetFormat().equals(matrixFormat.MATRIX_FORMAT_CSR)) return m;
		Pointer rowIndPtr = new Pointer();
		cudaMalloc(rowIndPtr, (m.GetM()+1)*Sizeof.INT);
		cusparseXcoo2csr(handle, m.GetRowIndPtr(), m.GetNnz(), m.GetM(), rowIndPtr, CUSPARSE_INDEX_BASE_ZERO);
		return new SparseMatrixGPU(rowIndPtr, m.GetColIndPtr(), m.GetValPtr(), m.GetM(), m.GetN(), m.GetNnz(), matrixFormat.MATRIX_FORMAT_CSR);
	}
	
	//Matrix Transposition Conversions
	
	
	//Vector Type Conversions
	public static DenseVector DVGPUtoDV(DenseVectorGPU v) {
		Pointer ptr = v.GetPtr();
		int sz = v.GetSize();
		double[] data = new double[sz];
		cudaMemcpy(Pointer.to(data), ptr, sz*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
		return new DenseVector(data);
	}
	
	public static DenseVectorGPU DVtoDVGPU(DenseVector v) {
		return new DenseVectorGPU(v.GetData());
	}
	
	//Matrix Type Conversions
	public static DenseMatrix SMtoDM(SparseMatrix m) {
		if (m.GetFormat().equals(matrixFormat.MATRIX_FORMAT_CSR)) {
			System.out.println("Convert to COO fomat before converting to dense!!");
			return null;
		}
		int h = m.GetM();
		int w = m.GetN();
		double[][] data = new double[h][w];
		int[] rowInd = m.GetRowInd();
		int[] colInd = m.GetColInd();
		double[] vals = m.GetVals();
		int nnz = m.GetNnz();		
		for (int i = 0; i < nnz; i++) {
			data[rowInd[i]][colInd[i]] =  vals[i];
		}		
		return new DenseMatrix(data);
	}
	
	public static DenseMatrix DMGPUtoDM(DenseMatrixGPU m) {
		double[] data = new double[m.GetSize()];
		cudaMemcpy(Pointer.to(data), m.GetPtr(), m.GetSize()*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
		double[][] in = new double[m.GetM()][m.GetN()];
		int ix = 0;
		for (int i = 0; i < m.GetM(); i++) {
			for (int j = 0; j < m.GetN(); j++) {
				in[i][j] = data[ix++];
			}
		}
		return new DenseMatrix(in);
	}
	
	public static DenseMatrix SMGPUtoDM(SparseMatrixGPU m) {
		//SparseGPU->Sparse->Dense
		return SMtoDM(SMGPUtoSM(m));
	}
	
	public static DenseMatrixGPU SMGPUtoDMGPU(SparseMatrixGPU m) {
		//SparseGPU->Sparse->Dense->DenseGPU
		return DMtoDMGPU(SMtoDM(SMGPUtoSM(m)));
	}
	
	public static DenseMatrixGPU SMtoDMGPU(SparseMatrix m) {
		//Sparse->Dense->DenseGPU
		return DMtoDMGPU(SMtoDM(m));
	}
	
	public static DenseMatrixGPU DMtoDMGPU(DenseMatrix m) {
		return new DenseMatrixGPU(m.GetData());
	}
	
	public static SparseMatrix DMtoSM(DenseMatrix m) {
		int h = m.GetM();
		int w = m.GetN();
		double[][] data = m.GetData();
		int nnz = 0;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				if (data[i][j] != 0) nnz++;
			}
		}
		int[] rowInd = new int[nnz];
		int[] colInd = new int[nnz];
		double[] vals = new double[nnz];
		int ix = 0;
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				if (data[i][j] != 0) {
					rowInd[ix] = i;
					colInd[ix] = j;
					vals[ix++] = data[i][j];
				}
			}
		}
		return new SparseMatrix(rowInd, colInd, vals, h, w, nnz, matrixFormat.MATRIX_FORMAT_COO);
	}
	
	public static SparseMatrix SMGPUtoSM(SparseMatrixGPU m) {
		int rowSize;
		if (m.GetFormat() == matrixFormat.MATRIX_FORMAT_COO) {
			rowSize = m.GetNnz();
		} else {
			rowSize = m.GetM()+1;
		}
		int[] rowInd = new int[rowSize];
		int[] colInd = new int[m.GetNnz()];
		double[] vals = new double[m.GetNnz()];
		
		cudaMemcpy(Pointer.to(rowInd), m.GetRowIndPtr(), rowSize*Sizeof.INT, cudaMemcpyDeviceToHost);
		cudaMemcpy(Pointer.to(colInd), m.GetColIndPtr(), m.GetNnz()*Sizeof.INT, cudaMemcpyDeviceToHost);
		cudaMemcpy(Pointer.to(vals), m.GetValPtr(), m.GetNnz()*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
		
		return new SparseMatrix(rowInd, colInd, vals, m.GetM(), m.GetN(), m.GetNnz(), m.GetFormat());

	}
	
	public static SparseMatrix DMGPUtoSM(DenseMatrixGPU m) {
		//DenseGPU->Dense->Sparse
		return DMtoSM(DMGPUtoDM(m));
	}
	
	public static SparseMatrixGPU DMGPUtoSMGPU(DenseMatrixGPU m) {
		//DenseGPU->Dense->Sparse->SparseGPU
		return SMtoSMGPU(DMtoSM(DMGPUtoDM(m)));
	}
	
	public static SparseMatrixGPU SMtoSMGPU(SparseMatrix m) {
		return new SparseMatrixGPU(m.GetRowInd(), m.GetColInd(), m.GetVals(), m.GetM(), m.GetN(), m.GetNnz(), m.GetFormat());
	}
	
	public static SparseMatrixGPU DMtoSMGPU(DenseMatrix m) {
		//Dense->Sparse->SparseGPU
		return SMtoSMGPU(DMtoSM(m));
	}
}
