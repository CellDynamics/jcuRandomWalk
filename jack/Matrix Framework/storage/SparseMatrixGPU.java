package storage;

import jcuda.*;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.jcusparse.*;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import static jcuda.runtime.cudaMemcpyKind.*;
import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;

import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import utils.Constants.matrixFormat;
import utils.Conversions;

public class SparseMatrixGPU {
	//TODO
	private cusparseHandle handle = new cusparseHandle();
	private cusparseMatDescr descr = new cusparseMatDescr();
	private Pointer rowIndPtr = new Pointer();
	private Pointer colIndPtr = new Pointer();
	private Pointer valPtr = new Pointer();
	private int[] rowIndHost;
	private int[] colIndHost;
	private double[] valHost;
	private int m;
	private int n;
	private int nnz;
	private matrixFormat fmt;
	
	public SparseMatrixGPU(int[] rowInd, int[] colInd, double[] vals, int m, int n, int nnz, matrixFormat fmt) {
		this.rowIndHost = rowInd;
		this.colIndHost = colInd;
		this.valHost = vals;
		this.m = m;
		this.n = n;
		this.nnz = nnz;
		this.fmt = fmt;
		
		cusparseCreateMatDescr(this.descr);
		cusparseSetMatType(this.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
		
		
		int rowSize;
		if (this.fmt == matrixFormat.MATRIX_FORMAT_COO) {
			rowSize = this.nnz;
		} else {
			rowSize = this.m + 1;
		}
		
		cudaMalloc(rowIndPtr, rowSize*Sizeof.INT);
		cudaMalloc(colIndPtr, this.nnz*Sizeof.INT);
		cudaMalloc(valPtr, this.nnz*Sizeof.DOUBLE);
		cudaMemcpy(rowIndPtr, Pointer.to(rowIndHost), rowSize*Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(colIndPtr, Pointer.to(colIndHost), this.nnz*Sizeof.INT, cudaMemcpyHostToDevice);
		cudaMemcpy(valPtr, Pointer.to(valHost), this.nnz*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
		
	}
	
	public SparseMatrixGPU(Pointer rowIndPtr, Pointer colIndPtr, Pointer valPtr, int m, int n, int nnz, matrixFormat fmt) {
		this.rowIndPtr = rowIndPtr;
		this.colIndPtr = colIndPtr;
		this.valPtr = valPtr;
		this.m = m;
		this.n = n;
		this.nnz = nnz;
		this.fmt = fmt;
		cusparseCreateMatDescr(this.descr);
		cusparseSetMatType(this.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	}
	
	public SparseMatrixGPU(cusparseMatDescr descr, Pointer rowIndPtr, Pointer colIndPtr, Pointer valPtr, int m, int n, int nnz, matrixFormat fmt) {
		this.descr = descr;
		this.rowIndPtr = rowIndPtr;
		this.colIndPtr = colIndPtr;
		this.valPtr = valPtr;
		this.m = m;
		this.n = n;
		this.nnz = nnz;
		this.fmt = fmt;
	}
	
	public cusparseMatDescr GetDescr() {
		return this.descr;
	}
	
	public Pointer GetRowIndPtr() {
		return this.rowIndPtr;
	}
	
	public Pointer GetColIndPtr() {
		return this.colIndPtr;
	}
	
	public Pointer GetValPtr() {
		return this.valPtr;
	}
	
	public int GetM() {
		return this.m;
	}
	
	public int GetN() {
		return this.n;
	}
	
	public int GetNnz() {
		return this.nnz;
	}
	
	public matrixFormat GetFormat() {
		return this.fmt;
	}
	
	public void Free() {
		cudaFree(rowIndPtr);
		cudaFree(colIndPtr);
		cudaFree(valPtr);
		
	}
	
	public String toString() {
		if (this.fmt.equals(matrixFormat.MATRIX_FORMAT_COO)) return Conversions.SMGPUtoDM(this).toString();
		return Conversions.SMGPUtoDM(Conversions.SMGPUtoCOO(this.handle, this)).toString();
	}
	
}
