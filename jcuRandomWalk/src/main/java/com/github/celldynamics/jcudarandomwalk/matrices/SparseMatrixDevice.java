package com.github.celldynamics.jcudarandomwalk.matrices;

import static jcuda.jcusparse.JCusparse.cusparseCreateMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseDcsr2csc;
import static jcuda.jcusparse.JCusparse.cusparseDcsrgemm;
import static jcuda.jcusparse.JCusparse.cusparseDestroyMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseSetMatIndexBase;
import static jcuda.jcusparse.JCusparse.cusparseSetMatType;
import static jcuda.jcusparse.JCusparse.cusparseXcoo2csr;
import static jcuda.jcusparse.JCusparse.cusparseXcsr2coo;
import static jcuda.jcusparse.JCusparse.cusparseXcsrgemmNnz;
import static jcuda.jcusparse.cusparseAction.CUSPARSE_ACTION_NUMERIC;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import org.apache.commons.lang3.NotImplementedException;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.runtime.JCuda;

/**
 * Represent sparse matrix on the GPU.
 * 
 * @author p.baniukiewicz
 *
 */
public class SparseMatrixDevice extends SparseMatrix {

  /**
   * UID.
   */
  private static final long serialVersionUID = 2760825038592785223L;
  /**
   * Handle to cusparse driver.
   * 
   * It must be created before use: <tt>JCusparse.cusparseCreate(SparseMatrixDevice.handle);</tt>
   * and then destroyed: <tt>JCusparse.cusparseDestroy(SparseMatrixDevice.handle);</tt>
   */
  public static cusparseHandle handle = new cusparseHandle();
  private cusparseMatDescr descr = new cusparseMatDescr();
  private Pointer rowIndPtr = new Pointer();
  private Pointer colIndPtr = new Pointer();
  private Pointer valPtr = new Pointer();

  /**
   * Set up sparse engine. Should not be called directly.
   */
  SparseMatrixDevice() {
    cusparseCreateMatDescr(descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  }

  /**
   * Create sparse matrix on GPU.
   * 
   * <p>Specified arrays must have proper size depending on <tt>SparseMatrixType</tt>.
   * 
   * @param rowInd indices of rows in matrixInputFormat
   * @param colInd indices of cols in matrixInputFormat
   * @param val values
   * @param matrixInputFormat format of input arrays
   */
  public SparseMatrixDevice(int[] rowInd, int[] colInd, double[] val,
          SparseMatrixType matrixInputFormat) {
    this();
    if (matrixInputFormat == SparseMatrixType.MATRIX_FORMAT_COO
            && ((rowInd.length != colInd.length) || (rowInd.length != val.length))) {
      throw new IllegalArgumentException("Input arrays should have the same length in COO format");
    }
    this.matrixFormat = matrixInputFormat;
    nnz = rowInd.length;
    this.rowInd = rowInd;
    this.colInd = colInd;
    this.val = val;
    updateDimension();
    cudaMalloc(colIndPtr, colInd.length * Sizeof.INT);
    cudaMalloc(rowIndPtr, rowInd.length * Sizeof.INT);
    cudaMalloc(valPtr, val.length * Sizeof.DOUBLE);
    cudaMemcpy(rowIndPtr, Pointer.to(rowInd), getElementNumber() * Sizeof.INT,
            cudaMemcpyHostToDevice);
    cudaMemcpy(colIndPtr, Pointer.to(colInd), getElementNumber() * Sizeof.INT,
            cudaMemcpyHostToDevice);
    cudaMemcpy(valPtr, Pointer.to(val), getElementNumber() * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
  }

  /**
   * Wrap matrix existing on GPU.
   * 
   * @param rowIndPtr Pointer to rows indices vector
   * @param colIndPtr Pointer to columns indices vector
   * @param valPtr Pointer to values vector
   * @param nrows number of rows (of full matrix)
   * @param ncols number of columns
   * @param nnz number of non zero elements
   * @param fmt CUDA matrix format
   * 
   * @see cusparseMatDescr
   * @see SparseMatrixType
   */
  public SparseMatrixDevice(Pointer rowIndPtr, Pointer colIndPtr, Pointer valPtr, int nrows,
          int ncols, int nnz, SparseMatrixType fmt) {
    this();
    this.rowIndPtr = rowIndPtr;
    this.colIndPtr = colIndPtr;
    this.valPtr = valPtr;
    this.rowNumber = nrows;
    this.colNumber = ncols;
    this.nnz = nnz;
    this.matrixFormat = fmt;
  }

  /**
   * Wrap matrix existing on GPU.
   * 
   * @param descr matrix descriptor
   * @param rowIndPtr Pointer to rows indices vector
   * @param colIndPtr Pointer to columns indices vector
   * @param valPtr Pointer to values vector
   * @param nrows number of rows (of full matrix)
   * @param ncols number of columns
   * @param nnz number of non zero elements
   * @param fmt CUDA matrix format
   * @see cusparseMatDescr
   * @see SparseMatrixType
   */
  public SparseMatrixDevice(cusparseMatDescr descr, Pointer rowIndPtr, Pointer colIndPtr,
          Pointer valPtr, int nrows, int ncols, int nnz, SparseMatrixType fmt) {
    this();
    this.descr = descr;
    this.rowIndPtr = rowIndPtr;
    this.colIndPtr = colIndPtr;
    this.valPtr = valPtr;
    this.rowNumber = nrows;
    this.colNumber = ncols;
    this.nnz = nnz;
    this.matrixFormat = fmt;
  }

  /**
   * Copy indices from device to host.
   */
  public void retrieveFromDevice() {
    int indRowSize = computeIndicesLength();
    rowInd = new int[indRowSize];
    colInd = new int[getElementNumber()];
    val = new double[getElementNumber()];
    cudaMemcpy(Pointer.to(rowInd), getRowIndPtr(), indRowSize * Sizeof.INT, cudaMemcpyDeviceToHost);
    cudaMemcpy(Pointer.to(colInd), getColIndPtr(), getElementNumber() * Sizeof.INT,
            cudaMemcpyDeviceToHost);
    cudaMemcpy(Pointer.to(val), getValPtr(), getElementNumber() * Sizeof.DOUBLE,
            cudaMemcpyDeviceToHost);
  }

  /**
   * Compute sizes of indices arrays depending on matrix type.
   * 
   * @return currently number of row indices.
   */
  private int computeIndicesLength() {
    int indRowSize; // depends on sparse matrix type
    switch (matrixFormat) {
      case MATRIX_FORMAT_COO:
        indRowSize = getElementNumber(); // nnz
        break;
      case MATRIX_FORMAT_CSR:
        indRowSize = getRowNumber() + 1;
        break;
      default:
        throw new IllegalArgumentException("Sparse format not supported.");
    }
    return indRowSize;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrixGpu#free()
   */
  @Override
  public void free() {
    // TODO protect against freeing already fried
    cudaFree(rowIndPtr);
    cudaFree(colIndPtr);
    cudaFree(valPtr);
    cusparseDestroyMatDescr(descr);
  }

  /**
   * @return the descr
   */
  public cusparseMatDescr getDescr() {
    return descr;
  }

  /**
   * @return the rowIndPtr
   */
  public Pointer getRowIndPtr() {
    return rowIndPtr;
  }

  /**
   * @return the colIndPtr
   */
  public Pointer getColIndPtr() {
    return colIndPtr;
  }

  /**
   * @return the valPtr
   */
  public Pointer getValPtr() {
    return valPtr;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrix#getElementNumber()
   */
  @Override
  public int getElementNumber() {
    // TODO Consider cusparse<t>nnz()
    return super.getElementNumber();
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrix#convert2csr()
   */
  @Override
  public ISparseMatrix convert2csr() {
    if (matrixFormat == SparseMatrixType.MATRIX_FORMAT_CSR) {
      return this;
    } else {
      Pointer rowIndPtr = new Pointer();
      cudaMalloc(rowIndPtr, (getRowNumber() + 1) * Sizeof.INT);
      cusparseXcoo2csr(handle, getRowIndPtr(), getElementNumber(), getRowNumber(), rowIndPtr,
              CUSPARSE_INDEX_BASE_ZERO);
      return new SparseMatrixDevice(rowIndPtr, getColIndPtr(), getValPtr(), getRowNumber(),
              getColNumber(), getElementNumber(), SparseMatrixType.MATRIX_FORMAT_CSR);
    }
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrix#convert2coo()
   */
  @Override
  public ISparseMatrix convert2coo() {
    if (matrixFormat == SparseMatrixType.MATRIX_FORMAT_COO) {
      return this;
    } else {
      Pointer rowIndPtr = new Pointer();
      cudaMalloc(rowIndPtr, getElementNumber() * Sizeof.INT);
      cusparseXcsr2coo(handle, getRowIndPtr(), getElementNumber(), getRowNumber(), rowIndPtr,
              CUSPARSE_INDEX_BASE_ZERO);
      return new SparseMatrixDevice(rowIndPtr, getColIndPtr(), getValPtr(), getRowNumber(),
              getColNumber(), getElementNumber(), SparseMatrixType.MATRIX_FORMAT_COO);
    }
  }

  /*
   * (non-Javadoc)
   * 
   * @see
   * com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrixGpu#multiplyThisBy(com.github.
   * celldynamics.jcudarandomwalk.matrices.ISparseMatrixGpu)
   */
  @Override
  public IMatrix multiply(IMatrix in) {
    if (this.getColNumber() != in.getRowNumber()) {
      throw new IllegalArgumentException("Incompatibile sizes");
    }
    if (((ISparseMatrix) in).getSparseMatrixType() != SparseMatrixType.MATRIX_FORMAT_CSR) {
      throw new IllegalArgumentException("multiply requires CSR input format.");
    }
    SparseMatrixDevice m2;
    // if not instance og GPU try to convert it to GPU
    m2 = (SparseMatrixDevice) in.toGpu();

    int m = this.getRowNumber();
    int n = m2.getColNumber();
    int k = this.getColNumber();
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
    cudaMalloc(rowIndOutPtr, (m + 1) * Sizeof.INT);
    cusparseXcsrgemmNnz(handle, n_t, n_t, m, n, k, this.getDescr(), this.getElementNumber(),
            this.getRowIndPtr(), this.getColIndPtr(), m2.getDescr(), m2.getElementNumber(),
            m2.getRowIndPtr(), m2.getColIndPtr(), descrOut, rowIndOutPtr, nnzOutPtr);
    JCuda.cudaDeviceSynchronize();
    cudaMemcpy(Pointer.to(nnzOut), nnzOutPtr, Sizeof.INT, cudaMemcpyDeviceToHost);
    cudaMalloc(colIndOutPtr, nnzOut[0] * Sizeof.INT);
    cudaMalloc(valOutPtr, nnzOut[0] * Sizeof.DOUBLE);
    cusparseDcsrgemm(handle, n_t, n_t, m, n, k, this.getDescr(), this.getElementNumber(),
            this.getValPtr(), this.getRowIndPtr(), this.getColIndPtr(), m2.getDescr(),
            m2.getElementNumber(), m2.getValPtr(), m2.getRowIndPtr(), m2.getColIndPtr(), descrOut,
            valOutPtr, rowIndOutPtr, colIndOutPtr);
    JCuda.cudaDeviceSynchronize();
    return new SparseMatrixDevice(descrOut, rowIndOutPtr, colIndOutPtr, valOutPtr, m, n, nnzOut[0],
            SparseMatrixType.MATRIX_FORMAT_CSR);
  }

  /**
   * Create copy of this matrix on Host.
   * 
   * <p>Note that this is shallow copy. You may wish to call {@link #retrieveFromDevice()} before.
   * 
   * @return reference to host matrix
   * @see #retrieveFromDevice()
   */
  @Override
  public ISparseMatrix toCpu() {
    if (colInd == null || rowInd == null || val == null) {
      retrieveFromDevice();
    }
    return new SparseMatrixHost(getRowInd(), getColInd(), getVal(), matrixFormat);
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrix#transpose()
   */
  @Override
  public ISparseMatrix transpose() {
    SparseMatrixDevice csrm = (SparseMatrixDevice) this.convert2csr();

    Pointer colIndPtr = new Pointer();
    cudaMalloc(colIndPtr, (csrm.getColNumber() + 1) * Sizeof.INT);
    Pointer rowIndPtr = new Pointer();
    cudaMalloc(rowIndPtr, csrm.getElementNumber() * Sizeof.INT);
    Pointer valPtr = new Pointer();
    cudaMalloc(valPtr, csrm.getElementNumber() * Sizeof.DOUBLE);

    cusparseDcsr2csc(handle, csrm.getRowNumber(), csrm.getColNumber(), csrm.getElementNumber(),
            csrm.getValPtr(), csrm.getRowIndPtr(), csrm.getColIndPtr(), valPtr, rowIndPtr,
            colIndPtr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);

    // cusparseXcoo2csr(handle, getRowIndPtr(), getElementNumber(), getRowNumber(), rowIndPtr,
    // CUSPARSE_INDEX_BASE_ZERO);
    // return new SparseMatrixDevice(rowIndPtr, getColIndPtr(), getValPtr(), getRowNumber(),
    // getColNumber(), getElementNumber(), SparseMatrixType.MATRIX_FORMAT_CSR);
    return new SparseMatrixDevice(colIndPtr, rowIndPtr, valPtr, csrm.getColNumber(),
            csrm.getRowNumber(), getElementNumber(), SparseMatrixType.MATRIX_FORMAT_CSR);

  }

  /*
   * (non-Javadoc)
   * 
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString() {
    retrieveFromDevice();
    return super.toString();
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#removeRows(int[])
   */
  @Override
  public IMatrix removeRows(int[] rows) {
    throw new NotImplementedException("Not implemented");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#removeCols(int[])
   */
  @Override
  public IMatrix removeCols(int[] cols) {
    throw new NotImplementedException("Not implemented");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#toGpu()
   */
  @Override
  public IMatrix toGpu() {
    return this;
  }

  @Override
  public IMatrix sumAlongRows() {
    throw new NotImplementedException("Not implemented");
  }

}
