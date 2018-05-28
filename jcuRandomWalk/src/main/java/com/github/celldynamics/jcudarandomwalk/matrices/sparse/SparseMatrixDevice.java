package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasSaxpy;
import static jcuda.jcublas.JCublas2.cublasScopy;
import static jcuda.jcublas.JCublas2.cublasSdot;
import static jcuda.jcublas.JCublas2.cublasSnrm2;
import static jcuda.jcublas.JCublas2.cublasSscal;
import static jcuda.jcusparse.JCusparse.cusparseCreateCsrilu02Info;
import static jcuda.jcusparse.JCusparse.cusparseCreateCsrsv2Info;
import static jcuda.jcusparse.JCusparse.cusparseCreateMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseCreateSolveAnalysisInfo;
import static jcuda.jcusparse.JCusparse.cusparseDcsr2csc;
import static jcuda.jcusparse.JCusparse.cusparseDcsrgemm;
import static jcuda.jcusparse.JCusparse.cusparseDestroyCsrilu02Info;
import static jcuda.jcusparse.JCusparse.cusparseDestroyCsrsv2Info;
import static jcuda.jcusparse.JCusparse.cusparseDestroyMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseDestroySolveAnalysisInfo;
import static jcuda.jcusparse.JCusparse.cusparseScsrilu02;
import static jcuda.jcusparse.JCusparse.cusparseScsrilu02_analysis;
import static jcuda.jcusparse.JCusparse.cusparseScsrilu02_bufferSize;
import static jcuda.jcusparse.JCusparse.cusparseScsrmv;
import static jcuda.jcusparse.JCusparse.cusparseScsrsv2_analysis;
import static jcuda.jcusparse.JCusparse.cusparseScsrsv2_bufferSize;
import static jcuda.jcusparse.JCusparse.cusparseScsrsv2_solve;
import static jcuda.jcusparse.JCusparse.cusparseScsrsv_analysis;
import static jcuda.jcusparse.JCusparse.cusparseScsrsv_solve;
import static jcuda.jcusparse.JCusparse.cusparseSetMatDiagType;
import static jcuda.jcusparse.JCusparse.cusparseSetMatFillMode;
import static jcuda.jcusparse.JCusparse.cusparseSetMatIndexBase;
import static jcuda.jcusparse.JCusparse.cusparseSetMatType;
import static jcuda.jcusparse.JCusparse.cusparseSetPointerMode;
import static jcuda.jcusparse.JCusparse.cusparseXcoo2csr;
import static jcuda.jcusparse.JCusparse.cusparseXcsr2coo;
import static jcuda.jcusparse.JCusparse.cusparseXcsrgemmNnz;
import static jcuda.jcusparse.JCusparse.cusparseXcsrilu02_zeroPivot;
import static jcuda.jcusparse.cusparseAction.CUSPARSE_ACTION_NUMERIC;
import static jcuda.jcusparse.cusparseDiagType.CUSPARSE_DIAG_TYPE_NON_UNIT;
import static jcuda.jcusparse.cusparseDiagType.CUSPARSE_DIAG_TYPE_UNIT;
import static jcuda.jcusparse.cusparseFillMode.CUSPARSE_FILL_MODE_LOWER;
import static jcuda.jcusparse.cusparseFillMode.CUSPARSE_FILL_MODE_UPPER;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparsePointerMode.CUSPARSE_POINTER_MODE_DEVICE;
import static jcuda.jcusparse.cusparsePointerMode.CUSPARSE_POINTER_MODE_HOST;
import static jcuda.jcusparse.cusparseSolvePolicy.CUSPARSE_SOLVE_POLICY_NO_LEVEL;
import static jcuda.jcusparse.cusparseSolvePolicy.CUSPARSE_SOLVE_POLICY_USE_LEVEL;
import static jcuda.jcusparse.cusparseStatus.CUSPARSE_STATUS_ZERO_PIVOT;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import org.apache.commons.lang3.NotImplementedException;

import com.github.celldynamics.jcudarandomwalk.matrices.ICudaLibHandles;
import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVectorDevice;
import com.github.celldynamics.jcudarandomwalk.matrices.dense.IDenseVector;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusparse.csrilu02Info;
import jcuda.jcusparse.csrsv2Info;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.jcusparse.cusparseSolveAnalysisInfo;
import jcuda.runtime.JCuda;

/**
 * Represent sparse matrix on the GPU.
 * 
 * @author p.baniukiewicz
 *
 */
public class SparseMatrixDevice extends SparseMatrix implements ICudaLibHandles {

  /**
   * Default UID.
   */
  private static final long serialVersionUID = 2760825038592785223L;

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
   * Return cuSparse descriptor.
   * 
   * @return the descr
   */
  public cusparseMatDescr getDescr() {
    return descr;
  }

  /**
   * Return rows indices pointer.
   * 
   * @return the rowIndPtr
   */
  public Pointer getRowIndPtr() {
    return rowIndPtr;
  }

  /**
   * Return columns indices pointer.
   * 
   * @return the colIndPtr
   */
  public Pointer getColIndPtr() {
    return colIndPtr;
  }

  /**
   * Return pointer to values.
   * 
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

    cusparseMatDescr descrOut = new cusparseMatDescr();
    cusparseCreateMatDescr(descrOut);
    cusparseSetMatType(descrOut, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrOut, CUSPARSE_INDEX_BASE_ZERO);

    Pointer nnzOutPtr = new Pointer();
    Pointer rowIndOutPtr = new Pointer();
    int nt = CUSPARSE_OPERATION_NON_TRANSPOSE;
    int n = m2.getColNumber();
    int k = this.getColNumber();
    int[] nnzOut = new int[1];
    cudaMalloc(nnzOutPtr, Sizeof.INT);
    cudaMalloc(rowIndOutPtr, (m + 1) * Sizeof.INT);
    cusparseXcsrgemmNnz(handle, nt, nt, m, n, k, this.getDescr(), this.getElementNumber(),
            this.getRowIndPtr(), this.getColIndPtr(), m2.getDescr(), m2.getElementNumber(),
            m2.getRowIndPtr(), m2.getColIndPtr(), descrOut, rowIndOutPtr, nnzOutPtr);
    JCuda.cudaDeviceSynchronize();
    Pointer colIndOutPtr = new Pointer();
    Pointer valOutPtr = new Pointer();
    cudaMemcpy(Pointer.to(nnzOut), nnzOutPtr, Sizeof.INT, cudaMemcpyDeviceToHost);
    cudaMalloc(colIndOutPtr, nnzOut[0] * Sizeof.INT);
    cudaMalloc(valOutPtr, nnzOut[0] * Sizeof.DOUBLE);
    cusparseDcsrgemm(handle, nt, nt, m, n, k, this.getDescr(), this.getElementNumber(),
            this.getValPtr(), this.getRowIndPtr(), this.getColIndPtr(), m2.getDescr(),
            m2.getElementNumber(), m2.getValPtr(), m2.getRowIndPtr(), m2.getColIndPtr(), descrOut,
            valOutPtr, rowIndOutPtr, colIndOutPtr);
    JCuda.cudaDeviceSynchronize();
    return new SparseMatrixDevice(descrOut, rowIndOutPtr, colIndOutPtr, valOutPtr, m, n, nnzOut[0],
            SparseMatrixType.MATRIX_FORMAT_CSR);
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#toCpu()
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

  @Override
  public double[] luSolve(IDenseVector b_gpuPtrAny, boolean iLuBiCGStabSolve) {

    if (getColNumber() != getRowNumber()) {
      throw new IllegalArgumentException("Left matrix must be square");
    }
    DenseVectorDevice b_gpuPtr = (DenseVectorDevice) b_gpuPtrAny.toGpu();
    int m = getRowNumber();
    int n = m;
    Pointer AcsrRowIndex_gpuPtr = getRowIndPtr();
    Pointer AcooColIndex_gpuPtr = getColIndPtr();
    Pointer AcooVal_gpuPtr = getValPtr();
    // some useful constants
    float[] one_host = { 1.f };
    float[] zero_host = { 0.f };
    float[] minus_one_host = { -1.f };

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
    cudaMalloc(x_gpuPtr, m * Sizeof.FLOAT);
    cudaMalloc(z_gpuPtr, m * Sizeof.FLOAT);

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

    cusparseCreateMatDescr(descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);
    int policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    int trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;

    cusparseMatDescr descr_U = new cusparseMatDescr();

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
    cudaMalloc(iLUcsrRowIndex_gpuPtr, (m + 1) * Sizeof.INT);
    cudaMalloc(iLUcooColIndex_gpuPtr, nnz * Sizeof.INT);
    cudaMalloc(iLUcooVal_gpuPtr, nnz * Sizeof.FLOAT);
    cudaMemcpy(iLUcsrRowIndex_gpuPtr, AcsrRowIndex_gpuPtr, (m + 1) * Sizeof.INT,
            cudaMemcpyDeviceToDevice);
    cudaMemcpy(iLUcooColIndex_gpuPtr, AcooColIndex_gpuPtr, nnz * Sizeof.INT,
            cudaMemcpyDeviceToDevice);
    cudaMemcpy(iLUcooVal_gpuPtr, AcooVal_gpuPtr, nnz * Sizeof.FLOAT, cudaMemcpyDeviceToDevice);

    // set up buffer
    int[] pBufferSize_iLU = new int[1];
    int[] pBufferSize_L = new int[1];
    int[] pBufferSize_U = new int[1];
    int pBufferSize;
    Pointer pBuffer = new Pointer();

    // step 3: query how much memory used in csrilu02 and csrsv2, and
    // allocate the buffer
    cusparseScsrilu02_bufferSize(handle, m, nnz, descr_iLU, AcooVal_gpuPtr, AcsrRowIndex_gpuPtr,
            AcooColIndex_gpuPtr, info_iLU, pBufferSize_iLU);
    cusparseScsrsv2_bufferSize(handle, trans_L, m, nnz, descr_L, AcooVal_gpuPtr,
            AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, info_L, pBufferSize_L);
    cusparseScsrsv2_bufferSize(handle, trans_U, m, nnz, descr_U, AcooVal_gpuPtr,
            AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, info_U, pBufferSize_U);

    pBufferSize = Math.max(pBufferSize_iLU[0], Math.max(pBufferSize_L[0], pBufferSize_U[0]));
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

    cusparseScsrilu02_analysis(handle, m, nnz, descr_iLU, AcooVal_gpuPtr, AcsrRowIndex_gpuPtr,
            AcooColIndex_gpuPtr, info_iLU, policy_iLU, pBuffer);

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
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseXcsrilu02_zeroPivot(handle, info_iLU,
            structural_zero)) {
      int[] sz = new int[1];
      cudaMemcpy(Pointer.to(sz), structural_zero, Sizeof.INT, cudaMemcpyDeviceToHost); // copy
                                                                                       // results
                                                                                       // back
      System.out.printf("A(%d,%d) is missing\n", sz[0], sz[0]);
    }
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    cusparseScsrsv2_analysis(handle, trans_L, m, nnz, descr_L, AcooVal_gpuPtr, AcsrRowIndex_gpuPtr,
            AcooColIndex_gpuPtr, info_L, policy_L, pBuffer);

    cusparseScsrsv2_analysis(handle, trans_U, m, nnz, descr_U, AcooVal_gpuPtr, AcsrRowIndex_gpuPtr,
            AcooColIndex_gpuPtr, info_U, policy_U, pBuffer);

    // step 5: M = L * U
    cusparseScsrilu02(handle, m, nnz, descr_iLU, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
            iLUcooColIndex_gpuPtr, info_iLU, policy_iLU, pBuffer);

    Pointer numerical_zero = new Pointer();
    cudaMalloc(numerical_zero, Sizeof.INT);

    // same trick of switching modes needed here
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseXcsrilu02_zeroPivot(handle, info_iLU,
            numerical_zero)) {
      int[] nz = new int[1];
      cudaMemcpy(Pointer.to(nz), numerical_zero, Sizeof.INT, cudaMemcpyDeviceToHost); // copy
                                                                                      // results
                                                                                      // back
      System.out.printf("U(%d,%d) is zero\n", nz[0], nz[0]);
    }
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    // step 6: solve L*z = x
    cusparseScsrsv2_solve(handle, trans_L, m, nnz, Pointer.to(one_host), descr_L, iLUcooVal_gpuPtr,
            iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, info_L, b_gpuPtr.getValPtr(), z_gpuPtr,
            policy_L, pBuffer);

    // step 7: solve U*y = z
    cusparseScsrsv2_solve(handle, trans_U, m, nnz, Pointer.to(one_host), descr_U, iLUcooVal_gpuPtr,
            iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, info_U, z_gpuPtr, x_gpuPtr, policy_U,
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
      // 3.The matrixA (valA, csrRowPtrA, csrColIndA) and the incompleteâˆ’
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

      cudaMalloc(p, m * Sizeof.FLOAT);
      cudaMalloc(ph, m * Sizeof.FLOAT);
      cudaMalloc(q, m * Sizeof.FLOAT);
      cudaMalloc(r, m * Sizeof.FLOAT);
      cudaMalloc(rw, m * Sizeof.FLOAT);
      cudaMalloc(s, m * Sizeof.FLOAT);
      cudaMalloc(t, m * Sizeof.FLOAT);

      // BiCGStab parameters (all on host)
      float[] nrmr0 = new float[1];
      float[] nrmr = new float[1];

      float[] rho = { 1.f };
      float[] rhop = new float[1];
      float[] alpha = { 1.f };
      float[] beta = { 0.1f };
      float[] omega = { 1.f };
      float[] temp = new float[1];
      float[] temp2 = new float[1];

      float[] float_host = new float[1]; // used as helper variable to
      // pass floats

      // BiCGStab numerical parameters
      int maxit = 200; // maximum number of iterations
      float tol = 1e-3f; // tolerance nrmr / nrmr0[0], which is size of
      // current errors divided by initial error

      // create the info and analyse the lower and upper triangular
      // factors
      cusparseSolveAnalysisInfo infoL = new cusparseSolveAnalysisInfo();
      cusparseCreateSolveAnalysisInfo(infoL);
      cusparseSolveAnalysisInfo infoU = new cusparseSolveAnalysisInfo();
      cusparseCreateSolveAnalysisInfo(infoU);

      cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, descr_L,
              iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoL);
      cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, descr_U,
              iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoU);

      // 1 : compute initial residual r = b âˆ’ A x0 ( using initial guess in
      // x )
      cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, Pointer.to(one_host),
              getDescr(), AcooVal_gpuPtr, AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, x_gpuPtr,
              Pointer.to(zero_host), r);
      cublasSscal(cublashandle, n, Pointer.to(minus_one_host), r, 1);
      cublasSaxpy(cublashandle, n, Pointer.to(one_host), b_gpuPtr.getValPtr(), 1, r, 1);

      // 2 : Set p=r and \tilde{r}=r
      cublasScopy(cublashandle, n, r, 1, p, 1);
      cublasScopy(cublashandle, n, r, 1, rw, 1);
      cublasSnrm2(cublashandle, n, r, 1, Pointer.to(nrmr0));

      // 3 : repeat until convergence (based on maximum number of
      // iterations and relative residual)

      for (int i = 0; i < maxit; i++) {

        System.out.println("Iteration " + i);

        // 4 : \rho = \tilde{ r }Ë†{T} r
        rhop[0] = rho[0];

        cublasSdot(cublashandle, n, rw, 1, r, 1, Pointer.to(rho));

        if (i > 0) {
          // 1 2 : \beta = (\rho{ i } / \rho { i âˆ’ 1}) ( \alpha /
          // \omega )
          beta[0] = (rho[0] / rhop[0]) * (alpha[0] / omega[0]);

          // 1 3 : p = r + \beta ( p âˆ’ \omega v )

          float_host[0] = -omega[0];
          cublasSaxpy(cublashandle, n, Pointer.to(float_host), q, 1, p, 1);
          cublasSscal(cublashandle, n, Pointer.to(beta), p, 1);
          cublasSaxpy(cublashandle, n, Pointer.to(one_host), r, 1, p, 1);
        }

        // 1 5 : A \ hat{p} = p ( sparse lower and upper triangular
        // solves )
        cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, Pointer.to(one_host),
                descr_L, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoL, p,
                t);

        cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, Pointer.to(one_host),
                descr_U, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoU, t,
                ph);

        // 1 6 : q = A \ hat{p} ( sparse matrixâˆ’vector multiplication )
        cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, Pointer.to(one_host),
                getDescr(), AcooVal_gpuPtr, AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, ph,
                Pointer.to(zero_host), q);

        // 1 7 : \alpha = \rho_{ i } / ( \tilde{ r }Ë†{T} q )

        jcuda.jcublas.JCublas2.cublasSdot(cublashandle, n, rw, 1, q, 1, Pointer.to(temp));

        alpha[0] = rho[0] / temp[0];

        // 1 8 : s = r âˆ’ \alpha q

        float_host[0] = -alpha[0];
        cublasSaxpy(cublashandle, n, Pointer.to(float_host), q, 1, r, 1);

        // 1 9 : x = x + \alpha \ hat{p};

        cublasSaxpy(cublashandle, n, Pointer.to(alpha), ph, 1, x_gpuPtr, 1);

        // 2 0 : check for convergence

        cublasSnrm2(cublashandle, n, r, 1, Pointer.to(nrmr));

        if (nrmr[0] / nrmr0[0] < tol) {
          break;
        }
        // 2 3 : M \ hat{ s } = r ( sparse lower and upper triangular
        // solves )

        cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, Pointer.to(one_host),
                descr_L, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoL, r,
                t);

        cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, Pointer.to(one_host),
                descr_U, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoU, t,
                s);

        // 2 4 : t = A \ hat{ s } ( sparse matrixâˆ’vector multiplication
        // )

        cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, Pointer.to(one_host),
                getDescr(), AcooVal_gpuPtr, AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, s,
                Pointer.to(zero_host), t);

        // 2 5 : \omega = ( tË†{T} s ) / ( tË†{T} t )

        cublasSdot(cublashandle, n, t, 1, r, 1, Pointer.to(temp));
        cublasSdot(cublashandle, n, t, 1, t, 1, Pointer.to(temp2));

        omega[0] = temp[0] / temp2[0];

        // 2 6 : x = x + \omega \ hat{ s }

        cublasSaxpy(cublashandle, n, Pointer.to(omega), s, 1, x_gpuPtr, 1);

        // cudaMemcpy(Pointer.to(result_host), t, 100*Sizeof.FLOAT,
        // cudaMemcpyDeviceToHost); //copy results back
        // for(int ii=0;ii<100;ii++)
        // System.out.println("Here t "+ii +" "+result_host[ii]);

        // 2 7 : r = s âˆ’ \omega t

        float_host[0] = -omega[0];
        cublasSaxpy(cublashandle, n, Pointer.to(float_host), t, 1, r, 1);

        // check for convergence

        cublasSnrm2(cublashandle, n, r, 1, Pointer.to(nrmr));

        if (nrmr[0] / nrmr0[0] < tol) {
          break;
        }

        System.out.println("nrmr: " + nrmr[0] + " nrmr0: " + nrmr0[0] + " alpha: " + alpha[0]
                + " beta: " + beta[0] + " rho: " + rho[0] + " temp: " + temp[0] + " temp2: "
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
    double result_host[] = new double[m]; // array to hold results

    // JCuda.cudaDeviceSynchronize();
    // copy results back
    cudaMemcpy(Pointer.to(result_host), x_gpuPtr, m * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);

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
