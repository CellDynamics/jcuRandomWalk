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
import static jcuda.jcusparse.JCusparse.cusparseDestroyCsrilu02Info;
import static jcuda.jcusparse.JCusparse.cusparseDestroyCsrsv2Info;
import static jcuda.jcusparse.JCusparse.cusparseDestroyMatDescr;
import static jcuda.jcusparse.JCusparse.cusparseDestroySolveAnalysisInfo;
import static jcuda.jcusparse.JCusparse.cusparseScsr2csc;
import static jcuda.jcusparse.JCusparse.cusparseScsrgemm;
import static jcuda.jcusparse.JCusparse.cusparseScsric0;
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
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_SYMMETRIC;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_TRIANGULAR;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE;
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

import java.util.Arrays;

import org.apache.commons.lang3.time.StopWatch;

import com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVectorDevice;
import com.github.celldynamics.jcurandomwalk.ArrayTools;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusparse.csrilu02Info;
import jcuda.jcusparse.csrsv2Info;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseMatDescr;
import jcuda.jcusparse.cusparseSolveAnalysisInfo;
import jcuda.runtime.JCuda;

/**
 * Represent sparse matrix on the GPU.
 * 
 * @author p.baniukiewicz
 *
 */
public class SparseMatrixDevice extends SparseCoordinates {

  /**
   * 
   */
  private static final long serialVersionUID = -6342278651189202672L;

  /**
   * Reasons of stopping diffusion process.
   * 
   * @author p.baniukiewicz
   *
   */
  private enum StoppedBy {
    /**
     * Maximum number of iterations reached.
     */
    ITERATIONS,
    /**
     * Found NaN in solution.
     */
    NANS,
    /**
     * Found Inf in solution.
     */
    INFS,
    /**
     * Relative error smaller than limit.
     */
    RELERR
  }

  private cusparseMatDescr descr = new cusparseMatDescr();
  private Pointer rowIndPtr = new Pointer();
  private Pointer colIndPtr = new Pointer();
  private Pointer valPtr = new Pointer();
  private int nnz; // number of nonzero elements, specific to GPU implementtion
  private cusparseHandle cusparseHandle = null;
  private cublasHandle cublasHandle = null;
  // preserving analysis between runs
  private boolean done = false; // if true new analysis is skipped
  private boolean useCheating = false;
  // note that this should be called in order with other allocations. This is why destroying
  // in free() does not work. To prevent memory leaks on GPU device is reseted on each job.
  private cusparseSolveAnalysisInfo infoL = new cusparseSolveAnalysisInfo();
  private cusparseSolveAnalysisInfo infoU = new cusparseSolveAnalysisInfo();

  /**
   * Set up sparse engine. Should not be called directly.
   */
  public SparseMatrixDevice() {
  }

  /**
   * Set up sparse engine. Should not be called directly.
   * 
   * @param handle handle to library
   */
  public SparseMatrixDevice(cusparseHandle handle, cublasHandle cublasHandle) {
    this();
    this.cusparseHandle = handle;
    this.cublasHandle = cublasHandle;
  }

  /**
   * Create basic matrix.
   */
  private void initialiseMatrixStruct() {
    cusparseCreateMatDescr(descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  }

  /**
   * Create basic matrix.
   * 
   * @param descr description
   */
  private void initialiseMatrixStruct(cusparseMatDescr descr) {
    this.descr = descr;
  }

  /**
   * Create sparse matrix on GPU. Number of rows and columns
   * is computed automatically.
   * 
   * <p>Specified arrays must have proper size depending on <tt>SparseMatrixType</tt>. Note that
   * this constructor will remove any 0 filled rows or columns which might not be
   * correct.
   * 
   * @param rowInd indices of rows in matrixInputFormat
   * @param colInd indices of cols in matrixInputFormat
   * @param val values
   * @param matrixInputFormat format of input arrays
   * @param handle handle to library
   * @param cublasHandle
   */
  public SparseMatrixDevice(int[] rowInd, int[] colInd, float[] val,
          SparseMatrixType matrixInputFormat, cusparseHandle handle, cublasHandle cublasHandle) {
    this(handle, cublasHandle);
    initialiseMatrixStruct();
    if (matrixInputFormat == SparseMatrixType.MATRIX_FORMAT_COO
            && ((rowInd.length != colInd.length) || (rowInd.length != val.length))) {
      throw new IllegalArgumentException("Input arrays should have the same length in COO format");
    }
    // FIXME sue super constructor
    this.matrixFormat = matrixInputFormat;
    nnz = val.length;
    this.rowInd = rowInd;
    this.colInd = colInd;
    this.val = val;
    updateDimension();
    transferToGpu(rowInd, colInd, val);
  }

  /**
   * Create sparse matrix on GPU.
   * 
   * <p>Specified arrays must have proper size depending on <tt>SparseMatrixType</tt>.
   * 
   * @param rowInd indices of rows in matrixInputFormat
   * @param colInd indices of cols in matrixInputFormat
   * @param val values
   * @param rowNumber number of rows
   * @param colNumber number of columns
   * @param matrixInputFormat format of input arrays
   * @param handle handle to library
   * @param cublasHandle
   */
  public SparseMatrixDevice(int[] rowInd, int[] colInd, float[] val, int rowNumber, int colNumber,
          SparseMatrixType matrixInputFormat, cusparseHandle handle, cublasHandle cublasHandle) {
    this(handle, cublasHandle);
    initialiseMatrixStruct();
    if (matrixInputFormat == SparseMatrixType.MATRIX_FORMAT_COO
            && ((rowInd.length != colInd.length) || (rowInd.length != val.length))) {
      throw new IllegalArgumentException("Input arrays should have the same length in COO format");
    }
    this.matrixFormat = matrixInputFormat;
    nnz = val.length;
    this.rowInd = rowInd;
    this.colInd = colInd;
    this.val = val;
    this.rowNumber = rowNumber;
    this.colNumber = colNumber;
    transferToGpu(rowInd, colInd, val);
  }

  /**
   * Create sparse matrix on GPU. Copy constructor.
   * 
   * <p>Specified arrays must have proper size depending on <tt>SparseMatrixType</tt>.
   * 
   * @param rowInd indices of rows in matrixInputFormat
   * @param colInd indices of cols in matrixInputFormat
   * @param val values
   * @param rowNumber number of rows
   * @param colNumber number of columns
   * @param matrixInputFormat format of input arrays
   * @param uploadToGpu if true it will be transfered to gpu, otherwise will stay on cpu
   * @param handle handle to library
   * @param cublasHandle
   */
  SparseMatrixDevice(int[] rowInd, int[] colInd, float[] val, int rowNumber, int colNumber,
          SparseMatrixType matrixInputFormat, boolean uploadToGpu, cusparseHandle handle,
          cublasHandle cublasHandle) {
    this(handle, cublasHandle);
    initialiseMatrixStruct();
    if (matrixInputFormat == SparseMatrixType.MATRIX_FORMAT_COO
            && ((rowInd.length != colInd.length) || (rowInd.length != val.length))) {
      throw new IllegalArgumentException("Input arrays should have the same length in COO format");
    }
    this.matrixFormat = matrixInputFormat;
    nnz = val.length;
    this.rowInd = Arrays.copyOf(rowInd, rowInd.length);
    this.colInd = Arrays.copyOf(colInd, colInd.length);
    this.val = Arrays.copyOf(val, val.length);
    this.rowNumber = rowNumber;
    this.colNumber = colNumber;
    if (uploadToGpu) {
      transferToGpu(rowInd, colInd, val);
    }
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
   * @param handle handle to library
   * @param cublasHandle
   * 
   * @see cusparseMatDescr
   * @see SparseMatrixType
   */
  public SparseMatrixDevice(Pointer rowIndPtr, Pointer colIndPtr, Pointer valPtr, int nrows,
          int ncols, int nnz, SparseMatrixType fmt, cusparseHandle handle,
          cublasHandle cublasHandle) {
    this(handle, cublasHandle);
    initialiseMatrixStruct();
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
   * @param rowIndPtr Pointer to rows indices vector
   * @param colIndPtr Pointer to columns indices vector
   * @param valPtr Pointer to values vector
   * @param nrows number of rows (of full matrix)
   * @param ncols number of columns
   * @param nnz number of non zero elements
   * @param fmt CUDA matrix format
   * @param handle handle to library
   * @param descr matrix description
   * @param cublasHandle
   * 
   * @see cusparseMatDescr
   * @see SparseMatrixType
   */
  public SparseMatrixDevice(Pointer rowIndPtr, Pointer colIndPtr, Pointer valPtr, int nrows,
          int ncols, int nnz, SparseMatrixType fmt, cusparseMatDescr descr, cusparseHandle handle,
          cublasHandle cublasHandle) {
    this(handle, cublasHandle);
    initialiseMatrixStruct(descr);
    this.rowIndPtr = rowIndPtr;
    this.colIndPtr = colIndPtr;
    this.valPtr = valPtr;
    this.rowNumber = nrows;
    this.colNumber = ncols;
    this.nnz = nnz;
    this.matrixFormat = fmt;
  }

  // /**
  // * Wrap matrix existing on GPU.
  // *
  // * @param descr matrix descriptor
  // * @param rowIndPtr Pointer to rows indices vector
  // * @param colIndPtr Pointer to columns indices vector
  // * @param valPtr Pointer to values vector
  // * @param nrows number of rows (of full matrix)
  // * @param ncols number of columns
  // * @param nnz number of non zero elements
  // * @param fmt CUDA matrix format
  // * @param handle handle to library
  // * @param cublasHandle
  // * @see cusparseMatDescr
  // * @see SparseMatrixType
  // */
  // public SparseMatrixDevice(cusparseMatDescr descr, Pointer rowIndPtr, Pointer colIndPtr,
  // Pointer valPtr, int nrows, int ncols, int nnz, SparseMatrixType fmt,
  // cusparseHandle handle, cublasHandle cublasHandle) {
  // this(handle, cublasHandle);
  // initialiseMatrixStruct(descr);
  // this.rowIndPtr = rowIndPtr;
  // this.colIndPtr = colIndPtr;
  // this.valPtr = valPtr;
  // this.rowNumber = nrows;
  // this.colNumber = ncols;
  // this.nnz = nnz;
  // this.matrixFormat = fmt;
  // }

  /**
   * If set,LU analysis is run only once during first call of
   * {@link #luSolve(DenseVectorDevice, boolean, int, float)}.
   * 
   * @param useCheating the useCheating to set
   */
  public void setUseCheating(boolean useCheating) {
    this.useCheating = useCheating;
  }

  /**
   * Copy indices from device to host.
   */
  public void retrieveFromDevice() {
    int indRowSize = computeIndicesLength();
    rowInd = new int[indRowSize];
    colInd = new int[getElementNumber()];
    val = new float[getElementNumber()];
    cudaMemcpy(Pointer.to(rowInd), getRowIndPtr(), indRowSize * Sizeof.INT, cudaMemcpyDeviceToHost);
    cudaMemcpy(Pointer.to(colInd), getColIndPtr(), getElementNumber() * Sizeof.INT,
            cudaMemcpyDeviceToHost);
    cudaMemcpy(Pointer.to(val), getValPtr(), getElementNumber() * Sizeof.FLOAT,
            cudaMemcpyDeviceToHost);
  }

  /**
   * @param rowInd
   * @param colInd
   * @param val
   */
  private void transferToGpu(int[] rowInd, int[] colInd, float[] val) {
    cudaMalloc(colIndPtr, colInd.length * Sizeof.INT);
    cudaMalloc(rowIndPtr, rowInd.length * Sizeof.INT);
    cudaMalloc(valPtr, val.length * Sizeof.FLOAT);
    cudaMemcpy(rowIndPtr, Pointer.to(rowInd), rowInd.length * Sizeof.INT, cudaMemcpyHostToDevice);
    cudaMemcpy(colIndPtr, Pointer.to(colInd), colInd.length * Sizeof.INT, cudaMemcpyHostToDevice);
    cudaMemcpy(valPtr, Pointer.to(val), val.length * Sizeof.FLOAT, cudaMemcpyHostToDevice);
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

  /**
   * 
   */
  public void free() {
    // try {
    // if (useCheating) { // if false all released in code
    // cusparseDestroySolveAnalysisInfo(infoL);
    // cusparseDestroySolveAnalysisInfo(infoU);
    // }
    // } catch (Exception e) {
    // LOGGER.debug("cusparseDestroySolveAnalysisInfo already freed? " + e.getMessage());
    // }
    try {
      cudaFree(rowIndPtr);
    } catch (Exception e) {
      LOGGER.debug("rowIndPtr already freed? " + e.getMessage());
    }
    try {
      cudaFree(colIndPtr);
    } catch (Exception e) {
      LOGGER.debug("colIndPtr already freed? " + e.getMessage());
    }
    try {
      cudaFree(valPtr);
    } catch (Exception e) {
      LOGGER.debug("valPtr already freed? " + e.getMessage());
    }
    try {
      cusparseDestroyMatDescr(descr);
    } catch (Exception e) {
      LOGGER.debug("descr already freed? " + e.getMessage());
    }
    this.descr = new cusparseMatDescr();
    this.rowIndPtr = new Pointer();
    this.colIndPtr = new Pointer();
    this.valPtr = new Pointer();
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
    return nnz;
  }

  public SparseMatrixDevice convert2csr() {
    if (matrixFormat == SparseMatrixType.MATRIX_FORMAT_CSR) {
      return this;
    } else {
      Pointer rowIndPtr = new Pointer();
      cudaMalloc(rowIndPtr, (getRowNumber() + 1) * Sizeof.INT);
      cusparseXcoo2csr(cusparseHandle, getRowIndPtr(), getElementNumber(), getRowNumber(),
              rowIndPtr, CUSPARSE_INDEX_BASE_ZERO);
      return new SparseMatrixDevice(rowIndPtr, getColIndPtr(), getValPtr(), getRowNumber(),
              getColNumber(), getElementNumber(), SparseMatrixType.MATRIX_FORMAT_CSR,
              cusparseHandle, cublasHandle);
    }
  }

  public SparseMatrixDevice convert2coo() {
    if (matrixFormat == SparseMatrixType.MATRIX_FORMAT_COO) {
      return this;
    } else {
      Pointer rowIndPtr = new Pointer();
      cudaMalloc(rowIndPtr, getElementNumber() * Sizeof.INT);
      cusparseXcsr2coo(cusparseHandle, getRowIndPtr(), getElementNumber(), getRowNumber(),
              rowIndPtr, CUSPARSE_INDEX_BASE_ZERO);
      return new SparseMatrixDevice(rowIndPtr, getColIndPtr(), getValPtr(), getRowNumber(),
              getColNumber(), getElementNumber(), SparseMatrixType.MATRIX_FORMAT_COO,
              cusparseHandle, cublasHandle);
    }
  }

  public SparseMatrixDevice multiply(SparseMatrixDevice in) {
    if (this.getColNumber() != in.getRowNumber()) {
      throw new IllegalArgumentException("Incompatibile sizes");
    }
    if (in.getSparseMatrixType() != SparseMatrixType.MATRIX_FORMAT_CSR) {
      throw new IllegalArgumentException("multiply requires CSR input format.");
    }

    int m = this.getRowNumber();

    cusparseMatDescr descrOut = new cusparseMatDescr();
    cusparseCreateMatDescr(descrOut);
    cusparseSetMatType(descrOut, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrOut, CUSPARSE_INDEX_BASE_ZERO);

    Pointer nnzOutPtr = new Pointer();
    Pointer rowIndOutPtr = new Pointer();
    int nt = CUSPARSE_OPERATION_NON_TRANSPOSE;
    int n = in.getColNumber();
    int k = this.getColNumber();
    int[] nnzOut = new int[1];
    cudaMalloc(nnzOutPtr, Sizeof.INT);
    cudaMalloc(rowIndOutPtr, (m + 1) * Sizeof.INT);
    cusparseXcsrgemmNnz(cusparseHandle, nt, nt, m, n, k, this.getDescr(), this.getElementNumber(),
            this.getRowIndPtr(), this.getColIndPtr(), in.getDescr(), in.getElementNumber(),
            in.getRowIndPtr(), in.getColIndPtr(), descrOut, rowIndOutPtr, nnzOutPtr);
    JCuda.cudaDeviceSynchronize();
    Pointer colIndOutPtr = new Pointer();
    Pointer valOutPtr = new Pointer();
    cudaMemcpy(Pointer.to(nnzOut), nnzOutPtr, Sizeof.INT, cudaMemcpyDeviceToHost);
    cudaMalloc(colIndOutPtr, nnzOut[0] * Sizeof.INT);
    cudaMalloc(valOutPtr, nnzOut[0] * Sizeof.FLOAT);
    cusparseScsrgemm(cusparseHandle, nt, nt, m, n, k, this.getDescr(), this.getElementNumber(),
            this.getValPtr(), this.getRowIndPtr(), this.getColIndPtr(), in.getDescr(),
            in.getElementNumber(), in.getValPtr(), in.getRowIndPtr(), in.getColIndPtr(), descrOut,
            valOutPtr, rowIndOutPtr, colIndOutPtr);
    JCuda.cudaDeviceSynchronize();
    return new SparseMatrixDevice(rowIndOutPtr, colIndOutPtr, valOutPtr, m, n, nnzOut[0],
            SparseMatrixType.MATRIX_FORMAT_CSR, descrOut, cusparseHandle, cublasHandle);
  }

  /**
   * Download data from GPU.
   * 
   * @param isforce if true download always, otherwise only if local instances of arrays are empty.
   */
  public void toCpu(boolean isforce) {
    if (isforce || colInd == null || colInd.length == 0 || rowInd == null || rowInd.length == 0
            || val == null || val.length == 0) {
      retrieveFromDevice();
    }
  }

  /**
   * @return Transposed matrix
   */
  public SparseMatrixDevice transpose() {
    SparseMatrixDevice csrm = (SparseMatrixDevice) this.convert2csr();

    Pointer colIndPtr = new Pointer();
    cudaMalloc(colIndPtr, (csrm.getColNumber() + 1) * Sizeof.INT);
    Pointer rowIndPtr = new Pointer();
    cudaMalloc(rowIndPtr, csrm.getElementNumber() * Sizeof.INT);
    Pointer valPtr = new Pointer();
    cudaMalloc(valPtr, csrm.getElementNumber() * Sizeof.FLOAT);

    cusparseScsr2csc(cusparseHandle, csrm.getRowNumber(), csrm.getColNumber(),
            csrm.getElementNumber(), csrm.getValPtr(), csrm.getRowIndPtr(), csrm.getColIndPtr(),
            valPtr, rowIndPtr, colIndPtr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);

    // cusparseXcoo2csr(handle, getRowIndPtr(), getElementNumber(), getRowNumber(), rowIndPtr,
    // CUSPARSE_INDEX_BASE_ZERO);
    // return new SparseMatrixDevice(rowIndPtr, getColIndPtr(), getValPtr(), getRowNumber(),
    // getColNumber(), getElementNumber(), SparseMatrixType.MATRIX_FORMAT_CSR);
    return new SparseMatrixDevice(colIndPtr, rowIndPtr, valPtr, csrm.getColNumber(),
            csrm.getRowNumber(), getElementNumber(), SparseMatrixType.MATRIX_FORMAT_CSR,
            cusparseHandle, cublasHandle);

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

  public SparseMatrixDevice removeRows(int[] rows) {
    LOGGER.debug("RemoveRows run on CPU");
    SparseMatrixDevice ttmp = this.convert2coo();
    SparseMatrixDevice tmp = new SparseMatrixDevice(ttmp.getRowInd(), ttmp.getColInd(),
            ttmp.getVal(), ttmp.getRowNumber(), ttmp.getColNumber(), ttmp.getSparseMatrixType(),
            false, cusparseHandle, cublasHandle);
    // tmp is on cpu
    tmp.removeRowsIndices(rows);
    tmp.nnz = tmp.val.length;
    tmp.toGpu();
    return tmp;
  }

  public SparseMatrixDevice removeCols(int[] cols) {
    LOGGER.debug("removeCols run on CPU");
    SparseMatrixDevice ttmp = this.convert2coo();
    SparseMatrixDevice tmp = new SparseMatrixDevice(ttmp.getRowInd(), ttmp.getColInd(),
            ttmp.getVal(), ttmp.getRowNumber(), ttmp.getColNumber(), ttmp.getSparseMatrixType(),
            false, cusparseHandle, cublasHandle);
    // tmp is on cpu
    tmp.removeColsIndices(cols);
    tmp.nnz = tmp.val.length;
    tmp.toGpu();
    return tmp;
  }

  public void toGpu() {
    free(); // free old, always valid as called in constructor
    transferToGpu(rowInd, colInd, val);
  }

  /**
   * Sum along all rows.
   * 
   * <p>It is faster to remove rows first and then call this method than to use
   * {@link #sumAlongRows(Integer[])}.
   * 
   * @return sum
   */
  public DenseVectorDevice sumAlongRows() {
    this.toCpu(true);
    float[] sum = sumAlongRowsIndices();
    return new DenseVectorDevice(getRowNumber(), 1, sum);
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrix#getRowInd()
   */
  @Override
  public int[] getRowInd() {
    if (rowInd == null || rowInd.length == 0) {
      retrieveFromDevice();
    }
    return super.getRowInd();
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrix#getVal()
   */
  @Override
  public float[] getVal() {
    if (val == null || val.length == 0) {
      retrieveFromDevice();
    }
    return super.getVal();
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrix#getColInd()
   */
  @Override
  public int[] getColInd() {
    if (colInd == null || colInd.length == 0) {
      retrieveFromDevice();
    }
    return super.getColInd();
  }

  /**
   * Solve linear system. Performs LU factorisation.
   * 
   * @param b_gpuPtrAny
   * @param iLuBiCGStabSolve
   * @param iter
   * @param tol
   * @return solution
   */
  public float[] luSolve(DenseVectorDevice b_gpuPtrAny, boolean iLuBiCGStabSolve, int iter,
          float tol) {
    LOGGER.info("Starting LU solver");
    StopWatch timer = StopWatch.createStarted();
    if (getColNumber() != getRowNumber()) {
      throw new IllegalArgumentException("Left matrix must be square");
    }
    if (this.matrixFormat != SparseMatrixType.MATRIX_FORMAT_CSR) {
      throw new IllegalArgumentException("Left matrix should be in CSR");
    }
    StoppedBy stoppedReason = StoppedBy.ITERATIONS; // default assumption
    DenseVectorDevice b_gpuPtr = b_gpuPtrAny;
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
    cudaMalloc(x_gpuPtr, m * Sizeof.FLOAT); // changed to DOUBLE
    cudaMalloc(z_gpuPtr, m * Sizeof.FLOAT); // changed to DOUBLE

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

    timer.split();
    LOGGER.info("... Step 1 accomplished in " + timer.toSplitString());
    timer.unsplit();
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
    cudaMalloc(iLUcooVal_gpuPtr, nnz * Sizeof.FLOAT); // changed to DOUBLE
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

    timer.split();
    LOGGER.info("... Step 2 accomplished in " + timer.toSplitString());
    timer.unsplit();

    // step 3: query how much memory used in csrilu02 and csrsv2, and
    // allocate the buffer
    cusparseScsrilu02_bufferSize(cusparseHandle, m, nnz, descr_iLU, AcooVal_gpuPtr,
            AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, info_iLU, pBufferSize_iLU);
    cusparseScsrsv2_bufferSize(cusparseHandle, trans_L, m, nnz, descr_L, AcooVal_gpuPtr,
            AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, info_L, pBufferSize_L);
    cusparseScsrsv2_bufferSize(cusparseHandle, trans_U, m, nnz, descr_U, AcooVal_gpuPtr,
            AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, info_U, pBufferSize_U);

    pBufferSize = Math.max(pBufferSize_iLU[0], Math.max(pBufferSize_L[0], pBufferSize_U[0]));
    // System.out.println("in csrSparseMatrix.LuSolve(),buffersize = "+
    // pBufferSize);

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc(pBuffer, pBufferSize);

    timer.split();
    LOGGER.info("... Step 3 accomplished in " + timer.toSplitString());
    timer.unsplit();

    // step 4: perform analysis of incomplete Cholesky on M
    // perform analysis of triangular solve on L
    // perform analysis of triangular solve on U
    // The lower(upper) triangular part of M has the same sparsity pattern
    // as L(U),
    // we can do analysis of csrilu0 and csrsv2 simultaneously.

    cusparseScsrilu02_analysis(cusparseHandle, m, nnz, descr_iLU, AcooVal_gpuPtr,
            AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, info_iLU, policy_iLU, pBuffer);

    Pointer structural_zero = new Pointer();
    cudaMalloc(structural_zero, Sizeof.INT);

    // int[] cusparsePointerMode = new int[1];
    // default mode seems to be HOST
    // cusparseGetPointerMode(handle, cusparsePointerMode);
    // System.out.printf("Cusparse pointer mode %d \n",
    // cusparsePointerMode[0]);
    // we need to switch to DEVICE before using cusparseXcsrilu02_zeroPivot,
    // for obscure reasons, and switch back to HOST afterwards
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE);
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseXcsrilu02_zeroPivot(cusparseHandle, info_iLU,
            structural_zero)) {
      int[] sz = new int[1];
      cudaMemcpy(Pointer.to(sz), structural_zero, Sizeof.INT, cudaMemcpyDeviceToHost); // copy
                                                                                       // results
                                                                                       // back
      System.out.printf("A(%d,%d) is missing\n", sz[0], sz[0]);
    }
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);

    cusparseScsrsv2_analysis(cusparseHandle, trans_L, m, nnz, descr_L, AcooVal_gpuPtr,
            AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, info_L, policy_L, pBuffer);

    cusparseScsrsv2_analysis(cusparseHandle, trans_U, m, nnz, descr_U, AcooVal_gpuPtr,
            AcsrRowIndex_gpuPtr, AcooColIndex_gpuPtr, info_U, policy_U, pBuffer);

    timer.split();
    LOGGER.info("... Step 4 accomplished in " + timer.toSplitString());
    timer.unsplit();

    // step 5: M = L * U
    cusparseScsrilu02(cusparseHandle, m, nnz, descr_iLU, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
            iLUcooColIndex_gpuPtr, info_iLU, policy_iLU, pBuffer);

    Pointer numerical_zero = new Pointer();
    cudaMalloc(numerical_zero, Sizeof.INT);

    // same trick of switching modes needed here
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE);
    if (CUSPARSE_STATUS_ZERO_PIVOT == cusparseXcsrilu02_zeroPivot(cusparseHandle, info_iLU,
            numerical_zero)) {
      int[] nz = new int[1];
      cudaMemcpy(Pointer.to(nz), numerical_zero, Sizeof.INT, cudaMemcpyDeviceToHost); // copy
                                                                                      // results
                                                                                      // back
      System.out.printf("U(%d,%d) is zero\n", nz[0], nz[0]);
    }
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);

    timer.split();
    LOGGER.info("... Step 5 accomplished in " + timer.toSplitString());
    timer.unsplit();

    // step 6: solve L*z = x
    cusparseScsrsv2_solve(cusparseHandle, trans_L, m, nnz, Pointer.to(one_host), descr_L,
            iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, info_L,
            b_gpuPtr.getValPtr(), z_gpuPtr, policy_L, pBuffer);

    timer.split();
    LOGGER.info("... Step 6 accomplished in " + timer.toSplitString());
    timer.unsplit();

    // step 7: solve U*y = z
    cusparseScsrsv2_solve(cusparseHandle, trans_U, m, nnz, Pointer.to(one_host), descr_U,
            iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, info_U, z_gpuPtr,
            x_gpuPtr, policy_U, pBuffer);

    timer.split();
    LOGGER.info("... Step 7 accomplished in " + timer.toSplitString());
    timer.unsplit();

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

      cudaMalloc(p, m * Sizeof.FLOAT); // changed to DOUBLE
      cudaMalloc(ph, m * Sizeof.FLOAT); // changed to DOUBLE
      cudaMalloc(q, m * Sizeof.FLOAT); // changed to DOUBLE
      cudaMalloc(r, m * Sizeof.FLOAT); // changed to DOUBLE
      cudaMalloc(rw, m * Sizeof.FLOAT); // changed to DOUBLE
      cudaMalloc(s, m * Sizeof.FLOAT); // changed to DOUBLE
      cudaMalloc(t, m * Sizeof.FLOAT); // changed to DOUBLE

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
      // current errors divided by initial error

      // create the info and analyse the lower and upper triangular
      // factors
      // cusparseSolveAnalysisInfo infoL = new cusparseSolveAnalysisInfo();
      // cusparseSolveAnalysisInfo infoU = new cusparseSolveAnalysisInfo();
      // cusparseCreateSolveAnalysisInfo(infoL);
      // cusparseCreateSolveAnalysisInfo(infoU);

      // the slow one
      if (!useCheating || !done) {
        cusparseCreateSolveAnalysisInfo(infoL);
        cusparseCreateSolveAnalysisInfo(infoU);
        cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, descr_L,
                iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoL);
        cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, descr_U,
                iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr, iLUcooColIndex_gpuPtr, infoU);
        done = true;
      } else {
        LOGGER.warn("Cheating is enabled. Using structure from previous analysis.");
      }
      timer.split();
      LOGGER.debug("... Step cusparseScsrsv_analysis accomplished in " + timer.toSplitString());
      timer.unsplit();
      // 1 : compute initial residual r = b âˆ’ A x0 ( using initial guess in
      // x )
      cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
              Pointer.to(one_host), getDescr(), AcooVal_gpuPtr, AcsrRowIndex_gpuPtr,
              AcooColIndex_gpuPtr, x_gpuPtr, Pointer.to(zero_host), r);
      cublasSscal(cublashandle, n, Pointer.to(minus_one_host), r, 1);
      cublasSaxpy(cublashandle, n, Pointer.to(one_host), b_gpuPtr.getValPtr(), 1, r, 1);
      timer.split();
      LOGGER.debug("... Step cusparseScsrmv accomplished in " + timer.toSplitString());
      timer.unsplit();

      // 2 : Set p=r and \tilde{r}=r
      cublasScopy(cublashandle, n, r, 1, p, 1);
      cublasScopy(cublashandle, n, r, 1, rw, 1);
      cublasSnrm2(cublashandle, n, r, 1, Pointer.to(nrmr0));
      timer.split();
      LOGGER.debug("... Step cublasSnrm2 accomplished in " + timer.toSplitString());
      timer.unsplit();
      // 3 : repeat until convergence (based on maximum number of
      // iterations and relative residual)

      timer.split();
      LOGGER.info("... Step pre CG accomplished in " + timer.toSplitString());
      timer.unsplit();

      for (int i = 0; i < iter; i++) {

        System.out.print("Iteration " + i + "\r");

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
        cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n,
                Pointer.to(one_host), descr_L, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
                iLUcooColIndex_gpuPtr, infoL, p, t);

        cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n,
                Pointer.to(one_host), descr_U, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
                iLUcooColIndex_gpuPtr, infoU, t, ph);

        // 1 6 : q = A \ hat{p} ( sparse matrixâˆ’vector multiplication )
        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
                Pointer.to(one_host), getDescr(), AcooVal_gpuPtr, AcsrRowIndex_gpuPtr,
                AcooColIndex_gpuPtr, ph, Pointer.to(zero_host), q);

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
          stoppedReason = StoppedBy.RELERR;
          break;
        }
        // 2 3 : M \ hat{ s } = r ( sparse lower and upper triangular
        // solves )

        cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n,
                Pointer.to(one_host), descr_L, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
                iLUcooColIndex_gpuPtr, infoL, r, t);

        cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n,
                Pointer.to(one_host), descr_U, iLUcooVal_gpuPtr, iLUcsrRowIndex_gpuPtr,
                iLUcooColIndex_gpuPtr, infoU, t, s);

        // 2 4 : t = A \ hat{ s } ( sparse matrixâˆ’vector multiplication
        // )

        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
                Pointer.to(one_host), getDescr(), AcooVal_gpuPtr, AcsrRowIndex_gpuPtr,
                AcooColIndex_gpuPtr, s, Pointer.to(zero_host), t);

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
          stoppedReason = StoppedBy.RELERR;
          break;
        }

        LOGGER.debug("nrmr: " + nrmr[0] + " nrmr0: " + nrmr0[0] + " alpha: " + alpha[0] + " beta: "
                + beta[0] + " rho: " + rho[0] + " temp: " + temp[0] + " temp2: " + temp2[0]
                + " omega: " + omega[0]);

      }
      System.out.println();
      LOGGER.info("Solver stopped by " + stoppedReason);

      cudaFree(p);
      cudaFree(ph);
      cudaFree(q);
      cudaFree(r);
      cudaFree(rw);
      cudaFree(s);
      cudaFree(t);
      if (!useCheating) {
        // note that this should be called in order with other allocations. This is why destroying
        // in free() does not work. To prevent memory leaks on GPU device is reseted on each job.
        cusparseDestroySolveAnalysisInfo(infoL);
        cusparseDestroySolveAnalysisInfo(infoU);
      }

      cublasDestroy(cublashandle);

    } // CG routine
      // /needs changing
    float result_host[] = new float[m]; // array to hold results

    // JCuda.cudaDeviceSynchronize();
    // copy results back
    cudaMemcpy(Pointer.to(result_host), x_gpuPtr, m * Sizeof.FLOAT, cudaMemcpyDeviceToHost);

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
    timer.stop();
    LOGGER.info("LU solver finished in " + timer.toString());
    return result_host;
  }

  /**
   * Return lower triangle from this matrix. Assume this symmetric.
   * 
   * @return lower triangle
   */
  public SparseMatrixDevice getLowerTriangle() {
    StopWatch timer = StopWatch.createStarted();
    LOGGER.info("Extracting lower triangle from symmetric L");
    // helpers
    // cut lower triangle from this array (on CPU)
    this.toCpu(false);
    int nnzA = this.getElementNumber();
    int rowsA = this.getRowNumber();
    int[] h_csrRowPtrA = this.getRowInd();
    int[] h_csrColIndA = this.getColInd();
    float[] h_csrValA = this.getVal();
    int LOWER = (int) (0.5 * (nnzA + rowsA));
    float[] h_val = new float[LOWER];
    int[] h_col = new int[LOWER];
    int[] h_row = new int[rowsA + 1];
    // populate lower triangular column indices and row offsets for zero fill-in IC
    h_row[rowsA] = LOWER;
    int k = 0;
    for (int i = 0; i < rowsA; i++) {
      h_row[i] = k;
      int numRowElements = h_csrRowPtrA[i + 1] - h_csrRowPtrA[i];
      int m = 0;
      for (int j = 0; j < numRowElements; j++) {
        if (!(h_csrColIndA[h_csrRowPtrA[i] + j] > i)) {
          h_col[h_row[i] + m] = h_csrColIndA[h_csrRowPtrA[i] + j];
          h_val[h_row[i] + m] = h_csrValA[h_csrRowPtrA[i] + j];
          k++;
          m++;
        }
      }
    }
    cusparseMatDescr descrA = new cusparseMatDescr(); // description of triangle A
    cusparseCreateMatDescr(descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    Pointer d_csrRowPtrA = ArrayTools.cudaMallocCopy(h_row, rowsA + 1);
    Pointer d_csrColIndA = ArrayTools.cudaMallocCopy(h_col, LOWER);
    Pointer d_csrValA = ArrayTools.cudaMallocCopy(h_val, LOWER);
    // h_row. h_col. h_val are CPU coordinates of lower triangle of this matrix
    timer.stop();
    LOGGER.debug("getLowerTriangle accomplished in " + timer.toString());
    return new SparseMatrixDevice(d_csrRowPtrA, d_csrColIndA, d_csrValA, rowsA, LOWER, LOWER,
            SparseMatrixType.MATRIX_FORMAT_CSR, descrA, cusparseHandle, cublasHandle);
  }

  /**
   * @return Cholesky decomposition.
   */
  public DenseVectorDevice getCholesky() {
    StopWatch timer = StopWatch.createStarted();
    LOGGER.info("Performing Cholesky decomposition");
    Pointer d_valChol = new Pointer(); // values of lower triangle, will be modified
    cudaMalloc(d_valChol, Sizeof.FLOAT * colNumber);
    cudaMemcpy(d_valChol, getValPtr(), Sizeof.FLOAT * colNumber, cudaMemcpyDeviceToDevice);
    // Cholesky factorization
    cusparseSolveAnalysisInfo infoA = new cusparseSolveAnalysisInfo();
    cusparseCreateSolveAnalysisInfo(infoA);
    cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, rowNumber, colNumber,
            descr, getValPtr(), getRowIndPtr(), getColIndPtr(), infoA);
    timer.split();
    LOGGER.debug("... Step cusparseScsrsv_analysis accomplished in " + timer.toSplitString());
    timer.unsplit();
    cusparseScsric0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, rowNumber, descr, d_valChol, // will
            getRowIndPtr(), getColIndPtr(), infoA);
    cusparseDestroySolveAnalysisInfo(infoA);
    timer.stop();
    LOGGER.debug("... Step cusparseScsric0 accomplished in " + timer.toString());
    return new DenseVectorDevice(nnz, 1, d_valChol);
  }

  // public static cusparseMatDescr analyseU(cusparseHandle cusparseHandle, DenseVectorDevice chol)
  // {
  // StopWatch timer = StopWatch.createStarted();
  // Pointer d_valChol = chol.getValPtr();
  // cusparseMatDescr descrL = new cusparseMatDescr();
  // cusparseCreateMatDescr(descrL);
  // cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
  // cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
  // cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_NON_UNIT);
  // cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
  // cusparseSolveAnalysisInfo infoL = new cusparseSolveAnalysisInfo();
  // cusparseCreateSolveAnalysisInfo(infoL);
  // cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
  // b_gpuPtrAny.getRowNumber(), getColNumber(), descrL, d_valChol, getRowIndPtr(),
  // getColIndPtr(), infoL);
  // timer.stop();;
  // LOGGER.debug("... Step cusparseScsrsv_analysis accomplished in " + timer.toString());
  // }
  /**
   * For symmetric matrices, performs Cholesky factorization. Require Lower triangle only.
   * 
   * @param b_gpuPtrAny RHS
   * @param chol Cholesky decomposition or null to calculate it in place.
   * @param iter max number of iter
   * @param tol tolerance
   * @return solution
   */
  public float[] luSolveSymmetric(DenseVectorDevice b_gpuPtrAny, DenseVectorDevice chol, int iter,
          float tol) {
    LOGGER.info("Starting LU solver");
    StopWatch timer = StopWatch.createStarted();
    if (this.matrixFormat != SparseMatrixType.MATRIX_FORMAT_CSR) {
      throw new IllegalArgumentException("Left matrix should be in CSR");
    }
    // TODO test if this is triangle (from cusparseMatDescr)
    if (chol == null) {
      chol = getCholesky();
    }

    float[] h_x = new float[rowNumber]; // result
    Pointer d_x = ArrayTools.cudaMallocCopy(h_x, rowNumber);
    Pointer d_b = b_gpuPtrAny.getValPtr();

    // analyse L and U part of Cholesky result
    Pointer d_valChol = chol.getValPtr();
    cusparseMatDescr descrL = new cusparseMatDescr();
    cusparseCreateMatDescr(descrL);
    cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
    cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);

    // this is pernamently turned on for Chol
    if (!useCheating || !done) {
      cusparseCreateSolveAnalysisInfo(infoL);
      cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
              b_gpuPtrAny.getRowNumber(), getColNumber(), descrL, d_valChol, getRowIndPtr(),
              getColIndPtr(), infoL);
      timer.split();
      LOGGER.debug("... Step cusparseScsrsv_analysis accomplished in " + timer.toSplitString());
      timer.unsplit();

      cusparseCreateSolveAnalysisInfo(infoU);
      cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, getRowNumber(),
              getColNumber(), descrL, d_valChol, getRowIndPtr(), getColIndPtr(), infoU);
      timer.split();
      LOGGER.debug("... Step cusparseScsrsv_analysis accomplished in " + timer.toSplitString());
      timer.unsplit();
      done = true;
    }

    Pointer d_t = ArrayTools.cudaMallocCopy(new float[0], getRowNumber());
    Pointer d_p = ArrayTools.cudaMallocCopy(new float[0], getRowNumber());
    Pointer d_q = ArrayTools.cudaMallocCopy(new float[0], getRowNumber());
    Pointer d_z = ArrayTools.cudaMallocCopy(new float[0], getRowNumber());
    Pointer d_r = ArrayTools.cudaMallocCopy(new float[0], getRowNumber());
    float[] zero = new float[] { 0.0f };
    float[] one = new float[] { 1.0f };
    float[] negone = new float[] { -1.0f };
    float[] rho = new float[1];
    float[] rhop = new float[1];
    float[] normr0 = new float[1];
    float[] normr = new float[1];
    float[] beta = new float[1];
    float[] ptAp = new float[1];
    float[] alpha = new float[1];
    float[] negalpha = new float[1];
    //!>
    cusparseScsrmv(cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            getRowNumber(),
            getRowNumber(),
            getColNumber(),
            Pointer.to(one),
            descr,
            getValPtr(),
            getRowIndPtr(),
            getColIndPtr(),
            d_x,
            Pointer.to(zero),
            d_r);
    //!<
    timer.split();
    LOGGER.debug("... Step cusparseScsrmv accomplished in " + timer.toSplitString());
    timer.unsplit();

    cublasSscal(cublasHandle, getRowNumber(), Pointer.to(negone), d_r, 1);
    cublasSaxpy(cublasHandle, getRowNumber(), Pointer.to(one), d_b, 1, d_r, 1);
    cublasSdot(cublasHandle, getRowNumber(), d_r, 1, d_z, 1, Pointer.to(rho));
    cublasSnrm2(cublasHandle, getRowNumber(), d_r, 1, Pointer.to(normr0));
    normr[0] = normr0[0];
    timer.split();
    LOGGER.debug("... Step cublas accomplished in " + timer.toSplitString());
    timer.unsplit();
    //!>
    int i = 0;
    while (normr[0]/normr0[0] > tol && i < iter) {
      System.out.print("Iteration " + i + "\r");
      cusparseScsrsv_solve(cusparseHandle,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              getRowNumber(),
              Pointer.to(one),
              descrL,
              d_valChol,
              getRowIndPtr(),
              getColIndPtr(),
              infoL,
              d_r,
              d_t);
      cusparseScsrsv_solve(cusparseHandle,
              CUSPARSE_OPERATION_TRANSPOSE,
              getRowNumber(),
              Pointer.to(one),
              descrL,
              d_valChol,
              getRowIndPtr(),
              getColIndPtr(),
              infoU,
              d_t,
              d_z);
      rhop[0] = rho[0];
      cublasSdot(cublasHandle, getRowNumber(), d_r, 1, d_z, 1, Pointer.to(rho));
      if (i == 0) {
        cublasScopy(cublasHandle, getRowNumber(), d_z, 1, d_p, 1);
      } else {
        beta[0] = rho[0]/rhop[0];
        cublasSaxpy(cublasHandle, getRowNumber(), Pointer.to(beta), d_p, 1, d_z, 1);
        cublasScopy(cublasHandle, getRowNumber(), d_z, 1, d_p, 1);
      }
      cusparseScsrmv(cusparseHandle,
              CUSPARSE_OPERATION_NON_TRANSPOSE,
              getRowNumber(),
              getRowNumber(),
              getColNumber(),
              Pointer.to(one),
              descr,
              getValPtr(),
              getRowIndPtr(),
              getColIndPtr(),
              d_p,
              Pointer.to(zero),
              d_q);
      cublasSdot(cublasHandle, getRowNumber(), d_p, 1, d_q, 1, Pointer.to(ptAp));
      alpha[0] = rho[0] / ptAp[0];
      cublasSaxpy(cublasHandle, getRowNumber(), Pointer.to(alpha), d_p, 1, d_x, 1);
      negalpha[0] = -alpha[0];
      cublasSaxpy(cublasHandle, getRowNumber(), Pointer.to(negalpha), d_q, 1, d_r, 1);
      cublasSnrm2(cublasHandle, getRowNumber(), d_r, 1, Pointer.to(normr));
      LOGGER.debug("Iter: "+ i + " Relerr: "+normr[0]/normr0[0]);
      i++;
    }
    //!<
    if (i >= iter) {
      LOGGER.info("Solver stopped by " + StoppedBy.ITERATIONS);
    } else {
      LOGGER.info("Solver stopped by " + StoppedBy.RELERR);
    }
    cudaMemcpy(Pointer.to(h_x), d_x, Sizeof.FLOAT * getRowNumber(), cudaMemcpyDeviceToHost);

    cudaFree(d_t);
    cudaFree(d_p);
    cudaFree(d_q);
    cudaFree(d_z);
    cudaFree(d_r);
    cudaFree(d_x);
    cusparseDestroyMatDescr(descrL);
    if (!useCheating) {
      cusparseDestroySolveAnalysisInfo(infoU);
      cusparseDestroySolveAnalysisInfo(infoL);
    }
    timer.stop();
    LOGGER.info("LU solver finished in " + timer.toString());
    return h_x;
  }

  /**
   * Produce instance of {@link SparseMatrixDevice}.
   * 
   * @param raw COO matrix on the CPU
   * @param handle cusparse handle
   * @param cublasHandle cublas handle
   * @return instance
   */
  public static SparseMatrixDevice factory(SparseCoordinates raw, cusparseHandle handle,
          cublasHandle cublasHandle) {
    return new SparseMatrixDevice(raw.getRowInd(), raw.getColInd(), raw.getVal(),
            raw.getRowNumber(), raw.getColNumber(), SparseMatrixType.MATRIX_FORMAT_COO, handle,
            cublasHandle);
  }

}
