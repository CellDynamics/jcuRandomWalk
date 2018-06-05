package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Common base for CPU (base type) and GPU.
 * 
 * @author p.baniukiewicz
 *
 */
public abstract class SparseMatrix implements ISparseMatrix {

  /**
   * The Constant LOGGER.
   */
  static final Logger LOGGER = LoggerFactory.getLogger(SparseMatrix.class.getName());

  protected SparseMatrixType matrixFormat = SparseMatrixType.MATRIX_FORMAT_COO;
  protected int nnz; // number of nonzero elements

  protected int[] rowInd; // rows
  protected int[] colInd; // cols
  protected float[] val; // value
  protected int rowNumber; // number of rows
  protected int colNumber; // number of cols

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrix#getRowInd()
   */
  @Override
  public int[] getRowInd() {
    if (rowInd == null && this instanceof SparseMatrixDevice) {
      ((SparseMatrixDevice) this).retrieveFromDevice();
    }
    return rowInd;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrix#getColInd()
   */
  @Override
  public int[] getColInd() {
    if (colInd == null && this instanceof SparseMatrixDevice) {
      ((SparseMatrixDevice) this).retrieveFromDevice();
    }
    return colInd;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getVal()
   */
  @Override
  public float[] getVal() {
    if (val == null && this instanceof SparseMatrixDevice) {
      ((SparseMatrixDevice) this).retrieveFromDevice();
    }
    return val;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getElementNumber()
   */
  @Override
  public int getElementNumber() {
    return nnz;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrix#getSparseMatrixType()
   */
  @Override
  public SparseMatrixType getSparseMatrixType() {
    return matrixFormat;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getRowNumber()
   */
  @Override
  public int getRowNumber() {
    return rowNumber;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getColNumber()
   */
  @Override
  public int getColNumber() {
    return colNumber;
  }

  /**
   * Find maximum row nad column number and set number of rows and columns respectively.
   * 
   * <p>This method can be used for updating number of rows and columns if they changed but its
   * execution is expensive.
   */
  public void updateDimension() {
    // TODO see 9.3. cusparse<t>csrgemm() to get nnz and rows number?
    colNumber = IntStream.of(getColInd()).parallel().max().getAsInt() + 1; // assuming 0 based
    rowNumber = IntStream.of(getRowInd()).parallel().max().getAsInt() + 1;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#full()
   */
  public double[][] full() {
    if (getColNumber() == 0 || getRowNumber() == 0) {
      updateDimension();
    }
    int ncols = getColNumber();
    int nrows = getRowNumber();
    if (nrows * ncols > 1e5) {
      LOGGER.warn("Sparse matrix is large (" + nrows + "," + ncols + ")");
    }
    double[][] ret = new double[ncols][];
    for (int c = 0; c < ncols; c++) {
      ret[c] = new double[nrows];
    }
    for (int l = 0; l < getElementNumber(); l++) {
      ret[getColInd()[l]][getRowInd()[l]] = getVal()[l];
    }
    return ret;
  }

  /*
   * (non-Javadoc)
   * 
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString() {
    return "SparseMatrix [matrixFormat=" + matrixFormat + ", rowNumber=" + rowNumber
            + ", colNumber=" + colNumber + ", nnz=" + nnz + ", rowInd=" + Arrays.toString(rowInd)
            + ", colInd=" + Arrays.toString(colInd) + ", val=" + Arrays.toString(val) + "]";
  }

  /**
   * Produce sparse matrix of the same type as specified from raw data.
   * 
   * @param type type of output matrix, can be even empty dummy object
   * @param rowInd indices of rows
   * @param colInd indices of columns
   * @param val values
   * @param matrixInputFormat matrix format
   * @return matrix of type 'type' from above arrays
   */
  public static ISparseMatrix sparseMatrixFactory(ISparseMatrix type, int[] rowInd, int[] colInd,
          float[] val, SparseMatrixType matrixInputFormat) {
    if (rowInd.length == 0 || colInd.length == 0 || val.length == 0) {
      throw new IllegalArgumentException(
              "One or more arrays passed to sparseMatrixFactory are 0-sized");
    }
    Class<? extends ISparseMatrix> classToLoad = type.getClass();
    Class<?>[] carg = new Class[4]; // Our constructor has 4 arguments
    carg[0] = int[].class;
    carg[1] = int[].class;
    carg[2] = float[].class;
    carg[3] = SparseMatrixType.class;
    try {
      return (ISparseMatrix) classToLoad.getDeclaredConstructor(carg).newInstance(rowInd, colInd,
              val, matrixInputFormat);
    } catch (InstantiationException | IllegalAccessException | IllegalArgumentException
            | InvocationTargetException | NoSuchMethodException | SecurityException e) {
      throw new IllegalArgumentException("Can not create object instance: " + e.getMessage());
    }
  }

  /**
   * Produce sparse matrix of the same type as specified from raw data.
   * 
   * @param type type of output matrix, can be even empty dummy object
   * @param rowInd indices of rows
   * @param colInd indices of columns
   * @param val values
   * @param rowNumber number of rows
   * @param colNumber number of columns
   * @param matrixInputFormat matrix format
   * @return matrix of type 'type' from above arrays
   */
  public static ISparseMatrix sparseMatrixFactory(ISparseMatrix type, int[] rowInd, int[] colInd,
          float[] val, int rowNumber, int colNumber, SparseMatrixType matrixInputFormat) {
    if (rowInd.length == 0 || colInd.length == 0 || val.length == 0) {
      throw new IllegalArgumentException(
              "One or more arrays passed to sparseMatrixFactory are 0-sized");
    }
    Class<? extends ISparseMatrix> classToLoad = type.getClass();
    Class<?>[] carg = new Class[6]; // Our constructor has 4 arguments
    carg[0] = int[].class;
    carg[1] = int[].class;
    carg[2] = float[].class;
    carg[3] = int.class;
    carg[4] = int.class;
    carg[5] = SparseMatrixType.class;
    try {
      return (ISparseMatrix) classToLoad.getDeclaredConstructor(carg).newInstance(rowInd, colInd,
              val, rowNumber, colNumber, matrixInputFormat);
    } catch (InstantiationException | IllegalAccessException | IllegalArgumentException
            | InvocationTargetException | NoSuchMethodException | SecurityException e) {
      throw new IllegalArgumentException("Can not create object instance: " + e.getMessage());
    }
  }

}
