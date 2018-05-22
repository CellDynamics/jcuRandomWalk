package com.github.celldynamics.jcudarandomwalk.matrices;

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
  protected double[] val; // value
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
  public double[] getVal() {
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
    colNumber = IntStream.of(getColInd()).max().getAsInt() + 1; // assuming 0 based
    rowNumber = IntStream.of(getRowInd()).max().getAsInt() + 1;
  }

  /**
   * Convert sparse coordinates to full matrix.
   * 
   * <p>Only small matrixes. Column ordered. [col][row] like x,y
   * 
   * @return full 2D matrix [cols][rows]
   */
  public double[][] full() {
    updateDimension();
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
}
