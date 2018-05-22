package com.github.celldynamics.jcudarandomwalk.matrices;

import org.apache.commons.lang3.NotImplementedException;

/**
 * Structure for holding coordinates for sparse matrices.
 * 
 * <p>The coordinates are stored in three separate vectors for x, y and value. Coordinates are
 * 0-based. MAtrix format is COO.
 * 
 * @author baniu
 *
 */
public class SparseMatrixHost extends SparseMatrix implements IStoredOnCpu {
  private int i = 0; // counter

  /**
   * Create empty storage for specified number of sparse elements. Format COO
   * 
   * @param size size of the storage
   */
  public SparseMatrixHost(int size) {
    this.nnz = size;
    rowInd = new int[size];
    colInd = new int[size];
    val = new double[size];
    matrixFormat = SparseMatrixType.MATRIX_FORMAT_COO;
  }

  /**
   * Create sparse matrix from indices. Note that arrays are not copied.
   * 
   * @param rowInd rows
   * @param colInd columns
   * @param val values
   * @param type type of sparse matrix
   */
  public SparseMatrixHost(int[] rowInd, int[] colInd, double[] val, SparseMatrixType type) {
    switch (type) {
      case MATRIX_FORMAT_COO:
        if ((rowInd.length != colInd.length) || (rowInd.length != val.length)) {
          throw new IllegalArgumentException(
                  "Input arrays should have the same length in COO format");
        }
        this.rowInd = rowInd;
        this.colInd = colInd;
        this.val = val;
        this.nnz = rowInd.length;
        this.i = this.nnz; // to block adding
        break;
      default:
        throw new NotImplementedException("This format is not implemented yet");
    }

  }

  /**
   * Add entry (coordinates and value) to store. Matrix is created in CCO format.
   * 
   * @param r row coordinate
   * @param c column coordinate
   * @param v value stored under [r,c]
   * @throws ArrayIndexOutOfBoundsException
   */
  public void add(int r, int c, double v) {
    rowInd[i] = r;
    colInd[i] = c;
    val[i] = v;
    i++;
  }

  @Override
  public ISparseMatrix convert2csr() {
    // consider this in abstract with conversion through host
    throw new NotImplementedException("this is not supported");
  }

  @Override
  public ISparseMatrix convert2coo() {
    // consider this in abstract with conversion through host
    throw new NotImplementedException("this is not supported");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrixCpu#toGpu()
   */
  @Override
  public ISparseMatrix toGpu() {
    return new SparseMatrixDevice(getRowInd(), getColInd(), getVal(), matrixFormat);
  }

  @Override
  public ISparseMatrix multiply(ISparseMatrix in) {
    throw new NotImplementedException("this is not supported");
  }

}