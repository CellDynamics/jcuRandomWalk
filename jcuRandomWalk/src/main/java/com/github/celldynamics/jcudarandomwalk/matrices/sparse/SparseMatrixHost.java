package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import java.security.InvalidParameterException;
import java.util.Arrays;
import java.util.stream.IntStream;

import org.apache.commons.lang3.NotImplementedException;

import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.dense.IDenseVector;

/**
 * Structure for holding coordinates for sparse matrices.
 * 
 * <p>The coordinates are stored in three separate vectors for x, y and value. Coordinates are
 * 0-based. MAtrix format is COO.
 * 
 * @author baniu
 *
 */
public class SparseMatrixHost extends SparseMatrix {

  private int counter = 0; // counter

  /**
   * Create empty matrix of size 0.
   */
  public SparseMatrixHost() {
    this(0);
  }

  /**
   * Create empty storage for specified number of sparse elements. Format COO
   * 
   * @param size size of the storage
   * @see #add(int, int, double)
   * @see #updateDimension()
   */
  public SparseMatrixHost(int size) {
    this.nnz = size;
    rowInd = new int[size];
    colInd = new int[size];
    val = new float[size];
    matrixFormat = SparseMatrixType.MATRIX_FORMAT_COO;
  }

  /**
   * Create sparse matrix from indices. Note that arrays are not copied. Number of rows and columns
   * is computed automatically.
   * 
   * <p>Note that this constructor will remove any 0 filled rows or columns which might not be
   * correct.
   * 
   * @param rowInd rows
   * @param colInd columns
   * @param val values
   * @param type type of sparse matrix
   */
  public SparseMatrixHost(int[] rowInd, int[] colInd, float[] val, SparseMatrixType type) {
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
        this.counter = this.nnz; // to block adding
        break;
      default:
        throw new NotImplementedException("This format is not implemented yet");
    }
    updateDimension();
  }

  /**
   * Create sparse matrix from indices. Note that arrays are not copied.
   * 
   * @param rowInd rows
   * @param colInd columns
   * @param val values
   * @param rowNumber number of rows
   * @param colNumber number of columns
   * @param type type of sparse matrix
   */
  public SparseMatrixHost(int[] rowInd, int[] colInd, float[] val, int rowNumber, int colNumber,
          SparseMatrixType type) {
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
        this.counter = this.nnz; // to block adding
        this.rowNumber = rowNumber;
        this.colNumber = colNumber;
        break;
      default:
        throw new NotImplementedException("This format is not implemented yet");
    }
  }

  /**
   * Add entry (coordinates and value) to store. Matrix is created in CCO format.
   * 
   * <p>This method does not update {@link #getRowNumber()} or {@link #getColNumber()}. The
   * {@link #updateDimension()} must be called explicitly and the end.
   * 
   * @param r row coordinate
   * @param c column coordinate
   * @param v value stored under [r,c]
   */
  public void add(int r, int c, float v) {
    rowInd[counter] = r;
    colInd[counter] = c;
    val[counter] = v;
    counter++;
  }

  @Override
  public ISparseMatrix convert2csr() {
    // consider this in abstract with conversion through host
    if (matrixFormat == SparseMatrixType.MATRIX_FORMAT_CSR) {
      return this;
    } else {
      throw new NotImplementedException("this is not supported");
    }
  }

  @Override
  public ISparseMatrix convert2coo() {
    // consider this in abstract with conversion through host
    if (matrixFormat == SparseMatrixType.MATRIX_FORMAT_COO) {
      return this;
    } else {
      throw new NotImplementedException("this is not supported");
    }
  }

  @Override
  public IMatrix multiply(IMatrix in) {
    throw new NotImplementedException("this is not supported");
  }

  @Override
  public ISparseMatrix transpose() {
    throw new NotImplementedException("this is not supported");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#removeRows(int[])
   */
  @Override
  public IMatrix removeRows(int[] rows) {
    LOGGER.trace("Removing " + rows.length + " rows");
    if (matrixFormat != SparseMatrixType.MATRIX_FORMAT_COO) {
      throw new InvalidParameterException(
              "This matrix must be in COO format. Perform explicit conversion.");
    }
    // any 1 at index i in this array stand for index i to remove from Lap
    int[] toRem = new int[this.getRowNumber()];
    for (int s = 0; s < rows.length; s++) {
      toRem[rows[s]] = 1;
    }
    // iterate over indices lists and mark those to remove by -1 - ROWS
    int[] rowInd = Arrays.copyOf(this.getRowInd(), this.getRowInd().length);
    int[] colInd = Arrays.copyOf(this.getColInd(), this.getColInd().length);
    for (int i = 0; i < rowInd.length; i++) {
      if (toRem[rowInd[i]] > 0) {
        rowInd[i] = -1; // to remove
      }
    }
    // compute number of nonzero elements that remains
    // ... and find how many rows is >=0 - valid rows
    int remainingRow = 0;
    for (int i = 0; i < rowInd.length; i++) {
      if (rowInd[i] >= 0) {
        remainingRow++;
      }
    }
    int[] newRowInd = new int[remainingRow];
    int[] newColInd = new int[remainingRow];
    float[] newVal = new float[remainingRow];
    int l = 0;
    for (int i = 0; i < this.getElementNumber(); i++) {
      if (rowInd[i] < 0) {
        continue;
      }
      newRowInd[l] = rowInd[i];
      newColInd[l] = colInd[i];
      newVal[l] = this.getVal()[i];
      l++;
    }
    compressIndices(toRem, newRowInd);

    ISparseMatrix reducedL;
    reducedL = SparseMatrix.sparseMatrixFactory(this, newRowInd, newColInd, newVal,
            this.getRowNumber() - rows.length, this.getColNumber(),
            SparseMatrixType.MATRIX_FORMAT_COO);
    LOGGER.trace("Rows removed");
    return reducedL;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#removeCols(int[])
   */
  @Override
  public IMatrix removeCols(int[] cols) {
    LOGGER.trace("Removing " + cols.length + " cols");
    if (matrixFormat != SparseMatrixType.MATRIX_FORMAT_COO) {
      throw new InvalidParameterException(
              "This matrix must be in COO format. Perform explicit conversion.");
    }
    // any 1 at index i in this array stand for index i to remove from Lap
    int[] toRem = new int[this.getColNumber()];
    for (int s = 0; s < cols.length; s++) {
      toRem[cols[s]] = 1;
    }
    // iterate over indices lists and mark those to remove by -1 - ROWS
    int[] rowInd = Arrays.copyOf(this.getRowInd(), this.getRowInd().length);
    int[] colInd = Arrays.copyOf(this.getColInd(), this.getColInd().length);
    for (int i = 0; i < rowInd.length; i++) {
      if (toRem[colInd[i]] > 0) {
        colInd[i] = -1; // to remove
      }
    }
    // compute number of nonzero elements that remains
    // ... and find how many rows is >=0 - valid rows
    int remainingCol = 0;
    for (int i = 0; i < colInd.length; i++) {
      if (colInd[i] >= 0) {
        remainingCol++;
      }
    }
    int[] newRowInd = new int[remainingCol];
    int[] newColInd = new int[remainingCol];
    float[] newVal = new float[remainingCol];
    int l = 0;
    for (int i = 0; i < this.getElementNumber(); i++) {
      if (colInd[i] < 0) {
        continue;
      }
      newRowInd[l] = rowInd[i];
      newColInd[l] = colInd[i];
      newVal[l] = this.getVal()[i];
      l++;
    }
    compressIndices(toRem, newColInd);

    ISparseMatrix reducedL;
    reducedL = SparseMatrix.sparseMatrixFactory(this, newRowInd, newColInd, newVal,
            this.getRowNumber(), this.getColNumber() - cols.length,
            SparseMatrixType.MATRIX_FORMAT_COO);
    LOGGER.trace("Cols removed");
    return reducedL;
  }

  /**
   * Removes empty rows and columns from sparse matrix. Should be called after removing rows or
   * columns.
   * 
   * @see SparseMatrixHost#removeCols(int[])
   * @see SparseMatrixHost#removeRows(int[])
   * @deprecated Not fully tested and not used
   */
  public void compressSparseIndices() {
    int[] ri = this.getRowInd();
    int[] ci = this.getColInd();
    float[] v = this.getVal();
    int[] rsum = new int[this.getRowNumber()];
    int[] csum = new int[this.getColNumber()];

    double[] ret = new double[this.getRowNumber()];
    for (int i = 0; i < ri.length; i++) { // along all row indices
      ret[ri[i]] += v[i]; // sum all elements from the same row
    }
    for (int i = 0; i < ret.length; i++) { // along all row indices
      if (ret[i] == 0) {
        rsum[i] = 1;
      }
    }

    ret = new double[this.getColNumber()];
    for (int i = 0; i < ci.length; i++) { // along all row indices
      ret[ci[i]] += v[i]; // sum all elements from the same row
    }
    for (int i = 0; i < ret.length; i++) { // along all col indices
      if (ret[i] == 0) {
        csum[i] = 1;
      }
    }
    int[] rcomp = compressIndices(rsum, ri);
    int[] ccomp = compressIndices(csum, ci);
    rowInd = rcomp;
    colInd = ccomp;
    updateDimension();
  }

  /**
   * Compress sparse indices, removing gaps.
   *
   * @param toRem array with "1" at positions to be removed. This array is modified.
   * @param newRowInd array to be processed. This array is modified and stands like an output
   * @return Array with compressed indices
   */
  private int[] compressIndices(int[] toRem, int[] newRowInd) {
    LOGGER.trace("Compressing indices. Removing from array of size of" + newRowInd.length);
    // compress
    // after removing indices from newColInd/RowInd it contains only valid nonzero elements
    // (without
    // those from deleted rows and
    // cols) but indexes contain gaps, e.g. if 2nd column was removed newColInd will keep next
    // column after as third whereas it should be moved to left and become the second
    // because we assumed square matrix we will go through toRem array and check which indexes were
    // removed (marked by 1 at index i - removed) and then decrease all indexes larger than those
    // removed in newColInd/newRowInd by one to shift them

    // cumSum over toRem
    // from array like this 0 0 0 1 1 0 1 0 0 -> 0 0 0 1 2 2 3 3 3
    int cumSum = 0;
    for (int i = 0; i < toRem.length; i++) {
      if (toRem[i] > 0) {
        toRem[i] += cumSum;
        cumSum++;
      } else {
        toRem[i] = cumSum;
      }
    }
    // int[] newRowIndcp = Arrays.copyOf(newRowInd, newRowInd.length);
    // array newRowIndcp contains indexes, whose match indexes of toRem array.
    // value over index i in toRem shows how much this index (as value in newRowIndcp should be
    // shifted)
    for (int i = 0; i < newRowInd.length; i++) {
      newRowInd[i] -= toRem[newRowInd[i]];
    }
    LOGGER.trace("Indices compressed");
    return newRowInd;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#free()
   */
  @Override
  public void free() {
    // do nothing
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#sumAlongRows()
   */
  @Override
  public IMatrix sumAlongRows() {
    int[] ri = this.getRowInd();
    float[] v = this.getVal();
    float[] ret = new float[this.getRowNumber()];
    for (int i = 0; i < ri.length; i++) { // along all row indices
      ret[ri[i]] += v[i]; // sum all elements from the same row
    }
    int[] ciret = new int[ret.length];
    int[] riret = IntStream.range(0, ret.length).toArray();
    Arrays.fill(ciret, 0);

    return SparseMatrix.sparseMatrixFactory(this, riret, ciret, ret, this.getRowNumber(), 1,
            SparseMatrixType.MATRIX_FORMAT_COO);
  }

  @Override
  public float[] luSolve(IDenseVector b_gpuPtr, boolean iLuBiCGStabSolve, int iter, float tol) {
    return val;
    // LOGGER.warn("luSolve run on GPU");
    // ISparseMatrix gp = this.toGpu();
    // float[] ret = gp.luSolve(b_gpuPtr, iLuBiCGStabSolve, iter, tol);
    // gp.free();
    // return ret;
  }

  @Override
  public double[][] full() {
    // TODO Auto-generated method stub
    return null;
  }

}