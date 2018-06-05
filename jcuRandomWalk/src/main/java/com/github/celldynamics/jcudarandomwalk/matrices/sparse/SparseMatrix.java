package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import java.lang.reflect.InvocationTargetException;
import java.security.InvalidParameterException;
import java.util.Arrays;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;

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

  protected int rowNumber; // number of rows
  protected int colNumber; // number of cols
  protected int nnz; // number of nonzero elements
  protected int[] rowInd; // rows
  protected int[] colInd; // cols
  protected float[] val; // value

  protected SparseMatrixType matrixFormat = SparseMatrixType.MATRIX_FORMAT_COO;

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
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getVal()
   */
  @Override
  public float[] getVal() {
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
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#getRowInd()
   */
  @Override
  public int[] getRowInd() {
    return rowInd;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#getColInd()
   */
  @Override
  public int[] getColInd() {
    return colInd;
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
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrix#getRowNumber()
   */
  @Override
  public int getRowNumber() {
    return rowNumber;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrix#getColNumber()
   */
  @Override
  public int getColNumber() {
    return colNumber;
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

  /*
   * (non-Javadoc)
   * 
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString() {
    return "SparseMatrix [rowNumber=" + rowNumber + ", colNumber=" + colNumber + ", nnz=" + nnz
            + ", rowInd=" + Arrays.toString(rowInd) + ", colInd=" + Arrays.toString(colInd)
            + ", val=" + Arrays.toString(val) + ", matrixFormat=" + matrixFormat + "]";
  }

}
