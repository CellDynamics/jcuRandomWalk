package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import java.io.Serializable;
import java.security.InvalidParameterException;
import java.util.Arrays;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Hold sparse coordinates.
 * 
 * <p>This is very principal structure used to build more sophisticated objects.
 * 
 * @author baniu
 *
 */
public class SparseCoordinates implements Serializable {

  static final Logger LOGGER = LoggerFactory.getLogger(SparseCoordinates.class.getName());

  /**
   * UID.
   */
  private static final long serialVersionUID = 3342426433330999288L;
  protected SparseMatrixType matrixFormat = SparseMatrixType.MATRIX_FORMAT_COO;
  protected int[] rowInd; // rows
  protected int[] colInd; // cols
  protected float[] val; // value
  protected int rowNumber; // number of rows
  protected int colNumber; // number of cols
  private int counter = 0; // counter

  public SparseCoordinates() {
    this(0);
  }

  /**
   * Create store.
   * 
   * @param size Size of store.
   */
  public SparseCoordinates(int size) {
    rowInd = new int[size];
    colInd = new int[size];
    val = new float[size];
    matrixFormat = SparseMatrixType.MATRIX_FORMAT_COO;
  }

  /**
   * @param matrixFormat
   * @param rowInd
   * @param colInd
   * @param val
   * @param rowNumber
   * @param colNumber
   */
  public SparseCoordinates(int[] rowInd, int[] colInd, float[] val, int rowNumber, int colNumber,
          SparseMatrixType matrixFormat) {
    this.matrixFormat = matrixFormat;
    this.rowInd = rowInd;
    this.colInd = colInd;
    this.val = val;
    this.rowNumber = rowNumber;
    this.colNumber = colNumber;
  }

  /**
   * @param matrixFormat
   * @param rowInd
   * @param colInd
   * @param val
   */
  public SparseCoordinates(int[] rowInd, int[] colInd, float[] val, SparseMatrixType matrixFormat) {
    this.matrixFormat = matrixFormat;
    this.rowInd = rowInd;
    this.colInd = colInd;
    this.val = val;
    updateDimension();
  }

  /**
   * Find maximum row nad column number and set number of rows and columns respectively.
   * 
   * <p>This method can be used for updating number of rows and columns if they changed but its
   * execution is expensive.
   */
  public void updateDimension() {
    colNumber = IntStream.of(getColInd()).parallel().max().getAsInt() + 1; // assuming 0 based
    rowNumber = IntStream.of(getRowInd()).parallel().max().getAsInt() + 1;
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

  /**
   * Get indices of rows.
   * 
   * @return indices of rows
   */
  public int[] getRowInd() {
    return rowInd;
  }

  /**
   * Get indices of columns.
   * 
   * @return indices of columns
   */
  public int[] getColInd() {
    return colInd;
  }

  /**
   * 
   * @param cols
   * @return
   */
  public void removeColsIndices(int[] cols) {
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
    int[] rowIndTmp = Arrays.copyOf(rowInd, rowInd.length);
    int[] colIndTmp = Arrays.copyOf(colInd, colInd.length);
    for (int i = 0; i < rowIndTmp.length; i++) {
      if (toRem[colIndTmp[i]] > 0) {
        colIndTmp[i] = -1; // to remove
      }
    }
    // compute number of nonzero elements that remains
    // ... and find how many rows is >=0 - valid rows
    int remainingCol = 0;
    for (int i = 0; i < colIndTmp.length; i++) {
      if (colIndTmp[i] >= 0) {
        remainingCol++;
      }
    }
    int[] newRowInd = new int[remainingCol];
    int[] newColInd = new int[remainingCol];
    float[] newVal = new float[remainingCol];
    int l = 0;
    for (int i = 0; i < colIndTmp.length; i++) {
      if (colIndTmp[i] < 0) {
        continue;
      }
      newRowInd[l] = rowIndTmp[i];
      newColInd[l] = colIndTmp[i];
      newVal[l] = this.getVal()[i];
      l++;
    }
    compressIndices(toRem, newColInd);

    this.rowInd = newRowInd;
    this.colInd = newColInd;
    this.val = newVal;
    colNumber -= cols.length;
    colNumber = colNumber < 0 ? 0 : colNumber;
    LOGGER.trace("Cols removed");
  }

  /**
   * 
   * @param cols
   * @return
   */
  public void removeRowsIndices(int[] rows) {
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
    int[] rowIndTmp = Arrays.copyOf(rowInd, rowInd.length);
    int[] colIndTmp = Arrays.copyOf(colInd, colInd.length);
    for (int i = 0; i < rowIndTmp.length; i++) {
      if (toRem[rowIndTmp[i]] > 0) {
        rowIndTmp[i] = -1; // to remove
      }
    }
    // compute number of nonzero elements that remains
    // ... and find how many rows is >=0 - valid rows
    int remainingRow = 0;
    for (int i = 0; i < rowIndTmp.length; i++) {
      if (rowIndTmp[i] >= 0) {
        remainingRow++;
      }
    }
    int[] newRowInd = new int[remainingRow];
    int[] newColInd = new int[remainingRow];
    float[] newVal = new float[remainingRow];
    int l = 0;
    for (int i = 0; i < rowIndTmp.length; i++) {
      if (rowIndTmp[i] < 0) {
        continue;
      }
      newRowInd[l] = rowIndTmp[i];
      newColInd[l] = colIndTmp[i];
      newVal[l] = val[i];
      l++;
    }
    compressIndices(toRem, newRowInd);

    this.rowInd = newRowInd;
    this.colInd = newColInd;
    this.val = newVal;
    rowNumber -= rows.length;
    rowNumber = rowNumber < 0 ? 0 : rowNumber;
    LOGGER.trace("Rows removed");
  }

  /**
   * 
   * @return
   */
  public float[] sumAlongRowsIndices() {
    int[] ri = this.getRowInd();
    float[] v = this.getVal();
    float[] ret = new float[this.getRowNumber()];
    // for (int i = 0; i < ri.length; i++) { // along all row indices
    // ret[ri[i]] += v[i]; // sum all elements from the same row
    // }
    IntStream.range(0, ri.length).parallel().forEach(i -> {
      ret[ri[i]] += v[i];
    });

    return ret;
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

  /**
   * Return values.
   * 
   * @return the val
   */
  public float[] getVal() {
    return val;
  }

  // /**
  // * Convert this object to specified Sparse object.
  // *
  // * @param type Sparse matrix type. Either {@link SparseMatrixHost}, {@link SparseMatrixDevice}
  // or
  // * {@link SparseMatrixOj}.
  // * @return {@link ISparseMatrix} object
  // */
  // public ISparseMatrix toSparse(ISparseMatrix type) {
  // return SparseMatrix.sparseMatrixFactory(type, rowInd, colInd, val, rowNumber, colNumber,
  // SparseMatrixType.MATRIX_FORMAT_COO);
  // }

  /**
   * Number of rows.
   * 
   * @return the rowNumber
   */
  public int getRowNumber() {
    return rowNumber;
  }

  /**
   * Number of columns.
   * 
   * @return the colNumber
   */
  public int getColNumber() {
    return colNumber;
  }

  /**
   * @return
   */
  public int getElementNumber() {
    return val.length;
  }

  /**
   * @return
   */
  public SparseMatrixType getSparseMatrixType() {
    return matrixFormat;
  }

  /**
   * @return
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
    return "SparseCoordinates [matrixFormat=" + matrixFormat + ", rowInd=" + Arrays.toString(rowInd)
            + ", colInd=" + Arrays.toString(colInd) + ", val=" + Arrays.toString(val)
            + ", rowNumber=" + rowNumber + ", colNumber=" + colNumber + ", counter=" + counter
            + "]";
  }

}
