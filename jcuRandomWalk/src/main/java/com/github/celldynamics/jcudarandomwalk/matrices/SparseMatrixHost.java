package com.github.celldynamics.jcudarandomwalk.matrices;

import java.security.InvalidParameterException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import org.apache.commons.lang3.ArrayUtils;
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
public class SparseMatrixHost extends SparseMatrix {

  /**
   * UID
   */
  private static final long serialVersionUID = -2934384684498319094L;
  private int i = 0; // counter

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
    updateDimension();
  }

  /**
   * Add entry (coordinates and value) to store. Matrix is created in CCO format.
   * 
   * <p>This method does not update {@link #getRowNumber()} or {@link #getColNumber()}. The
   * {@link #updateDimension()} must be called explicitelly and the end.
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
    if (matrixFormat != SparseMatrixType.MATRIX_FORMAT_COO) {
      throw new InvalidParameterException(
              "This matrix must be in COO format. Perform explicit conversion.");
    }
    // any 1 at index i in this array stand for index i to remove from Lap
    // merge two arrays in one because they can point the same row/column
    int[] toRem = new int[this.getRowNumber()];
    for (int s = 0; s < rows.length; s++) {
      toRem[rows[s]] = 1;
    }
    // iterate over indices lists and mark those to remove by -1 - ROWS
    int[] rowInd = this.getRowInd();
    int[] colInd = this.getColInd();
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
    double[] newVal = new double[remainingRow];
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
    int[] newRowIndcp = compressIndices(toRem, remainingRow, newRowInd);

    // now compress also rows that can be empty
    List<Integer> ilist = Arrays.asList(ArrayUtils.toObject(newColInd));
    // will contain 1 at index to remove
    int[] toRemCol = new int[newColInd.length];
    // find those rows that are not in new row indexes (indexes must be continous)
    int max = IntStream.of(newColInd).max().getAsInt() + 1;
    int[] colsRemove = IntStream.range(0, max).parallel().filter(x -> !ilist.contains(x)).toArray();
    IntStream.of(colsRemove).forEach(x -> toRemCol[x] = 1); // build toRem
    // compress rows
    int[] newColIndcp = compressIndices(toRemCol, newColInd.length, newColInd);

    ISparseMatrix reducedL;
    reducedL = SparseMatrix.sparseMatrixFactory(this, newRowIndcp, newColIndcp, newVal,
            SparseMatrixType.MATRIX_FORMAT_COO);

    return reducedL;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#removeCols(int[])
   */
  @Override
  public IMatrix removeCols(int[] cols) {
    if (matrixFormat != SparseMatrixType.MATRIX_FORMAT_COO) {
      throw new InvalidParameterException(
              "This matrix must be in COO format. Perform explicit conversion.");
    }
    // any 1 at index i in this array stand for index i to remove from Lap
    // merge two arrays in one because they can point the same row/column
    int[] toRem = new int[this.getColNumber()];
    for (int s = 0; s < cols.length; s++) {
      toRem[cols[s]] = 1;
    }
    // iterate over indices lists and mark those to remove by -1 - ROWS
    int[] rowInd = this.getRowInd();
    int[] colInd = this.getColInd();
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
    double[] newVal = new double[remainingCol];
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
    int[] newColIndcp = compressIndices(toRem, remainingCol, newColInd);

    // now compress also rows that can be empty
    List<Integer> ilist = Arrays.asList(ArrayUtils.toObject(newRowInd));
    // will contain 1 at index to remove
    int[] toRemRow = new int[newRowInd.length];
    // find those rows that are not in new row indexes (indexes must be continous)
    int max = IntStream.of(newRowInd).max().getAsInt() + 1;
    int[] rowsRemove = IntStream.range(0, max).parallel().filter(x -> !ilist.contains(x)).toArray();
    IntStream.of(rowsRemove).forEach(x -> toRemRow[x] = 1); // build toRem
    // compress rows
    int[] newRowIndcp = compressIndices(toRemRow, newRowInd.length, newRowInd);

    ISparseMatrix reducedL;
    reducedL = SparseMatrix.sparseMatrixFactory(this, newRowIndcp, newColIndcp, newVal,
            SparseMatrixType.MATRIX_FORMAT_COO);

    return reducedL;
  }

  /**
   * @param toRem
   * @param remainingRow
   * @param newRowInd
   * @return Array with compressed indices
   */
  private int[] compressIndices(int[] toRem, int remainingRow, int[] newRowInd) {
    // compress
    // after removing indices from newColInd/RowInd it contains only valid nonzero elements (without
    // those from deleted rows and
    // cols) but indexes contain gaps, e.g. if 2nd column was removed newColInd will keep next
    // column after as third whereas it should be moved to left and become the second
    // because we assumed square matrix we will go through toRem array and check which indexes were
    // removed (marked by 1 at index i - removed) and then decrease all indexes larger than those
    // removed in newColInd/newRowInd by one to shift them
    // These arrays need to be copied first otherwise next comparison would be wrong
    int[] newRowIndcp = Arrays.copyOf(newRowInd, newRowInd.length);
    for (int i = 0; i < toRem.length; i++) {
      if (toRem[i] > 0) { // compress all indices larger than i
        for (int k = 0; k < remainingRow; k++) { // go through sparse indexes
          if (newRowInd[k] > i) { // the same for rows
            newRowIndcp[k]--;
          }
        }

      }
    }
    return newRowIndcp;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#toCpu()
   */
  @Override
  public IMatrix toCpu() {
    return this;
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
    double[] v = this.getVal();
    double[] ret = new double[this.getRowNumber()];
    for (int i = 0; i < ri.length; i++) { // along all row indices
      ret[ri[i]] += v[i]; // sum all elements from the same row
    }
    int[] ci_ret = new int[ret.length];
    int[] ri_ret = IntStream.range(0, ret.length).toArray();
    Arrays.fill(ci_ret, 0);

    return SparseMatrix.sparseMatrixFactory(this, ri_ret, ci_ret, ret,
            SparseMatrixType.MATRIX_FORMAT_COO);
  }

}