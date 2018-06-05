package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import java.io.Serializable;
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
  protected int[] rowInd; // rows
  protected int[] colInd; // cols
  protected float[] val; // value
  protected int rowNumber; // number of rows
  protected int colNumber; // number of cols
  private int counter = 0; // counter

  /**
   * Create store.
   * 
   * @param size Size of store.
   */
  public SparseCoordinates(int size) {
    rowInd = new int[size];
    colInd = new int[size];
    val = new float[size];
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
   * Return values.
   * 
   * @return the val
   */
  public float[] getVal() {
    return val;
  }

  /**
   * Convert this object to specified Sparse object.
   * 
   * @param type Sparse matrix type. Either {@link SparseMatrixHost}, {@link SparseMatrixDevice} or
   *        {@link SparseMatrixOj}.
   * @return {@link ISparseMatrix} object
   */
  public ISparseMatrix toSparse(ISparseMatrix type) {
    return SparseMatrix.sparseMatrixFactory(type, rowInd, colInd, val, rowNumber, colNumber,
            SparseMatrixType.MATRIX_FORMAT_COO);
  }

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

}
