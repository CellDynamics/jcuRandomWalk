package com.github.celldynamics.jcudarandomwalk.matrices;

/**
 * Base Interface representing general matrix.
 * 
 * @author p.baniukiewicz
 *
 */
public interface IMatrix {

  /**
   * Get values from matrix.
   * 
   * @return Row ordered elements of matrix
   */
  public float[] getVal();

  /**
   * Get number of rows. Note that this operation can be time consuming for sparse matrices.
   * 
   * @return number of rows
   */
  public int getRowNumber();

  /**
   * Get number of columns. Note that this operation can be time consuming for sparse matrices.
   * 
   * @return number of columns
   */
  public int getColNumber();

  /**
   * Get number of elements in matrix. For Sparse it is number of non-zero elements.
   * 
   * @return number of elements. Size of vector returned by {@link #getVal()}
   */
  public int getElementNumber();

  /**
   * Remove rows from this matrix.
   * 
   * @param rows list of indices, 0-based, can not contain duplicates
   * @return Matrix without rows.
   */
  public IMatrix removeRows(int[] rows);

  /**
   * Remove cols from this matrix.
   * 
   * @param cols list of indices, 0-based, can not contain duplicates
   * @return Matrix without rows.
   */
  public IMatrix removeCols(int[] cols);

  /**
   * Matrix multiplication.
   * 
   * @param in right argument
   * @return this*in
   */
  public IMatrix multiply(IMatrix in);

  /**
   * Transpose sparse matrix.
   * 
   * @return transposed matrix in CSR format
   */
  public IMatrix transpose();

  // /**
  // * Send to GPU.
  // *
  // * <p>Class can throw java.lang.UnsupportedOperationException if that conversion is not
  // possible.
  // *
  // * @return GPU version.
  // */
  // public IMatrix toGpu();
  //
  // /**
  // * Retrieve from GPU. Return this if matrix is already on cpu.
  // *
  // * @return CPU version.
  // */
  // public IMatrix toCpu();

  /**
   * Release resources (if needed).
   */
  public void free();

  /**
   * Compute sum along rows. For Vectors it will be sum of all elements.
   * 
   * @return Vector of size Rx1
   */
  public IMatrix sumAlongRows();

}
