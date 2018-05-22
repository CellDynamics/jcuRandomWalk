package com.github.celldynamics.jcudarandomwalk.matrices;

/**
 * @author p.baniukiewicz
 *
 */
public interface IMatrix {

  /**
   * Get values from matrix.
   * 
   * @return Row ordered elements of matrix
   */
  public double[] getVal();

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

}
