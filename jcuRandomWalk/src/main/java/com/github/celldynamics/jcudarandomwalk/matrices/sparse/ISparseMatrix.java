package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.dense.IDenseVector;

/**
 * General interface representing sparse matrix.
 * 
 * @author p.baniukiewicz
 *
 */
public interface ISparseMatrix extends IMatrix {

  /**
   * Get i-coordinates vector. Rows, always on Host.
   * 
   * @return the vi
   * @see SparseMatrixDevice#retrieveFromDevice()
   */
  public int[] getRowInd();

  /**
   * Get j-coordinates vector. Columns, always on Host.
   * 
   * @return the vj
   * @see SparseMatrixDevice#retrieveFromDevice()
   */
  public int[] getColInd();

  /**
   * Get type of sparse matrix.
   * 
   * @return format
   * @see SparseMatrixType
   */
  public SparseMatrixType getSparseMatrixType();

  /**
   * Convert from COO to CSR.
   * 
   * @return converted CSR
   */
  public ISparseMatrix convert2csr();

  /**
   * Convert from CSR to COO.
   * 
   * @return converted COO
   */
  public ISparseMatrix convert2coo();

  /**
   * Convert sparse coordinates to full matrix.
   * 
   * <p>Only small matrixes. Column ordered. [col][row] like x,y
   * 
   * @return full 2D matrix [cols][rows]
   */
  public double[][] full();

  public float[] luSolve(IDenseVector b_gpuPtr, boolean iLuBiCGStabSolve);
}
