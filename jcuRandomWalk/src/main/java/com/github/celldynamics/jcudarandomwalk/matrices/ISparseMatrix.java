package com.github.celldynamics.jcudarandomwalk.matrices;

/**
 * Type of sparse matrix according to https://docs.nvidia.com/cuda/pdf/CUSPARSE_Library.pdf
 * 
 * @author p.baniukiewicz
 *
 */
enum SparseMatrixType {
  MATRIX_FORMAT_COO, MATRIX_FORMAT_CSR
}

/**
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
   * Matrix multiplication.
   * 
   * @param in right argument
   * @return this*in
   */
  public ISparseMatrix multiply(ISparseMatrix in);

}
