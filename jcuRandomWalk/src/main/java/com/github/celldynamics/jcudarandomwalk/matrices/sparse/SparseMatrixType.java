package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

/**
 * Type of sparse matrix according to https://docs.nvidia.com/cuda/pdf/CUSPARSE_Library.pdf
 * 
 * @author p.baniukiewicz
 *
 */
public enum SparseMatrixType {
  /**
   * COO format.
   */
  MATRIX_FORMAT_COO,
  /**
   * CSR format.
   */
  MATRIX_FORMAT_CSR
}
