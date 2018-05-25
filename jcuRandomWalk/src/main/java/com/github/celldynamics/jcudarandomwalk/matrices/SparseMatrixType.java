package com.github.celldynamics.jcudarandomwalk.matrices;

/**
 * Type of sparse matrix according to https://docs.nvidia.com/cuda/pdf/CUSPARSE_Library.pdf
 * 
 * @author p.baniukiewicz
 *
 */
public enum SparseMatrixType {
  MATRIX_FORMAT_COO, MATRIX_FORMAT_CSR
}
