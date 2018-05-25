package com.github.celldynamics.jcudarandomwalk.matrices;

/**
 * Represent matrix stored on GPU.
 * 
 * @author baniu
 *
 */
public interface IStoredOnGpu {

  /**
   * Retrieve from GPU.
   * 
   * TODO add parameter and use SparseMatrix.sparseMatrixFactory() to generate in required format
   * 
   * @return CPU version.
   */
  public ISparseMatrix toCpu();

  /**
   * Release resources.
   */
  public void free();
}
