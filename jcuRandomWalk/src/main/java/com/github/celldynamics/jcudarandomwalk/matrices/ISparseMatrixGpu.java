/**
 * 
 */
package com.github.celldynamics.jcudarandomwalk.matrices;

/**
 * Operations specific to Gpu.
 * 
 * @author p.baniukiewicz
 *
 */
public interface ISparseMatrixGpu extends ISparseMatrix {

  /**
   * Retrieve from GPU.
   * 
   * @return CPU version.
   */
  public ISparseMatrixCpu toCpu();

  /**
   * Release resources.
   */
  public void free();
}
