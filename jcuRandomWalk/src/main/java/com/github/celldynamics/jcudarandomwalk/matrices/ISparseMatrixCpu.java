package com.github.celldynamics.jcudarandomwalk.matrices;

/**
 * This interface should be implemented in all classes that are not in GPU.
 * 
 * @author p.baniukiewicz
 *
 */
public interface ISparseMatrixCpu extends ISparseMatrix {

  /**
   * Send to GPU.
   * 
   * <p>Class can throw java.lang.UnsupportedOperationException if that conversion is not possible.
   * 
   * @return GPU version.
   */
  public ISparseMatrixGpu toGpu();
}
