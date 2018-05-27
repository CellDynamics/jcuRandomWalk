package com.github.celldynamics.jcudarandomwalk.matrices;

import jcuda.jcusparse.cusparseHandle;

/**
 * Store GPU handles to cuda libraries.
 * 
 * @author baniu
 *
 */
public interface ICudaLibHandles {

  /**
   * Handle to cusparse driver.
   * 
   * <p>It must be created before use: <tt>JCusparse.cusparseCreate(SparseMatrixDevice.handle);</tt>
   * and then destroyed: <tt>JCusparse.cusparseDestroy(SparseMatrixDevice.handle);</tt>
   */
  public static final cusparseHandle handle = new cusparseHandle();
}
