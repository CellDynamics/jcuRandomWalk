package com.github.celldynamics.jcudarandomwalk.matrices;

/**
 * Represent matrix stored on CPU. This interface should be implemented by all classes representing
 * matrices except CUDA based.
 * 
 * @author baniu
 *
 */
public interface IStoredOnCpu {

  /**
   * Send to GPU.
   * 
   * <p>Class can throw java.lang.UnsupportedOperationException if that conversion is not possible.
   * 
   * @return GPU version.
   */
  public ISparseMatrix toGpu();
}
