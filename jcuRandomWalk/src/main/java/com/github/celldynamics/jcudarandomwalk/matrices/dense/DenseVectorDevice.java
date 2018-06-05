package com.github.celldynamics.jcudarandomwalk.matrices.dense;

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import com.github.celldynamics.jcudarandomwalk.matrices.ICudaLibHandles;
import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;

import jcuda.Pointer;
import jcuda.Sizeof;

/**
 * Dense vector on device.
 * 
 * @author baniu
 *
 */
public class DenseVectorDevice extends DenseVector implements ICudaLibHandles {

  /**
   * Default UID.
   */
  private static final long serialVersionUID = 8913630397097404339L;

  private Pointer valPtr = new Pointer();

  /**
   * Default constructor for building dense vector from list of values. Must be implemented in
   * concrete classes.
   * 
   * @param rows number of rows, rows or cols should be 1
   * @param cols number of columns, rows or cols should be 1
   * @param val values
   */
  public DenseVectorDevice(int rows, int cols, float[] val) {
    super(rows, cols);
    this.val = val;
    cudaMalloc(valPtr, getElementNumber() * Sizeof.FLOAT);
    cudaMemcpy(valPtr, Pointer.to(val), getElementNumber() * Sizeof.FLOAT, cudaMemcpyHostToDevice);
  }

  /**
   * Get pointer to data.
   * 
   * @return the valPtr
   */
  public Pointer getValPtr() {
    return valPtr;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVector#getVal()
   */
  @Override
  public float[] getVal() {
    if (val == null) {
      retrieveFromDevice();
    }
    return super.getVal();
  }

  /**
   * Copy indices from device to host.
   */
  public void retrieveFromDevice() {
    val = new float[getElementNumber()];
    cudaMemcpy(Pointer.to(val), getValPtr(), getElementNumber() * Sizeof.FLOAT,
            cudaMemcpyDeviceToHost);
  }

  @Override
  public IMatrix toGpu() {
    return this;
  }

  @Override
  public IMatrix toCpu() {
    if (val == null) {
      retrieveFromDevice();
    }
    return new DenseVectorHost(getRowNumber(), getColNumber(), getVal());
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrixGpu#free()
   */
  @Override
  public void free() {
    try {
      cudaFree(valPtr);
    } catch (Exception e) {
      LOGGER.debug("descr already freed? " + e.getMessage());
    }
  }
}
