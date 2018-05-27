package com.github.celldynamics.jcudarandomwalk.matrices.dense;

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;

import jcuda.Pointer;
import jcuda.Sizeof;

/**
 * Dense vector on device.
 * 
 * @author baniu
 *
 */
public class DenseVectorDevice extends DenseVector {

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
  public DenseVectorDevice(int rows, int cols, double[] val) {
    super(rows, cols);
    cudaMalloc(valPtr, getElementNumber() * Sizeof.DOUBLE);
    cudaMemcpy(valPtr, Pointer.to(val), getElementNumber() * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
  }

  /**
   * Get pointer to data.
   * 
   * @return the valPtr
   */
  public Pointer getValPtr() {
    return valPtr;
  }

  /**
   * Copy indices from device to host.
   */
  public void retrieveFromDevice() {
    val = new double[getElementNumber()];
    cudaMemcpy(Pointer.to(val), getValPtr(), getElementNumber() * Sizeof.DOUBLE,
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
    // TODO protect against freeing already fried
    cudaFree(valPtr);
  }
}
