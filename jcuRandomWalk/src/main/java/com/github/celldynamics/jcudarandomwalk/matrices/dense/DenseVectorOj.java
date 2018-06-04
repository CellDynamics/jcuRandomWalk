package com.github.celldynamics.jcudarandomwalk.matrices.dense;

import org.apache.commons.lang3.NotImplementedException;
import org.ojalgo.matrix.store.MatrixStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;

/**
 * Dense vector for OjAlg backend.
 * 
 * @author baniu
 *
 */
public class DenseVectorOj implements IDenseVector {

  /**
   * The Constant LOGGER.
   */
  static final Logger LOGGER = LoggerFactory.getLogger(DenseVectorOj.class.getName());

  /**
   * Factory of Oj sparse matrix.
   */
  public static final OjDenseVectorFactory FACTORY = new OjDenseVectorFactory();

  private int nrows;
  private int ncols;

  /**
   * OjAlg store wrapped by this class.
   */
  public MatrixStore<Double> mat;

  DenseVectorOj(MatrixStore<Double> mat) {
    this.mat = mat;
    this.nrows = (int) mat.countRows();
    this.ncols = (int) mat.countColumns();
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getVal()
   */
  @Override
  public float[] getVal() {
    throw new NotImplementedException("not implemented");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getRowNumber()
   */
  @Override
  public int getRowNumber() {
    throw new NotImplementedException("not implemented");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getColNumber()
   */
  @Override
  public int getColNumber() {
    throw new NotImplementedException("not implemented");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getElementNumber()
   */
  @Override
  public int getElementNumber() {
    throw new NotImplementedException("not implemented");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#removeRows(int[])
   */
  @Override
  public IMatrix removeRows(int[] rows) {
    throw new NotImplementedException("not implemented");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#removeCols(int[])
   */
  @Override
  public IMatrix removeCols(int[] cols) {
    throw new NotImplementedException("not implemented");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#multiply(com.github.celldynamics.
   * jcudarandomwalk.matrices.IMatrix)
   */
  @Override
  public IMatrix multiply(IMatrix in) {
    throw new NotImplementedException("not implemented");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#transpose()
   */
  @Override
  public IMatrix transpose() {
    throw new NotImplementedException("not implemented");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#toGpu()
   */
  @Override
  public IMatrix toGpu() {
    return this;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#toCpu()
   */
  @Override
  public IMatrix toCpu() {
    return this;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#free()
   */
  @Override
  public void free() {
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#sumAlongRows()
   */
  @Override
  public IMatrix sumAlongRows() {
    throw new NotImplementedException("not implemented");
  }

}
