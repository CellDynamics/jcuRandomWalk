package com.github.celldynamics.jcudarandomwalk.matrices.dense;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.NotImplementedException;
import org.ojalgo.access.ElementView1D;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;
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
   * OjAlg store wrapped by this class.
   */
  public MatrixStore<Double> mat;

  /**
   * Wrap oj object.
   * 
   * @param mat oj matrix
   */
  public DenseVectorOj(MatrixStore<Double> mat) {
    this.mat = mat;
  }

  /**
   * Create oj dense vector.
   * 
   * @param rows
   * @param cols
   * 
   * @param val values.
   */
  public DenseVectorOj(int rows, int cols, float[] val) {
    PrimitiveDenseStore b = PrimitiveDenseStore.FACTORY.makeZero(rows, cols);
    for (int i = 0; i < val.length; i++) {
      b.set(i, val[i]);
    }
    mat = b;
  }

  /**
   * Create oj dense vector.
   * 
   * @param val values.
   */
  public DenseVectorOj(float[] val) {
    PrimitiveDenseStore b = PrimitiveDenseStore.FACTORY.makeZero(val.length, 1);
    for (int i = 0; i < val.length; i++) {
      b.set(i, val[i]);
    }
    mat = b;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getVal()
   */
  @Override
  public float[] getVal() {
    // this returns column ordered nonzero indices. Matrix is ok, but only getVal returns in
    // different order.
    ElementView1D<Double, ?> nz = mat.nonzeros();
    List<Float> ret = new ArrayList<>();
    while (nz.hasNext()) {
      Number e = nz.next().get();
      ret.add(e.floatValue());
    }
    return ArrayUtils.toPrimitive(ret.toArray(new Float[0]));

  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getRowNumber()
   */
  @Override
  public int getRowNumber() {
    return (int) mat.countRows();
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getColNumber()
   */
  @Override
  public int getColNumber() {
    return (int) mat.countColumns();
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getElementNumber()
   */
  @Override
  public int getElementNumber() {
    return (int) mat.count();
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
