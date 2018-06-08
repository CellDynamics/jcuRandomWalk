package com.github.celldynamics.jcudarandomwalk.matrices.dense;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.ojalgo.access.ElementView1D;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Dense vector for OjAlg backend.
 * 
 * @author baniu
 *
 */
public class DenseVectorOj {

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

  public int getRowNumber() {
    return (int) mat.countRows();
  }

  public int getColNumber() {
    return (int) mat.countColumns();
  }

  public int getElementNumber() {
    return (int) mat.count();
  }

}
