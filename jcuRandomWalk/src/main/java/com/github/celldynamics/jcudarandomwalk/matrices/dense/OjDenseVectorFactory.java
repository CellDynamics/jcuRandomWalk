package com.github.celldynamics.jcudarandomwalk.matrices.dense;

import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;

/**
 * Factory of {@link DenseVectorOj}.
 * 
 * @author baniu
 *
 */
public final class OjDenseVectorFactory {
  /**
   * Create column vector.
   * 
   * @param val values
   * @return column vector
   */
  public IDenseVector make(float[] val) {
    PrimitiveDenseStore b = PrimitiveDenseStore.FACTORY.makeZero(val.length, 1);
    for (int i = 0; i < val.length; i++) {
      b.set(i, val[i]);
    }
    return new DenseVectorOj(b);
  }

  public IDenseVector make(MatrixStore<Double> mat) {
    return new DenseVectorOj(mat);
  }
}