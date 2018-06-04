package com.github.celldynamics.jcudarandomwalk.matrices.dense;

import org.ojalgo.matrix.store.PrimitiveDenseStore;

public final class OjDenseVectorFactory {
  public IDenseVector make(float[] val) {
    PrimitiveDenseStore b = PrimitiveDenseStore.FACTORY.makeZero(val.length, 1);
    for (int i = 0; i < val.length; i++) {
      b.set(i, val[i]);
    }
    return new DenseVectorOj(b);
  }
}