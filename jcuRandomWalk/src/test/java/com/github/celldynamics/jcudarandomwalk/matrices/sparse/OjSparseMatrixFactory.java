package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import java.util.stream.IntStream;

import org.ojalgo.matrix.store.SparseStore;

/**
 * @author p.baniukiewicz
 *
 */
public final class OjSparseMatrixFactory {

  public ISparseMatrix make(int[] rowInd, int[] colInd, float[] val) {
    if ((rowInd.length != colInd.length) || (rowInd.length != val.length)) {
      throw new IllegalArgumentException("Input arrays should have the same length in COO format");
    }
    int colNumber = IntStream.of(colInd).parallel().max().getAsInt() + 1; // assuming 0 based
    int rowNumber = IntStream.of(rowInd).parallel().max().getAsInt() + 1;
    SparseStore<Double> mtrxA = SparseStore.PRIMITIVE.make(rowNumber, colNumber);
    for (int i = 0; i < rowInd.length; i++) {
      mtrxA.set(rowInd[i], colInd[i], val[i]);
    }

    return new SparseMatrixOj(mtrxA);
  }

}
