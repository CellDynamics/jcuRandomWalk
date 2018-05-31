package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import org.junit.Test;
import org.ojalgo.matrix.decomposition.LU;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;
import org.ojalgo.matrix.store.SparseStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Simple tests.
 * 
 * @author baniu
 *
 */
public class OjAlgoTest {
  static final Logger LOGGER = LoggerFactory.getLogger(OjAlgoTest.class.getName());

  @Test
  public void testLuSolve() throws Exception {
    int[] rows =
            new int[] { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4 };
    int[] cols =
            new int[] { 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4 };
    double[] vals = new double[] { 0.9, 0.4, 0.1, 0.9, 0.1, 0.1, 0.9, 0.9, 0.6, 0.2, 0.4, 0.2, 0.6,
        0.4, 0.1, 0.3, 0.3, 0.5, 0.5, 0.2, 0.8, 0.1, 0.1, 0.4, 0.2 };
    double[] bval = new double[] { 6.1, 8, 4.7, 5.4, 3.9 };

    SparseStore<Double> mtrxA = SparseStore.PRIMITIVE.make(5, 5);
    for (int i = 0; i < rows.length; i++) {
      mtrxA.set(rows[i], cols[i], vals[i]);
    }

    PrimitiveDenseStore b = PrimitiveDenseStore.FACTORY.makeZero(5, 1);
    for (int i = 0; i < 5; i++) {
      b.set(i, bval[i]);
    }

    // JacobiSolver tmpJacobi = new JacobiSolver();
    // tmpJacobi.configurator().iterations(50).accuracy(NumberContext.getMath(6));
    // Optional<MatrixStore<Double>> ret = tmpJacobi.solve(mtrxA, b);
    final LU<Double> tmpA = LU.PRIMITIVE.make();
    tmpA.decompose(mtrxA);
    MatrixStore<Double> ret = tmpA.solve(mtrxA, b);
    LOGGER.debug("" + mtrxA.toString());
    LOGGER.debug("" + b.toString());
    LOGGER.debug("ret" + ret.toString());
  }
}
