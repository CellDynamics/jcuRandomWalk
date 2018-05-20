package com.github.celldynamics.jcudarandomwalk.matrices;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrix;
import com.github.celldynamics.jcurandomwalk.ArrayTools;

/**
 * Test class for {@link SparseMatrix}.
 * 
 * @author baniu
 *
 */
public class SparseMatrixTest {

  /**
   * The Constant LOGGER.
   */
  static final Logger LOGGER = LoggerFactory.getLogger(SparseMatrixTest.class.getName());

  /**
   * Test of {@link SparseMatrix#full()}.
   * 
   * @throws Exception on error
   */
  @Test
  public void testFull() throws Exception {
    SparseMatrix test = new SparseMatrix(3);
    /*
     * Create matrix
     * 0 1 0 0 2
     * 0 0 0 0 0
     * 0 3 0 0 0
     */
    test.getCrows()[0] = 0;
    test.getCrows()[1] = 0;
    test.getCrows()[2] = 2;

    test.getCcols()[0] = 1;
    test.getCcols()[1] = 4;
    test.getCcols()[2] = 1;

    test.getCval()[0] = 1;
    test.getCval()[1] = 2;
    test.getCval()[2] = 3;

    double[][] f = test.full();
    LOGGER.debug(ArrayTools.printArray(ArrayTools.array2Object(f)));

    /*
     * 0 2 4
     * 1 3 5
     */
    test = new SparseMatrix(6); // all elements
    test.getCrows()[0] = 0;
    test.getCrows()[1] = 0;
    test.getCrows()[2] = 0;
    test.getCrows()[3] = 1;
    test.getCrows()[4] = 1;
    test.getCrows()[5] = 1;

    test.getCcols()[0] = 0;
    test.getCcols()[1] = 1;
    test.getCcols()[2] = 2;
    test.getCcols()[3] = 0;
    test.getCcols()[4] = 1;
    test.getCcols()[5] = 2;

    test.getCval()[0] = 0;
    test.getCval()[1] = 2;
    test.getCval()[2] = 4;
    test.getCval()[3] = 1;
    test.getCval()[4] = 3;
    test.getCval()[5] = 5;
    f = test.full();
    LOGGER.debug(ArrayTools.printArray(ArrayTools.array2Object(f)));
    double l = 0;
    for (int c = 0; c < 3; c++) {
      for (int r = 0; r < 2; r++) {
        assertThat(f[c][r], closeTo(l++, 1e-5));
      }
    }
  }

}
