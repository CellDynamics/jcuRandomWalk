package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.contains;

import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;
import com.github.celldynamics.jcurandomwalk.ArrayTools;

/**
 * Test class for {@link SparseMatrixHost}.
 * 
 * @author baniu
 *
 */
public class SparseMatrixHostTest {

  /**
   * The Constant LOGGER.
   */
  static final Logger LOGGER = LoggerFactory.getLogger(SparseMatrixHostTest.class.getName());

  /**
   * Test of CCO format. Follow https://docs.nvidia.com/cuda/pdf/CUSPARSE_Library.pdf
   * 
   * @throws Exception on error
   */
  @Test
  public void testCusparse() throws Exception {
    SparseMatrixHost test = new SparseMatrixHost(9);
    test.add(0, 0, 1);
    test.add(0, 1, 4);
    test.add(1, 1, 2);
    test.add(1, 2, 3);
    test.add(2, 0, 5);
    test.add(2, 3, 7);
    test.add(2, 4, 8);
    test.add(3, 2, 9);
    test.add(3, 4, 6);
    double[][] f = test.full(); // f is addressed [col][row] like x,y
    LOGGER.debug(ArrayTools.printArray(ArrayTools.array2Object(f)));
    assertThat(f[0][0], is(1.0));
    assertThat(f[1][1], is(2.0));
    assertThat(f[2][3], is(9.0));
    assertThat(f.length, is(5));
    assertThat(f[0].length, is(4));
    assertThat(test.getColInd().length, is(9));
    assertThat(Arrays.asList(ArrayUtils.toObject(test.getVal())),
            contains(1.0, 4.0, 2.0, 3.0, 5.0, 7.0, 8.0, 9.0, 6.0)); // order from page 16

  }

  /**
   * Test of {@link SparseMatrixHost#full()}.
   * 
   * @throws Exception on error
   */
  @Test
  public void testFull() throws Exception {
    SparseMatrixHost test = new SparseMatrixHost(3);
    /*
     * Create matrix
     * 0 1 0 0 2
     * 0 0 0 0 0
     * 0 3 0 0 0
     */
    test.add(0, 1, 1);
    test.add(0, 4, 2);
    test.add(2, 1, 3);

    double[][] f = test.full();
    LOGGER.debug(ArrayTools.printArray(ArrayTools.array2Object(f)));
    LOGGER.debug(Arrays.toString(test.getVal()));
    LOGGER.debug(Arrays.toString(test.getRowInd()));
    LOGGER.debug(Arrays.toString(test.getColInd()));

    /*
     * 0 2 4
     * 1 3 5
     */
    test = new SparseMatrixHost(6); // all elements
    test.getRowInd()[0] = 0;
    test.getRowInd()[1] = 0;
    test.getRowInd()[2] = 0;
    test.getRowInd()[3] = 1;
    test.getRowInd()[4] = 1;
    test.getRowInd()[5] = 1;

    test.getColInd()[0] = 0;
    test.getColInd()[1] = 1;
    test.getColInd()[2] = 2;
    test.getColInd()[3] = 0;
    test.getColInd()[4] = 1;
    test.getColInd()[5] = 2;

    test.getVal()[0] = 0;
    test.getVal()[1] = 2;
    test.getVal()[2] = 4;
    test.getVal()[3] = 1;
    test.getVal()[4] = 3;
    test.getVal()[5] = 5;
    f = test.full();
    LOGGER.debug(ArrayTools.printArray(ArrayTools.array2Object(f)));
    double l = 0;
    for (int c = 0; c < 3; c++) {
      for (int r = 0; r < 2; r++) {
        assertThat(f[c][r], closeTo(l++, 1e-5));
      }
    }
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixHost#toGpu()}.
   * 
   * @throws Exception
   */
  @Test
  public void testToSparseMatrixDevice() throws Exception {
    int[] rowInd = new int[] { 0, 0, 1, 1, 2, 2, 2, 3, 3 };
    int[] colInd = new int[] { 0, 1, 1, 2, 0, 3, 4, 2, 4 };
    double[] val = new double[] { 1, 4, 2, 3, 5, 7, 8, 9, 6 };
    SparseMatrixHost test =
            new SparseMatrixHost(rowInd, colInd, val, SparseMatrixType.MATRIX_FORMAT_COO);
    SparseMatrixDevice spd = (SparseMatrixDevice) test.toGpu();
    assertThat(Arrays.asList(spd.getVal()), contains(val));
    assertThat(Arrays.asList(spd.getColInd()), contains(colInd));
    assertThat(Arrays.asList(spd.getRowInd()), contains(rowInd));
    assertThat(spd.getElementNumber(), is(9));
    spd.free();
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixHost#removeRows(int[])}.
   * 
   * @throws Exception
   */
  @Test
  public void testRemoveRows() throws Exception {
    // Laplacian is square, assume diagonal only
    int[] rI = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] cI = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    double[] v = new double[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };
    ISparseMatrix testL = new SparseMatrixHost(rI, cI, v, SparseMatrixType.MATRIX_FORMAT_COO);
    LOGGER.debug("Laplacean" + testL.toString());

    // remove row/co 1,2,3
    int[] toRem = new int[] { 1, 3, 4 };

    IMatrix ret = testL.removeRows(toRem);
    LOGGER.debug("Reduced" + ret.toString());
    assertThat(ret.getColNumber(), is(4));
    assertThat(ret.getRowNumber(), is(3));
    assertThat(ret.getElementNumber(), is(5));
    assertThat(Arrays.asList(ret.getVal()),
            contains(new double[] { 10.0, 101.0, 102.0, 12.0, 15.0 }));
    LOGGER.trace(
            "Red" + ArrayTools.printArray(ArrayTools.array2Object(((ISparseMatrix) ret).full())));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixHost#removeRows(int[])}.
   * 
   * @throws Exception
   */
  @Test
  public void testRemoveRows_1() throws Exception {
    // Laplacian is square, assume diagonal only
    int[] rI = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] cI = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    double[] v = new double[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };
    ISparseMatrix testL = new SparseMatrixHost(rI, cI, v, SparseMatrixType.MATRIX_FORMAT_COO);
    LOGGER.debug("Laplacean" + testL.toString());

    // remove row/co 1,2,3
    int[] toRem = new int[] { 2 };

    IMatrix ret = testL.removeRows(toRem);
    LOGGER.debug("Reduced" + ret.toString());
    assertThat(ret.getColNumber(), is(5));
    assertThat(ret.getRowNumber(), is(5));
    assertThat(ret.getElementNumber(), is(8));
    assertThat(Arrays.asList(ret.getVal()),
            contains(new double[] { 10.0, 101.0, 102.0, 11.0, 13.0, 131.0, 14.0, 15.0 }));
    LOGGER.trace(
            "Red" + ArrayTools.printArray(ArrayTools.array2Object(((ISparseMatrix) ret).full())));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixHost#removeCols(int[])}.
   * 
   * @throws Exception
   */
  @Test
  public void testRemoveCols() throws Exception {
    // Laplacian is square, assume diagonal only
    int[] rI = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] cI = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    double[] v = new double[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };
    ISparseMatrix testL = new SparseMatrixHost(rI, cI, v, SparseMatrixType.MATRIX_FORMAT_COO);
    LOGGER.debug("Laplacean" + testL.toString());

    // remove row/co 1,2,3
    int[] toRem = new int[] { 1, 3, 4 };

    IMatrix ret = testL.removeCols(toRem);
    LOGGER.debug("Reduced" + ret.toString());
    assertThat(ret.getColNumber(), is(3));
    assertThat(ret.getRowNumber(), is(4));
    assertThat(ret.getElementNumber(), is(5));
    assertThat(Arrays.asList(ret.getVal()),
            contains(new double[] { 10.0, 102.0, 12.0, 131.0, 15.0 }));
    LOGGER.trace(
            "Red" + ArrayTools.printArray(ArrayTools.array2Object(((ISparseMatrix) ret).full())));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixHost#removeCols(int[])}.
   * 
   * @throws Exception
   */
  @Test
  public void testRemoveCols_1() throws Exception {
    // Laplacian is square, assume diagonal only
    int[] rI = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] cI = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    double[] v = new double[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };
    ISparseMatrix testL = new SparseMatrixHost(rI, cI, v, SparseMatrixType.MATRIX_FORMAT_COO);
    LOGGER.debug("Laplacean" + testL.toString());

    // remove row/co 1,2,3
    int[] toRem = new int[] { 2 };

    IMatrix ret = testL.removeCols(toRem);
    LOGGER.debug("Reduced" + ret.toString());
    assertThat(ret.getColNumber(), is(5));
    assertThat(ret.getRowNumber(), is(5));
    assertThat(ret.getElementNumber(), is(8));
    assertThat(Arrays.asList(ret.getVal()),
            contains(new double[] { 10.0, 101.0, 102.0, 11.0, 13.0, 131.0, 14.0, 15.0 }));
    LOGGER.trace(
            "Red" + ArrayTools.printArray(ArrayTools.array2Object(((ISparseMatrix) ret).full())));
  }

  /**
   * @throws Exception
   */
  @Test
  public void testSumAlongRows() throws Exception {
    int[] rI = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] cI = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    double[] v = new double[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };
    ISparseMatrix testL = new SparseMatrixHost(rI, cI, v, SparseMatrixType.MATRIX_FORMAT_COO);

    IMatrix ret = testL.sumAlongRows();
    assertThat(Arrays.asList(ret.getVal()),
            contains(new double[] { 213.0, 11.0, 12.0, 13.0, 145.0, 15.0 }));
    LOGGER.trace(
            "Sum" + ArrayTools.printArray(ArrayTools.array2Object(((ISparseMatrix) ret).full())));
  }

}