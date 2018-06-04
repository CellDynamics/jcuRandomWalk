package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import static com.github.baniuk.ImageJTestSuite.matchers.arrays.ArrayMatchers.arrayCloseTo;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.is;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.apache.commons.lang3.NotImplementedException;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVectorOj;
import com.github.celldynamics.jcudarandomwalk.matrices.dense.IDenseVector;
import com.github.celldynamics.jcurandomwalk.TestDataGenerators;

/**
 * The Class ISparseMatrixOjTest.
 *
 * @author p.baniukiewicz
 */
public class SparseMatrixOjTest {

  static final Logger LOGGER = LoggerFactory.getLogger(SparseMatrixOjTest.class.getName());

  /**
   * Sets the up.
   *
   * @throws Exception the exception
   */
  @Before
  public void setUp() throws Exception {
  }

  /**
   * Tear down.
   *
   * @throws Exception the exception
   */
  @After
  public void tearDown() throws Exception {
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#getVal()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testGetVal() throws Exception {
    int[] ri = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] ci = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    float[] v = new float[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };
    // note that val returns in column order
    float[] exp = new float[] { 10, 131, 101, 11, 12, 13, 14, 102, 15 };

    ISparseMatrix mat = SparseMatrixOj.FACTORY.make(ri, ci, v);
    float[] retval = mat.getVal();
    assertThat(Arrays.asList(retval), contains(exp));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#getRowNumber()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testGetRowNumber() throws Exception {
    int[] ri = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] ci = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    float[] v = new float[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };

    ISparseMatrix mat = SparseMatrixOj.FACTORY.make(ri, ci, v);
    int ret = mat.getRowNumber();
    assertThat(ret, is(6));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#getRowNumber()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testGetRowNumber_1() throws Exception {
    int[] ri = new int[] { 0, 0, 0, 2, 3, 4, 4, 5 };
    int[] ci = new int[] { 0, 1, 5, 2, 3, 0, 4, 5 };
    float[] v = new float[] { 10, 101, 102, 12, 13, 131, 14, 15 };

    ISparseMatrix mat = SparseMatrixOj.FACTORY.make(ri, ci, v);
    int ret = mat.getRowNumber();
    assertThat(ret, is(6));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#getColNumber()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testGetColNumber() throws Exception {
    int[] ri = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] ci = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    float[] v = new float[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };

    ISparseMatrix mat = SparseMatrixOj.FACTORY.make(ri, ci, v);
    int ret = mat.getColNumber();
    assertThat(ret, is(6));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#getColNumber()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testGetColNumber_1() throws Exception {
    int[] ri = new int[] { 0, 0, 0, 1, 2, 3, 3, 4 };
    int[] ci = new int[] { 0, 1, 4, 1, 2, 0, 3, 4 };
    float[] v = new float[] { 10, 101, 102, 11, 13, 131, 14, 15 };

    ISparseMatrix mat = SparseMatrixOj.FACTORY.make(ri, ci, v);
    int ret = mat.getColNumber();
    assertThat(ret, is(5));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#getElementNumber()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testGetElementNumber() throws Exception {
    int[] ri = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] ci = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    float[] v = new float[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };

    ISparseMatrix mat = SparseMatrixOj.FACTORY.make(ri, ci, v);
    int el = mat.getElementNumber();
    assertThat(el, is(v.length));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#removeRows(int[])}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testRemoveRows() throws Exception {
    throw new RuntimeException("not yet implemented");
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#removeCols(int[])}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testRemoveCols() throws Exception {
    throw new RuntimeException("not yet implemented");
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#multiply(com.github.celldynamics.jcudarandomwalk.matrices.IMatrix)}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testMultiply() throws Exception {
    TestDataGenerators td = new TestDataGenerators();
    ISparseMatrix mat = SparseMatrixOj.FACTORY.make(td.rowInd, td.colInd, td.valRowOrder);
    ISparseMatrix b = SparseMatrixOj.FACTORY.make(td.rowInd1, td.colInd1, td.val1RowOrder);
    IMatrix ret = mat.multiply(b);
    float[] vals = ret.getVal();
    assertThat(Arrays.asList(vals), contains(new float[] { 17.0f, 8.0f, 5.0f, 8.0f, 13.0f, 27.0f,
        5.0f, 138.0f, 48.0f, 27.0f, 48.0f, 117.0f }));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#transpose()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testTranspose() throws Exception {
    TestDataGenerators td = new TestDataGenerators();
    ISparseMatrix mat = SparseMatrixOj.FACTORY.make(td.rowInd, td.colInd, td.valRowOrder);
    LOGGER.trace(((SparseMatrixOj) mat).mat.toString());
    IMatrix matt = mat.transpose();
    LOGGER.trace(((SparseMatrixOj) matt).mat.toString());
    float[] vals = matt.getVal();
    assertThat(Arrays.asList(vals), contains(td.val1ColOrder));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#toGpu()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testToGpu() throws Exception {
    ISparseMatrix r =
            SparseMatrixOj.FACTORY.make(new int[] { 1 }, new int[] { 1 }, new float[] { 1 });
    assertThat(r, is(r));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#toCpu()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testToCpu() throws Exception {
    ISparseMatrix r =
            SparseMatrixOj.FACTORY.make(new int[] { 1 }, new int[] { 1 }, new float[] { 1 });
    assertThat(r, is(r));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#free()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testFree() throws Exception {
    ISparseMatrix r =
            SparseMatrixOj.FACTORY.make(new int[] { 1 }, new int[] { 1 }, new float[] { 1 });
    r.free();
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#sumAlongRows()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testSumAlongRows() throws Exception {
    throw new RuntimeException("not yet implemented");
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#getRowInd()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testGetRowInd() throws Exception {
    int[] ri = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    // expected columns but taken for non-zero elements counted aloong rows
    int[] rexp = new int[] { 0, 4, 0, 1, 2, 3, 4, 0, 5 };
    int[] ci = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    float[] v = new float[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };

    ISparseMatrix mat = SparseMatrixOj.FACTORY.make(ri, ci, v);
    int[] ret = mat.getRowInd();
    assertThat(Arrays.asList(ret), contains(rexp));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#getColInd()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testGetColInd() throws Exception {
    int[] ri = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] ci = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    // expected columns but taken for non-zero elements counted aloong rows
    int[] cexp = new int[] { 0, 00, 1, 1, 2, 3, 4, 5, 5 };
    float[] v = new float[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };

    ISparseMatrix mat = SparseMatrixOj.FACTORY.make(ri, ci, v);
    int[] ret = mat.getColInd();
    assertThat(Arrays.asList(ret), contains(cexp));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#getSparseMatrixType()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testGetSparseMatrixType() throws Exception {
    ISparseMatrix r =
            SparseMatrixOj.FACTORY.make(new int[] { 1 }, new int[] { 1 }, new float[] { 1 });
    assertThat(r.getSparseMatrixType(), is(SparseMatrixType.MATRIX_FORMAT_COO));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#convert2csr()}.
   *
   * @throws Exception the exception
   */
  @Test(expected = NotImplementedException.class)
  public void testConvert2csr() throws Exception {
    SparseMatrixOj.FACTORY.make(new int[] { 1 }, new int[] { 1 }, new float[] { 1 }).convert2csr();
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#convert2coo()}.
   *
   * @throws Exception the exception
   */
  @Test(expected = NotImplementedException.class)
  public void testConvert2coo() throws Exception {
    SparseMatrixOj.FACTORY.make(new int[] { 1 }, new int[] { 1 }, new float[] { 1 }).convert2coo();
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#full()}.
   *
   * @throws Exception the exception
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixHostTest#testFull()
   */
  @Test
  public void testFull() throws Exception {
    ISparseMatrix test = SparseMatrixOj.FACTORY.make(new int[] { 0, 0, 2 }, new int[] { 1, 4, 1 },
            new float[] { 1, 2, 3 });
    LOGGER.trace("test: " + test.toString());
    double[][] f = test.full();
    assertThat(f[0][1], is(1.0));
    assertThat(f[0][4], is(2.0));
    assertThat(f[2][1], is(3.0));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#luSolve(com.github.celldynamics.jcudarandomwalk.matrices.dense.IDenseVector, boolean, int, float)}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testLuSolve() throws Exception {
    int[] rows =
            new int[] { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4 };
    int[] cols =
            new int[] { 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4 };
    float[] vals = new float[] { 0.9f, 0.4f, 0.1f, 0.9f, 0.1f, 0.1f, 0.9f, 0.9f, 0.6f, 0.2f, 0.4f,
        0.2f, 0.6f, 0.4f, 0.1f, 0.3f, 0.3f, 0.5f, 0.5f, 0.2f, 0.8f, 0.1f, 0.1f, 0.4f, 0.2f };
    float[] bval = new float[] { 6.1f, 8f, 4.7f, 5.4f, 3.9f };
    ISparseMatrix a = SparseMatrixOj.FACTORY.make(rows, cols, vals);
    IDenseVector b = DenseVectorOj.FACTORY.make(bval);

    float[] ret = a.luSolve(b, true, 0, 0);

    Double[] retd =
            IntStream.range(0, ret.length).mapToDouble(i -> ret[i]).boxed().toArray(Double[]::new);

    assertThat(retd, arrayCloseTo(new double[] { 1, 2, 3, 4, 5 }, 1e-5));
  }

}
