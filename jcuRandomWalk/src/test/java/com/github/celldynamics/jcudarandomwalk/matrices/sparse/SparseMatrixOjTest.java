package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.is;
import static org.junit.Assert.fail;

import java.util.Arrays;

import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;
import com.github.celldynamics.jcurandomwalk.TestDataGenerators;

// TODO: Auto-generated Javadoc
/**
 * The Class ISparseMatrixOjTest.
 *
 * @author p.baniukiewicz
 */
public class SparseMatrixOjTest {

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
    int[] ci = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] ri = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    float[] v = new float[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };

    ISparseMatrix mat = SparseMatrixOj.FACTORY.make(ri, ci, v);
    float[] retval = mat.getVal();
    assertThat(Arrays.asList(retval), contains(v));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#getRowNumber()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testGetRowNumber() throws Exception {
    int[] ci = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] ri = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
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
    int[] ci = new int[] { 0, 0, 0, 2, 3, 4, 4, 5 };
    int[] ri = new int[] { 0, 1, 5, 2, 3, 0, 4, 5 };
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
    int[] ci = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] ri = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
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
    int[] ci = new int[] { 0, 0, 0, 1, 2, 3, 3, 4 };
    int[] ri = new int[] { 0, 1, 4, 1, 2, 0, 3, 4 };
    float[] v = new float[] { 10, 101, 102, 11, 13, 131, 14, 15 };

    ISparseMatrix mat = SparseMatrixOj.FACTORY.make(ri, ci, v);
    int ret = mat.getColNumber();
    assertThat(ret, is(5));
  }

  // TODO add nonsquare matrices
  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#getElementNumber()}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testGetElementNumber() throws Exception {
    int[] ci = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] ri = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
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
  @Ignore
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
  @Ignore
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
    ISparseMatrix mat = SparseMatrixOj.FACTORY.make(td.rowInd, td.colInd, td.val);
    ISparseMatrix b = SparseMatrixOj.FACTORY.make(td.rowInd1, td.colInd1, td.val1);
    IMatrix ret = mat.multiply(b);
    float[] vals = ret.getVal();
    assertThat(Arrays.asList(vals), contains(new float[] { 17.0f, 8.0f, 5.0f, 8.0f, 13.0f, 27.0f,
        5.0f, 138.0f, 48.0f, 27.0f, 48.0f, 117.0f }));
    fail(); // TODO finish
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#transpose()}.
   *
   * @throws Exception the exception
   */
  @Test
  @Ignore
  public void testTranspose() throws Exception {
    throw new RuntimeException("not yet implemented");
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#toGpu()}.
   *
   * @throws Exception the exception
   */
  @Test
  @Ignore
  public void testToGpu() throws Exception {
    throw new RuntimeException("not yet implemented");
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#toCpu()}.
   *
   * @throws Exception the exception
   */
  @Test
  @Ignore
  public void testToCpu() throws Exception {
    throw new RuntimeException("not yet implemented");
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#free()}.
   *
   * @throws Exception the exception
   */
  @Test
  @Ignore
  public void testFree() throws Exception {
    throw new RuntimeException("not yet implemented");
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#sumAlongRows()}.
   *
   * @throws Exception the exception
   */
  @Test
  @Ignore
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
  @Ignore
  public void testGetRowInd() throws Exception {
    throw new RuntimeException("not yet implemented");
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#getColInd()}.
   *
   * @throws Exception the exception
   */
  @Test
  @Ignore
  public void testGetColInd() throws Exception {
    throw new RuntimeException("not yet implemented");
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#getSparseMatrixType()}.
   *
   * @throws Exception the exception
   */
  @Test
  @Ignore
  public void testGetSparseMatrixType() throws Exception {
    throw new RuntimeException("not yet implemented");
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#convert2csr()}.
   *
   * @throws Exception the exception
   */
  @Test
  @Ignore
  public void testConvert2csr() throws Exception {
    throw new RuntimeException("not yet implemented");
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#convert2coo()}.
   *
   * @throws Exception the exception
   */
  @Test
  @Ignore
  public void testConvert2coo() throws Exception {
    throw new RuntimeException("not yet implemented");
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#full()}.
   *
   * @throws Exception the exception
   */
  @Test
  @Ignore
  public void testFull() throws Exception {
    throw new RuntimeException("not yet implemented");
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj#luSolve(com.github.celldynamics.jcudarandomwalk.matrices.dense.IDenseVector, boolean, int, float)}.
   *
   * @throws Exception the exception
   */
  @Test
  @Ignore
  public void testLuSolve() throws Exception {
    throw new RuntimeException("not yet implemented");
  }

}
