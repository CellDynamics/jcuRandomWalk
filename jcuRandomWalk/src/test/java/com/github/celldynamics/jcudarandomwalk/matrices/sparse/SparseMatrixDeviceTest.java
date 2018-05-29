package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;

import java.util.Arrays;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcurandomwalk.ArrayTools;
import com.github.celldynamics.jcurandomwalk.TestDataGenerators;

import jcuda.jcusparse.JCusparse;
import jcuda.runtime.JCuda;

/**
 * @author p.baniukiewicz
 *
 */
public class SparseMatrixDeviceTest {

  /**
   * The Constant LOGGER.
   */
  static final Logger LOGGER = LoggerFactory.getLogger(SparseMatrixDeviceTest.class.getName());

  private SparseMatrixDevice obj;
  private SparseMatrixDevice obj1;

  private TestDataGenerators gen;

  /**
   * Enable exceptions.
   */
  @BeforeClass
  public static void before() {
    JCusparse.setExceptionsEnabled(true);
    JCuda.setExceptionsEnabled(true);
    JCusparse.cusparseCreate(SparseMatrixDevice.handle);
  }

  /**
   * Disable exceptions.
   */
  @AfterClass
  public static void after() {
    // cusparseDestroy(SparseMatrixDevice.handle);
    JCusparse.setExceptionsEnabled(false);
    JCuda.setExceptionsEnabled(false);
    JCusparse.cusparseDestroy(SparseMatrixDevice.handle);
  }

  /**
   * @throws java.lang.Exception
   */
  @Before
  public void setUp() throws Exception {
    gen = new TestDataGenerators();
    obj = new SparseMatrixDevice(gen.rowInd, gen.colInd, gen.val,
            SparseMatrixType.MATRIX_FORMAT_COO);
    obj1 = new SparseMatrixDevice(gen.rowInd1, gen.colInd1, gen.val1,
            SparseMatrixType.MATRIX_FORMAT_COO);
  }

  /**
   * @throws java.lang.Exception
   */
  @After
  public void tearDown() throws Exception {
    obj.free();
    obj1.free();
    obj = null;
    obj1 = null;
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice#SparseMatrixDevice(int[], int[], double[],SparseMatrixType)}.
   * 
   * <p>Check array sizes in COO mode.
   * 
   * @throws Exception
   */
  @Test(expected = IllegalArgumentException.class)
  public void testSparseMatrixDeviceIntArrayIntArrayDoubleArray() throws Exception {
    @SuppressWarnings("unused")
    SparseMatrixDevice spd = new SparseMatrixDevice(new int[] { 1, 2 }, new int[] { 1, 2, 3 },
            new float[] { 1, 2 }, SparseMatrixType.MATRIX_FORMAT_COO);
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice#SparseMatrixDevice(int[], int[], double[],SparseMatrixType)}.
   * 
   * <p>Check array sizes in COO mode.
   * 
   * @throws Exception
   */
  @Test(expected = IllegalArgumentException.class)
  public void testSparseMatrixDeviceIntArrayIntArrayDoubleArray_1() throws Exception {
    @SuppressWarnings("unused")
    SparseMatrixDevice spd = new SparseMatrixDevice(new int[] { 1, 2, 4 }, new int[] { 1, 2, 3 },
            new float[] { 1, 2 }, SparseMatrixType.MATRIX_FORMAT_COO);
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice#SparseMatrixDevice(int[], int[], double[],SparseMatrixType)}.
   * 
   * <p>Check array sizes in COO mode.
   * 
   * @throws Exception
   */
  @Test(expected = IllegalArgumentException.class)
  public void testSparseMatrixDeviceIntArrayIntArrayDoubleArray_2() throws Exception {
    @SuppressWarnings("unused")
    SparseMatrixDevice spd = new SparseMatrixDevice(new int[] { 1, 2 }, new int[] { 1, 2, 3 },
            new float[] { 1, 2, 3 }, SparseMatrixType.MATRIX_FORMAT_COO);
  }

  /**
   * Read created array from device without retrieving.
   * 
   * @throws Exception
   */
  @Test
  public void testGetRowInd() throws Exception {
    int[] r = obj.getRowInd();
    assertThat(Arrays.asList(r), contains(gen.rowInd));
  }

  /**
   * Read created array from device without retrieving.
   * 
   * @throws Exception
   */
  @Test
  public void testGetColInd() throws Exception {
    int[] c = obj.getColInd();
    assertThat(Arrays.asList(c), contains(gen.colInd));
  }

  /**
   * Read created array from device without retrieving.
   * 
   * @throws Exception
   */
  @Test
  public void testgetVal() throws Exception {
    float[] v = obj.getVal();
    assertThat(Arrays.asList(v), contains(gen.val));
  }

  /**
   * Read created array from device.
   * 
   * @throws Exception
   */
  @Test
  public void testGetRowInd_1() throws Exception {
    obj.retrieveFromDevice();
    assertThat(Arrays.asList(obj.getRowInd()), contains(gen.rowInd));
  }

  /**
   * Read created array from device.
   * 
   * @throws Exception
   */
  @Test
  public void testGetColInd_1() throws Exception {
    obj.retrieveFromDevice();
    assertThat(Arrays.asList(obj.getColInd()), contains(gen.colInd));
  }

  /**
   * Read created array from device.
   * 
   * @throws Exception
   */
  @Test()
  public void testgetVal_1() throws Exception {
    obj.retrieveFromDevice();
    assertThat(Arrays.asList(obj.getVal()), contains(gen.val));
    LOGGER.debug(ArrayTools.printArray(ArrayTools.array2Object(obj.full())));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice#toCpu()}.
   * 
   * @throws Exception
   */
  @Test
  public void testToSparseMatrixHost() throws Exception {
    ISparseMatrix sph = obj.toCpu();
    assertThat(Arrays.asList(sph.getVal()), contains(gen.val));
    assertThat(Arrays.asList(sph.getColInd()), contains(gen.colInd));
    assertThat(Arrays.asList(sph.getRowInd()), contains(gen.rowInd));
    assertThat(sph.getElementNumber(), is(gen.val.length));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice#multiply(com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrixGpu)}.
   * 
   * @throws Exception
   */
  @Test
  public void testMultiply() throws Exception {
    ISparseMatrix objcsr = obj.convert2csr();
    ISparseMatrix obj1csr = obj1.convert2csr();
    ISparseMatrix ret = (ISparseMatrix) objcsr.multiply(obj1csr);
    ISparseMatrix retcoo = ret.convert2coo();
    int[] r = retcoo.getRowInd();
    int[] c = retcoo.getColInd();
    float[] v = retcoo.getVal();
    // result of obj*obj in matlab
    assertThat(Arrays.asList(r), contains(new int[] { 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3 }));
    assertThat(Arrays.asList(c), contains(new int[] { 0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3 }));
    assertThat(Arrays.asList(v), contains(new float[] { 17.0f, 8.0f, 5.0f, 8.0f, 13.0f, 27.0f, 5.0f,
        138.0f, 48.0f, 27.0f, 48.0f, 117.0f }));
    assertThat(ret.getElementNumber(), is(12));
    // ((SparseMatrixDevice) objcsr).free(); // already freed
    // ((SparseMatrixDevice) obj1csr).free();
    ((SparseMatrixDevice) ret).free();
    // ((SparseMatrixDevice) retcoo).free();
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice#SparseMatrixDevice(int[], int[], double[], com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixType)}.
   * 
   * @throws Exception
   */
  @Test
  public void testSparseMatrixDevice() throws Exception {
    assertThat(obj.getElementNumber(), is(gen.val.length));
    assertThat(obj.getRowNumber(), is(4));
    assertThat(obj.getColNumber(), is(5));
    assertThat(obj.getSparseMatrixType(), is(SparseMatrixType.MATRIX_FORMAT_COO));

    assertThat(obj1.getElementNumber(), is(gen.val1.length));
    assertThat(obj1.getRowNumber(), is(5));
    assertThat(obj1.getColNumber(), is(4));
    assertThat(obj1.getSparseMatrixType(), is(SparseMatrixType.MATRIX_FORMAT_COO));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice#convert2csr()}.
   * 
   * @throws Exception
   */
  @Test
  public void testConvert2csr() throws Exception {
    ISparseMatrix ret = obj.convert2csr();

    assertThat(ret.getRowNumber(), is(4));
    assertThat(ret.getColNumber(), is(5));
    assertThat(ret.getElementNumber(), is(obj.getElementNumber()));
    assertThat(Arrays.asList(ret.getVal()), contains(obj.getVal()));
    assertThat(Arrays.asList(ret.getVal()), contains(gen.val));
    assertThat(Arrays.asList(ret.getColInd()), contains(gen.colInd)); // col does not change
    // from page 9 https://docs.nvidia.com/cuda/pdf/CUSPARSE_Library.pdf
    assertThat(Arrays.asList(ret.getRowInd()), contains(new int[] { 0, 2, 4, 7, 9 }));

    ISparseMatrix retret = ret.convert2coo(); // should be the same os original obj

    assertThat(Arrays.asList(retret.getVal()), contains(obj.getVal()));
    assertThat(Arrays.asList(retret.getRowInd()), contains(obj.getRowInd()));
    assertThat(Arrays.asList(retret.getColInd()), contains(obj.getColInd()));
    // ((SparseMatrixDevice) ret).free();
    // ((SparseMatrixDevice) retret).free();
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice#transpose()}.
   * 
   * @throws Exception
   */
  @Test
  public void testTranspose() throws Exception {
    ISparseMatrix tobj = obj.transpose().convert2coo();
    assertThat(tobj.getRowNumber(), is(obj1.getRowNumber()));
    assertThat(tobj.getColNumber(), is(obj1.getColNumber()));
    // resutls from manual transposition -> ob1
    assertThat(Arrays.asList(tobj.getColInd()), contains(gen.colInd1));
    assertThat(Arrays.asList(tobj.getRowInd()), contains(gen.rowInd1));
    assertThat(Arrays.asList(tobj.getVal()), contains(gen.val1));
    ((SparseMatrixDevice) tobj).free();

    // and from obj1 to obj
    tobj = obj1.transpose().convert2coo();
    assertThat(tobj.getRowNumber(), is(obj.getRowNumber()));
    assertThat(tobj.getColNumber(), is(obj.getColNumber()));
    // resutls from manual transposition -> ob1
    assertThat(Arrays.asList(tobj.getColInd()), contains(gen.colInd));
    assertThat(Arrays.asList(tobj.getRowInd()), contains(gen.rowInd));
    assertThat(Arrays.asList(tobj.getVal()), contains(gen.val));
    ((SparseMatrixDevice) tobj).free();

  }

}
