package com.github.celldynamics.jcudarandomwalk.matrices;

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
  private int[] rowInd;
  private int[] colInd;
  private double[] val;

  private SparseMatrixDevice obj1;
  private int[] rowInd1;
  private int[] colInd1;
  private double[] val1;

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
    rowInd = new int[] { 0, 0, 1, 1, 2, 2, 2, 3, 3 };
    colInd = new int[] { 0, 1, 1, 2, 0, 3, 4, 2, 4 };
    val = new double[] { 1, 4, 2, 3, 5, 7, 8, 9, 6 };
    obj = new SparseMatrixDevice(rowInd, colInd, val, SparseMatrixType.MATRIX_FORMAT_COO);

    rowInd1 = new int[] { 0, 0, 1, 1, 2, 2, 3, 4, 4 };
    colInd1 = new int[] { 0, 2, 0, 1, 1, 3, 2, 2, 3 };
    val1 = new double[] { 1, 5, 4, 2, 3, 9, 7, 8, 6 };
    obj1 = new SparseMatrixDevice(rowInd1, colInd1, val1, SparseMatrixType.MATRIX_FORMAT_COO);
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
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixDevice#SparseMatrixDevice(int[], int[], double[],SparseMatrixType)}.
   * 
   * <p>Check array sizes in COO mode.
   * 
   * @throws Exception
   */
  @Test(expected = IllegalArgumentException.class)
  public void testSparseMatrixDeviceIntArrayIntArrayDoubleArray() throws Exception {
    @SuppressWarnings("unused")
    SparseMatrixDevice spd = new SparseMatrixDevice(new int[] { 1, 2 }, new int[] { 1, 2, 3 },
            new double[] { 1, 2 }, SparseMatrixType.MATRIX_FORMAT_COO);
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixDevice#SparseMatrixDevice(int[], int[], double[],SparseMatrixType)}.
   * 
   * <p>Check array sizes in COO mode.
   * 
   * @throws Exception
   */
  @Test(expected = IllegalArgumentException.class)
  public void testSparseMatrixDeviceIntArrayIntArrayDoubleArray_1() throws Exception {
    @SuppressWarnings("unused")
    SparseMatrixDevice spd = new SparseMatrixDevice(new int[] { 1, 2, 4 }, new int[] { 1, 2, 3 },
            new double[] { 1, 2 }, SparseMatrixType.MATRIX_FORMAT_COO);
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixDevice#SparseMatrixDevice(int[], int[], double[],SparseMatrixType)}.
   * 
   * <p>Check array sizes in COO mode.
   * 
   * @throws Exception
   */
  @Test(expected = IllegalArgumentException.class)
  public void testSparseMatrixDeviceIntArrayIntArrayDoubleArray_2() throws Exception {
    @SuppressWarnings("unused")
    SparseMatrixDevice spd = new SparseMatrixDevice(new int[] { 1, 2 }, new int[] { 1, 2, 3 },
            new double[] { 1, 2, 3 }, SparseMatrixType.MATRIX_FORMAT_COO);
  }

  /**
   * Read created array from device without retrieving.
   * 
   * @throws Exception
   */
  @Test
  public void testGetRowInd() throws Exception {
    int[] r = obj.getRowInd();
    assertThat(Arrays.asList(r), contains(rowInd));
  }

  /**
   * Read created array from device without retrieving.
   * 
   * @throws Exception
   */
  @Test
  public void testGetColInd() throws Exception {
    int[] c = obj.getColInd();
    assertThat(Arrays.asList(c), contains(colInd));
  }

  /**
   * Read created array from device without retrieving.
   * 
   * @throws Exception
   */
  @Test
  public void testgetVal() throws Exception {
    double[] v = obj.getVal();
    assertThat(Arrays.asList(v), contains(val));
  }

  /**
   * Read created array from device.
   * 
   * @throws Exception
   */
  @Test
  public void testGetRowInd_1() throws Exception {
    obj.retrieveFromDevice();
    assertThat(Arrays.asList(obj.getRowInd()), contains(rowInd));
  }

  /**
   * Read created array from device.
   * 
   * @throws Exception
   */
  @Test
  public void testGetColInd_1() throws Exception {
    obj.retrieveFromDevice();
    assertThat(Arrays.asList(obj.getColInd()), contains(colInd));
  }

  /**
   * Read created array from device.
   * 
   * @throws Exception
   */
  @Test()
  public void testgetVal_1() throws Exception {
    obj.retrieveFromDevice();
    assertThat(Arrays.asList(obj.getVal()), contains(val));
    LOGGER.debug(ArrayTools.printArray(ArrayTools.array2Object(obj.full())));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixDevice#toCpu()}.
   * 
   * @throws Exception
   */
  @Test
  public void testToSparseMatrixHost() throws Exception {
    ISparseMatrix sph = obj.toCpu();
    assertThat(Arrays.asList(sph.getVal()), contains(val));
    assertThat(Arrays.asList(sph.getColInd()), contains(colInd));
    assertThat(Arrays.asList(sph.getRowInd()), contains(rowInd));
    assertThat(sph.getElementNumber(), is(9));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixDevice#multiply(com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrixGpu)}.
   * 
   * @throws Exception
   */
  @Test
  public void testMultiply() throws Exception {
    ISparseMatrix objcsr = obj.convert2csr();
    ISparseMatrix obj1csr = obj1.convert2csr();
    ISparseMatrix ret = objcsr.multiply(obj1csr);
    ISparseMatrix retcoo = ret.convert2coo();
    int[] r = retcoo.getRowInd();
    int[] c = retcoo.getColInd();
    double[] v = retcoo.getVal();
    // result of obj*obj in matlab
    assertThat(Arrays.asList(r), contains(new int[] { 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3 }));
    assertThat(Arrays.asList(c), contains(new int[] { 0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3 }));
    assertThat(Arrays.asList(v), contains(
            new double[] { 17.0, 8.0, 5.0, 8.0, 13.0, 27.0, 5.0, 138.0, 48.0, 27.0, 48.0, 117.0 }));
    assertThat(ret.getElementNumber(), is(12));
    // ((SparseMatrixDevice) objcsr).free(); // already freed
    // ((SparseMatrixDevice) obj1csr).free();
    ((SparseMatrixDevice) ret).free();
    // ((SparseMatrixDevice) retcoo).free();
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixDevice#SparseMatrixDevice(int[], int[], double[], com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixType)}.
   * 
   * @throws Exception
   */
  @Test
  public void testSparseMatrixDeviceIntArrayIntArrayDoubleArraySparseMatrixType() throws Exception {
    assertThat(obj.getElementNumber(), is(val.length));
    assertThat(obj.getRowNumber(), is(4));
    assertThat(obj.getColNumber(), is(5));
    assertThat(obj.getSparseMatrixType(), is(SparseMatrixType.MATRIX_FORMAT_COO));

    assertThat(obj1.getElementNumber(), is(val1.length));
    assertThat(obj1.getRowNumber(), is(5));
    assertThat(obj1.getColNumber(), is(4));
    assertThat(obj1.getSparseMatrixType(), is(SparseMatrixType.MATRIX_FORMAT_COO));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixDevice#convert2csr()}.
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
    assertThat(Arrays.asList(ret.getVal()), contains(val));
    assertThat(Arrays.asList(ret.getColInd()), contains(colInd)); // col does not change
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
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixDevice#transpose()}.
   * 
   * @throws Exception
   */
  @Test
  public void testTranspose() throws Exception {
    ISparseMatrix tobj = obj.transpose().convert2coo();
    assertThat(tobj.getRowNumber(), is(obj1.getRowNumber()));
    assertThat(tobj.getColNumber(), is(obj1.getColNumber()));
    // resutls from manual transposition -> ob1
    assertThat(Arrays.asList(tobj.getColInd()), contains(colInd1));
    assertThat(Arrays.asList(tobj.getRowInd()), contains(rowInd1));
    assertThat(Arrays.asList(tobj.getVal()), contains(val1));
    ((SparseMatrixDevice) tobj).free();

    // and from obj1 to obj
    tobj = obj1.transpose().convert2coo();
    assertThat(tobj.getRowNumber(), is(obj.getRowNumber()));
    assertThat(tobj.getColNumber(), is(obj.getColNumber()));
    // resutls from manual transposition -> ob1
    assertThat(Arrays.asList(tobj.getColInd()), contains(colInd));
    assertThat(Arrays.asList(tobj.getRowInd()), contains(rowInd));
    assertThat(Arrays.asList(tobj.getVal()), contains(val));
    ((SparseMatrixDevice) tobj).free();

  }

}
