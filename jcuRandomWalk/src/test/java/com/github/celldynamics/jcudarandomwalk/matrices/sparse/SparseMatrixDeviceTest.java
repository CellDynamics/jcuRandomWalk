package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import static com.github.baniuk.ImageJTestSuite.matchers.arrays.ArrayMatchers.arrayCloseTo;
import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.junit.Assume.assumeTrue;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVectorDevice;
import com.github.celldynamics.jcurandomwalk.ArrayTools;
import com.github.celldynamics.jcurandomwalk.TestDataGenerators;

import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.runtime.JCuda;

/**
 * GPU related tests.
 * 
 * @author p.baniukiewicz
 *
 */
public class SparseMatrixDeviceTest {

  /**
   * The Constant LOGGER.
   */
  static final Logger LOGGER = LoggerFactory.getLogger(SparseMatrixDeviceTest.class.getName());

  private static cusparseHandle handle;

  private SparseMatrixDevice obj;
  private SparseMatrixDevice obj1;

  private TestDataGenerators gen;

  /**
   * Check if there is cuda.
   * 
   * @return true if it is
   */
  public static boolean checkCuda() {
    try {
      JCusparse.setExceptionsEnabled(true);
    } catch (Error e) {
      return false;
    }
    return true;
  }

  private static final boolean isCuda = checkCuda();

  /**
   * Enable exceptions.
   */
  @BeforeClass
  public static void before() {
    if (isCuda) {
      handle = new cusparseHandle();
      JCusparse.setExceptionsEnabled(true);
      JCuda.setExceptionsEnabled(true);
      JCusparse.cusparseCreate(handle);
    }
  }

  /**
   * Disable exceptions.
   */
  @AfterClass
  public static void after() {
    if (isCuda) {
      // cusparseDestroy(SparseMatrixDevice.handle);
      JCusparse.setExceptionsEnabled(false);
      JCuda.setExceptionsEnabled(false);
      JCusparse.cusparseDestroy(handle);
    }
  }

  /**
   * @throws java.lang.Exception
   */
  @Before
  public void setUp() throws Exception {
    if (isCuda) {
      gen = new TestDataGenerators();
      obj = new SparseMatrixDevice(gen.rowInd, gen.colInd, gen.valRowOrder,
              SparseMatrixType.MATRIX_FORMAT_COO, handle);
      obj1 = new SparseMatrixDevice(gen.rowInd1, gen.colInd1, gen.val1RowOrder,
              SparseMatrixType.MATRIX_FORMAT_COO, handle);
    }
  }

  /**
   * @throws java.lang.Exception
   */
  @After
  public void tearDown() throws Exception {
    if (isCuda) {
      obj.free();
      obj1.free();
      obj = null;
      obj1 = null;
    }
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice#SparseMatrixDevice(int[], int[], float[], SparseMatrixType)}.
   * 
   * <p>Check array sizes in COO mode.
   * 
   * @throws Exception
   */
  @Test(expected = IllegalArgumentException.class)
  public void testSparseMatrixDeviceIntArrayIntArrayDoubleArray() throws Exception {
    assumeTrue(isCuda);
    @SuppressWarnings("unused")
    SparseMatrixDevice spd = new SparseMatrixDevice(new int[] { 1, 2 }, new int[] { 1, 2, 3 },
            new float[] { 1, 2 }, SparseMatrixType.MATRIX_FORMAT_COO, handle);
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice#SparseMatrixDevice(int[], int[], float[], SparseMatrixType)}.
   * 
   * <p>Check array sizes in COO mode.
   * 
   * @throws Exception
   */
  @Test(expected = IllegalArgumentException.class)
  public void testSparseMatrixDeviceIntArrayIntArrayDoubleArray_1() throws Exception {
    assumeTrue(isCuda);
    @SuppressWarnings("unused")
    SparseMatrixDevice spd = new SparseMatrixDevice(new int[] { 1, 2, 4 }, new int[] { 1, 2, 3 },
            new float[] { 1, 2 }, SparseMatrixType.MATRIX_FORMAT_COO, handle);
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice#SparseMatrixDevice(int[], int[], float[], SparseMatrixType)}.
   * 
   * <p>Check array sizes in COO mode.
   * 
   * @throws Exception
   */
  @Test(expected = IllegalArgumentException.class)
  public void testSparseMatrixDeviceIntArrayIntArrayDoubleArray_2() throws Exception {
    assumeTrue(isCuda);
    @SuppressWarnings("unused")
    SparseMatrixDevice spd = new SparseMatrixDevice(new int[] { 1, 2 }, new int[] { 1, 2, 3 },
            new float[] { 1, 2, 3 }, SparseMatrixType.MATRIX_FORMAT_COO, handle);
  }

  /**
   * Read created array from device without retrieving.
   * 
   * @throws Exception
   */
  @Test
  public void testGetRowInd() throws Exception {
    assumeTrue(isCuda);
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
    assumeTrue(isCuda);
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
    assumeTrue(isCuda);
    float[] v = obj.getVal();
    assertThat(Arrays.asList(v), contains(gen.valRowOrder));
  }

  /**
   * Read created array from device.
   * 
   * @throws Exception
   */
  @Test
  public void testGetRowInd_1() throws Exception {
    assumeTrue(isCuda);
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
    assumeTrue(isCuda);
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
    assumeTrue(isCuda);
    obj.retrieveFromDevice();
    assertThat(Arrays.asList(obj.getVal()), contains(gen.valRowOrder));
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
    assumeTrue(isCuda);
    obj.toCpu(false);
    assertThat(Arrays.asList(obj.getVal()), contains(gen.valRowOrder));
    assertThat(Arrays.asList(obj.getColInd()), contains(gen.colInd));
    assertThat(Arrays.asList(obj.getRowInd()), contains(gen.rowInd));
    assertThat(obj.getElementNumber(), is(gen.valRowOrder.length));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice#multiply(com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrixGpu)}.
   * 
   * @throws Exception
   */
  @Test
  public void testMultiply() throws Exception {
    assumeTrue(isCuda);
    SparseMatrixDevice objcsr = obj.convert2csr();
    SparseMatrixDevice obj1csr = obj1.convert2csr();
    SparseMatrixDevice ret = objcsr.multiply(obj1csr);
    SparseMatrixDevice retcoo = ret.convert2coo();
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
    assumeTrue(isCuda);
    assertThat(obj.getElementNumber(), is(gen.valRowOrder.length));
    assertThat(obj.getRowNumber(), is(4));
    assertThat(obj.getColNumber(), is(5));
    assertThat(obj.getSparseMatrixType(), is(SparseMatrixType.MATRIX_FORMAT_COO));

    assertThat(obj1.getElementNumber(), is(gen.val1RowOrder.length));
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
    assumeTrue(isCuda);
    SparseMatrixDevice ret = obj.convert2csr();
    // obj.free(); // can not be free here
    assertThat(ret.getRowNumber(), is(4));
    assertThat(ret.getColNumber(), is(5));
    assertThat(ret.getElementNumber(), is(obj.getElementNumber()));
    assertThat(Arrays.asList(ret.getVal()), contains(obj.getVal()));
    assertThat(Arrays.asList(ret.getVal()), contains(gen.valRowOrder));
    assertThat(Arrays.asList(ret.getColInd()), contains(gen.colInd)); // col does not change
    // from page 9 https://docs.nvidia.com/cuda/pdf/CUSPARSE_Library.pdf
    assertThat(Arrays.asList(ret.getRowInd()), contains(new int[] { 0, 2, 4, 7, 9 }));

    ret.toCpu(true);
    ret.toGpu();
    SparseMatrixDevice retret = ret.convert2coo(); // should be the same os original obj

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
    assumeTrue(isCuda);
    SparseMatrixDevice tobj = obj.transpose().convert2coo();
    assertThat(tobj.getRowNumber(), is(obj1.getRowNumber()));
    assertThat(tobj.getColNumber(), is(obj1.getColNumber()));
    // resutls from manual transposition -> ob1
    assertThat(Arrays.asList(tobj.getColInd()), contains(gen.colInd1));
    assertThat(Arrays.asList(tobj.getRowInd()), contains(gen.rowInd1));
    assertThat(Arrays.asList(tobj.getVal()), contains(gen.val1RowOrder));
    ((SparseMatrixDevice) tobj).free();

    // and from obj1 to obj
    tobj = obj1.transpose().convert2coo();
    assertThat(tobj.getRowNumber(), is(obj.getRowNumber()));
    assertThat(tobj.getColNumber(), is(obj.getColNumber()));
    // resutls from manual transposition -> ob1
    assertThat(Arrays.asList(tobj.getColInd()), contains(gen.colInd));
    assertThat(Arrays.asList(tobj.getRowInd()), contains(gen.rowInd));
    assertThat(Arrays.asList(tobj.getVal()), contains(gen.valRowOrder));
    ((SparseMatrixDevice) tobj).free();

  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice#luSolve(com.github.celldynamics.jcudarandomwalk.matrices.dense.IDenseVector, boolean, int, float)}.
   * 
   * <p>testJava.m
   * 
   * @throws Exception
   * 
   */
  @Test
  public void testLuSolve() throws Exception {
    assumeTrue(isCuda);
    int[] rows =
            new int[] { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4 };
    int[] cols =
            new int[] { 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4 };
    float[] vals = new float[] { 0.9f, 0.4f, 0.1f, 0.9f, 0.1f, 0.1f, 0.9f, 0.9f, 0.6f, 0.2f, 0.4f,
        0.2f, 0.6f, 0.4f, 0.1f, 0.3f, 0.3f, 0.5f, 0.5f, 0.2f, 0.8f, 0.1f, 0.1f, 0.4f, 0.2f };
    DenseVectorDevice b = new DenseVectorDevice(5, 1, new float[] { 6.1f, 8f, 4.7f, 5.4f, 3.9f });

    SparseMatrixDevice a =
            new SparseMatrixDevice(rows, cols, vals, SparseMatrixType.MATRIX_FORMAT_COO, handle);
    SparseMatrixDevice acsr = a.convert2csr();

    float[] ret = acsr.luSolve(b, true, 50, 1e-12f);
    // convert to double
    Double[] retd =
            IntStream.range(0, ret.length).mapToDouble(i -> ret[i]).boxed().toArray(Double[]::new);

    a.free();
    b.free();
    acsr.free();
    assertThat(retd, arrayCloseTo(new double[] { 1, 2, 3, 4, 5 }, 1e-5));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice#toCpu()}.
   * 
   * @throws Exception
   */
  @Test
  public void testToCpuToGpu() throws Exception {
    assumeTrue(isCuda);
    SparseMatrixDevice objcsr = obj.convert2csr(); // on gpu
    SparseMatrixDevice obj1csr = obj1.convert2csr(); // on gpu
    SparseMatrixDevice ret = objcsr.multiply(obj1csr); // on gpu
    SparseMatrixDevice retcoo = ret.convert2coo(); // on gpu
    retcoo.toCpu(true);
    int[] r = retcoo.getRowInd();
    int[] c = retcoo.getColInd();
    float[] v = retcoo.getVal();
    // result of obj*obj in matlab
    assertThat(Arrays.asList(r), contains(new int[] { 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3 }));
    assertThat(Arrays.asList(c), contains(new int[] { 0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3 }));
    assertThat(Arrays.asList(v), contains(new float[] { 17.0f, 8.0f, 5.0f, 8.0f, 13.0f, 27.0f, 5.0f,
        138.0f, 48.0f, 27.0f, 48.0f, 117.0f }));
    assertThat(ret.getElementNumber(), is(12));

    // change something
    v[0] = 200;
    c[5] = 2;
    retcoo.toGpu();
    retcoo.toCpu(true);
    float[] v1 = retcoo.getVal();
    int[] c1 = retcoo.getColInd();
    assertThat(Arrays.asList(v1), contains(new float[] { 200.0f, 8.0f, 5.0f, 8.0f, 13.0f, 27.0f,
        5.0f, 138.0f, 48.0f, 27.0f, 48.0f, 117.0f }));
    assertThat(Arrays.asList(c1), contains(new int[] { 0, 1, 2, 0, 1, 2, 0, 2, 3, 1, 2, 3 }));
  }

}
