package com.github.celldynamics.jcurandomwalk;

import static com.github.baniuk.ImageJTestSuite.matchers.arrays.ArrayMatchers.arrayCloseTo;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assume.assumeTrue;

import java.util.stream.IntStream;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVectorDevice;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixType;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import jcuda.Pointer;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.runtime.JCuda;

/**
 * General notebook for jCuda tests.
 * 
 * @author p.baniukiewicz
 *
 */
public class CudaTest {

  private static cusparseHandle handle;

  static {
    LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();
    Logger rootLogger = loggerContext.getLogger(CudaTest.class.getName());
    ((ch.qos.logback.classic.Logger) rootLogger).setLevel(Level.DEBUG);
  }

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
   * The Constant LOGGER.
   */
  static final Logger LOGGER = LoggerFactory.getLogger(CudaTest.class.getName());

  @Test
  public void test() {
    assumeTrue(isCuda);
    JCuda.setExceptionsEnabled(true);
    Pointer pointer = new Pointer();
    JCuda.cudaMalloc(pointer, 4);
    LOGGER.debug("Pointer: " + pointer);
    JCuda.cudaFree(pointer);
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

    float[] ret = acsr.luSolve1(b, true, 50, 1e-12f);
    // convert to double
    Double[] retd =
            IntStream.range(0, ret.length).mapToDouble(i -> ret[i]).boxed().toArray(Double[]::new);

    a.free();
    b.free();
    acsr.free();
    assertThat(retd, arrayCloseTo(new double[] { 1, 2, 3, 4, 5 }, 1e-5));
  }

}
