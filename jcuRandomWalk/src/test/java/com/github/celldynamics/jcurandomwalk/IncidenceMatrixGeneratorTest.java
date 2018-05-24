package com.github.celldynamics.jcurandomwalk;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.nullValue;

import java.io.File;
import java.util.Arrays;
import java.util.stream.DoubleStream;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrix;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import ij.ImageStack;

/**
 * Class test.
 * 
 * @author baniu
 *
 */
public class IncidenceMatrixGeneratorTest {

  static {
    LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();
    Logger rootLogger = loggerContext.getLogger(IncidenceMatrixGeneratorTest.class.getName());
    ((ch.qos.logback.classic.Logger) rootLogger).setLevel(Level.TRACE);
  }
  /**
   * The Constant LOGGER.
   */
  static final Logger LOGGER =
          LoggerFactory.getLogger(IncidenceMatrixGeneratorTest.class.getName());

  // dimensions of test stack
  private int width = 3;
  private int height = 4;
  private int nz = 2;
  private ImageStack stack;

  @Rule
  public TemporaryFolder folder = new TemporaryFolder();

  /**
   * Setup test 3d stack.
   * 
   * @throws java.lang.Exception on error
   */
  @Before
  public void setUp() throws Exception {
    stack = TestDataGenerators.getTestStack(width, height, nz, "double");
    LOGGER.trace(stack.toString());
    LOGGER.trace(Arrays.deepToString(stack.getProcessor(1).getFloatArray()));
    LOGGER.trace(Arrays.deepToString(stack.getProcessor(2).getFloatArray()));

  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.IncidenceMatrixGenerator#IncidenceMatrixGenerator(ImageStack)}.
   * 
   * <p>Compute incidence matrix for {@link TestDataGenerators#getTestStack(int, int, int, String)}
   * and then check if number of rows and columns is proper and if sum of elements in each row gives
   * 0.
   * <p>Check also min and max for each row that should be -1 and 1.
   * 
   * <p>Check sizes of weights and incidence.
   * 
   * @throws Exception on error
   */
  @Test
  public void testIncidenceMatrix() throws Exception {
    LOGGER.debug(
            ArrayTools.printArray(ArrayTools.array2Object(stack.getProcessor(1).getFloatArray())));
    LOGGER.debug(
            ArrayTools.printArray(ArrayTools.array2Object(stack.getProcessor(2).getFloatArray())));
    IncidenceMatrixGenerator obj = new IncidenceMatrixGenerator(stack);
    // check sizes
    ISparseMatrix incid = obj.getIncidence();
    assertThat(incid.getRowNumber(), is(46)); // numbers for tes stack
    assertThat(incid.getColNumber(), is(24));
    ISparseMatrix weights = obj.getWeights();
    assertThat(weights.getRowNumber(), is(46));
    assertThat(weights.getColNumber(), is(46));

    double[][] f = incid.full();
    LOGGER.debug(ArrayTools.printArray(ArrayTools.array2Object(f)));
    LOGGER.debug("Incidence: " + incid.toString());
    double[][] w = weights.full();
    LOGGER.trace(ArrayTools.printArray(ArrayTools.array2Object(w)));

    assertThat(f[0].length, is(IncidenceMatrixGenerator.getEdgesNumber(height, width, nz)));
    assertThat(f.length, is(IncidenceMatrixGenerator.getNodesNumber(height, width, nz)));
    // sum of each row is 0
    // check min and max, need to extract each row doe to column ordering in 2d array
    for (int r = 0; r < IncidenceMatrixGenerator.getEdgesNumber(height, width, nz); r++) {
      double s = 0;
      double[] row = new double[IncidenceMatrixGenerator.getNodesNumber(height, width, nz)];
      for (int c = 0; c < IncidenceMatrixGenerator.getNodesNumber(height, width, nz); c++) {
        s += f[c][r];
        row[c] = f[c][r];
      }
      assertThat(s, closeTo(0.0, 1e-6));
      double min = DoubleStream.of(row).min().getAsDouble();
      assertThat(min, closeTo(-1.0, 1e-6));
      double max = DoubleStream.of(row).max().getAsDouble();
      assertThat(max, closeTo(1.0, 1e-6));
    }
  }

  /**
   * Test of
   * {@link com.github.celldynamics.jcurandomwalk.IncidenceMatrixGenerator#lin20ind(int, int, int, int, int[])}.
   * 
   * <p>Address stack by linear index and get pixel value from obtained x,y,z coords which should be
   * the same as index for test stack
   * {@link TestDataGenerators#getTestStack(int, int, int, String)}.
   * 
   * @throws Exception on error
   */
  @Test
  public void testLin20ind() throws Exception {
    int[] ind = new int[3];
    for (int lin = 0; lin < width * height * nz; lin++) {
      IncidenceMatrixGenerator.lin20ind(lin, height, width, nz, ind);
      assertThat(stack.getVoxel(ind[1], ind[0], ind[2]), is((double) lin)); // ind[1] - x
    }
  }

  /**
   * Test of
   * {@link com.github.celldynamics.jcurandomwalk.IncidenceMatrixGenerator#ind20lin(int[], int, int, int)}.
   * 
   * <p>Check if Lin20ind==Ind20lin
   * 
   * @throws Exception on error
   */
  @Test
  public void testInd20lin() throws Exception {
    int[] ind = new int[3];
    for (int lin = 0; lin < width * height * nz; lin++) {
      IncidenceMatrixGenerator.lin20ind(lin, height, width, nz, ind);
      assertThat(IncidenceMatrixGenerator.ind20lin(ind, height, width, nz), is(lin));
    }
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.IncidenceMatrixGenerator#saveObject(java.lang.String)}.
   * 
   * Save and then restore object. compare both.
   * 
   * @throws Exception
   */
  @Test
  public void testSaveObject() throws Exception {
    IncidenceMatrixGenerator obj = new IncidenceMatrixGenerator(stack);
    double[][] objFull = obj.getIncidence().full();
    File filename = folder.newFile();
    obj.saveObject(filename.toString());
    // restore
    IncidenceMatrixGenerator restored = IncidenceMatrixGenerator.restoreObject(filename.toString());
    double[][] resFull = restored.getIncidence().full();
    for (int c = 0; c < resFull.length; c++) {
      assertThat(Arrays.asList(resFull[c]), contains(objFull[c]));
    }
    assertThat(restored.getWeights(), is(not(nullValue())));
    assertThat(restored.getSinkBox(), is(not(nullValue())));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.IncidenceMatrixGenerator#getEdgesNumber(int, int, int)}.
   * 
   * <p>Compare with {@link TestDataGenerators#getTestStack(int, int, int, String)}
   * 
   * @throws Exception
   */
  @Test
  public void testGetEdgesNumber() throws Exception {
    assertThat(IncidenceMatrixGenerator.getEdgesNumber(stack.getHeight(), stack.getWidth(),
            stack.getSize()), is(46));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.IncidenceMatrixGenerator#getNodesNumber(int, int, int)}.
   * 
   * <p>Compare with {@link TestDataGenerators#getTestStack(int, int, int, String)}
   * 
   * @throws Exception
   */
  @Test
  public void testGetNodesNumber() throws Exception {
    assertThat(IncidenceMatrixGenerator.getNodesNumber(stack.getHeight(), stack.getWidth(),
            stack.getSize()), is(24));
  }

  /**
   * Test method for
   * {@link IncidenceMatrixGenerator#computeWeight(ij.ImageStack, int[], int[], double, double, double)}.
   * 
   * <p>Compute weight and compare result with jcuRandomWalk/JCudaMatrix/Matlab/tests.java
   * 
   * @throws Exception
   */
  @Test
  public void testComputeWeight() throws Exception {
    IncidenceMatrixGenerator obj = new IncidenceMatrixGenerator();
    stack.setVoxel(0, 0, 0, 2.0); // x,y,z
    stack.setVoxel(1, 0, 0, 2.5);
    double sigmaGrad = 0.1;
    double sigmaMean = 1e6;
    double meanSource = 0.6;
    // compute between left pper corner and next to it on right
    double ret = obj.computeWeight(stack, new int[] { 0, 0, 0 }, new int[] { 0, 1, 0 }, sigmaGrad,
            sigmaMean, meanSource);
    assertThat(ret, is(closeTo(3.7267e-06, 1e-10)));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.IncidenceMatrixGenerator#computeSinkBox()}.
   * 
   * <p>Resuts generated in %% computeSinkBox testsJava.m
   * 
   * @throws Exception
   */
  @Test
  public void testComputeSinkBox() throws Exception {
    int width = 4;
    int height = 3;
    int nz = 3;
    stack = TestDataGenerators.getTestStack(width, height, nz, "double");
    IncidenceMatrixGenerator obj = new IncidenceMatrixGenerator(stack);
    obj.computeSinkBox();
    int[] b = obj.getSinkBox();
    LOGGER.trace("BBA: " + ArrayTools
            .printArray(ArrayTools.array2Object(stack.getProcessor(1).getFloatArray())));
    LOGGER.trace("BBA: " + ArrayTools
            .printArray(ArrayTools.array2Object(stack.getProcessor(2).getFloatArray())));
    LOGGER.trace("BBA: " + ArrayTools
            .printArray(ArrayTools.array2Object(stack.getProcessor(3).getFloatArray())));
    LOGGER.debug("BB: " + ArrayUtils.toString(b));
    assertThat(Arrays.asList(b), contains(new int[] { 12, 21, 13, 22, 14, 23, 15, 17, 18, 20, 0, 3,
        6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 24, 27, 30, 33, 25, 28, 31, 34, 26, 29, 32, 35 }));
  }
}
