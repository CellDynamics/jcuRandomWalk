package com.github.celldynamics.jcurandomwalk;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.junit.Assume.assumeTrue;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import ij.ImageStack;
import jcuda.jcusparse.JCusparse;

// TODO: Auto-generated Javadoc
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
    ((ch.qos.logback.classic.Logger) rootLogger).setLevel(Level.DEBUG);
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
  static final Logger LOGGER =
          LoggerFactory.getLogger(IncidenceMatrixGeneratorTest.class.getName());

  /** The width. */
  // dimensions of test stack
  private int width = 3;

  /** The height. */
  private int height = 4;

  /** The nz. */
  private int nz = 2;

  /** The stack. */
  private ImageStack stack;

  /** The folder. */
  @Rule
  public TemporaryFolder folder = new TemporaryFolder();

  /**
   * Setup test 3d stack.
   *
   * @throws Exception the exception
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
    assumeTrue(isCuda);
    RandomWalkOptions options = new RandomWalkOptions();
    options.algOptions.meanSource = 0.6;
    LOGGER.debug(
            ArrayTools.printArray(ArrayTools.array2Object(stack.getProcessor(1).getFloatArray())));
    LOGGER.debug(
            ArrayTools.printArray(ArrayTools.array2Object(stack.getProcessor(2).getFloatArray())));
    IncidenceMatrixGenerator obj = new IncidenceMatrixGenerator(stack, options.getAlgOptions());
    // check sizes
    // SparseMatrixDevice incidence = SparseMatrixDevice.factory(obj.getIncidence());
    assertThat(obj.getIncidence().getRowNumber(), is(46)); // numbers for tes stack
    assertThat(obj.getIncidence().getColNumber(), is(24));
    // SparseMatrixDevice weights = SparseMatrixDevice.factory(obj.getWeights());
    assertThat(obj.getWeights().getRowNumber(), is(46));
    assertThat(obj.getWeights().getColNumber(), is(46));

    double[][] f = obj.getIncidence().full();
    LOGGER.debug(ArrayTools.printArray(ArrayTools.array2Object(f)));
    LOGGER.debug("Incidence: " + obj.getIncidence().toString());
    double[][] w = obj.getWeights().full();
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
   * @throws Exception the exception
   */
  @Test
  public void testSaveObject() throws Exception {
    // TODO use factory to get Host from incidence
    RandomWalkOptions options = new RandomWalkOptions();
    options.algOptions.meanSource = 0.6;
    IncidenceMatrixGenerator obj = new IncidenceMatrixGenerator(stack, options.getAlgOptions());
    double[][] objFull = obj.getIncidence().full();
    File filename = folder.newFile();
    obj.saveObject(filename.toString());
    // restore
    IncidenceMatrixGenerator restored = IncidenceMatrixGenerator.restoreObject(filename.toString(),
            stack, options.getAlgOptions());

    double[][] resFull = restored.getIncidence().full();
    for (int c = 0; c < resFull.length; c++) {
      assertThat(Arrays.asList(resFull[c]), contains(objFull[c]));
    }
    assertThat(Arrays.asList(restored.getWeights().getColInd()),
            contains(obj.getWeights().getColInd()));
    assertThat(Arrays.asList(restored.getWeights().getRowInd()),
            contains(obj.getWeights().getRowInd()));
    assertThat(Arrays.asList(restored.getWeights().getVal()), contains(obj.getWeights().getVal()));
    assertThat(Arrays.asList(restored.getSinkBox()), contains(obj.getSinkBox()));
    assertThat(restored.getWeights().getColNumber(), is(obj.getWeights().getColNumber()));
    assertThat(restored.getWeights().getRowNumber(), is(obj.getWeights().getRowNumber()));
    assertThat(restored.getWeights().getVal().length, is(obj.getWeights().getVal().length));

    assertThat(restored.getIncidence().getColNumber(), is(obj.getIncidence().getColNumber()));
    assertThat(restored.getIncidence().getRowNumber(), is(obj.getIncidence().getRowNumber()));
    assertThat(restored.getIncidence().getVal().length, is(obj.getIncidence().getVal().length));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.IncidenceMatrixGenerator#getEdgesNumber(int, int, int)}.
   * 
   * <p>Compare with {@link TestDataGenerators#getTestStack(int, int, int, String)}
   *
   * @throws Exception the exception
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
   * @throws Exception the exception
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
   * @throws Exception the exception
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
   * @throws Exception the exception
   */
  @Test
  public void testComputeSinkBox() throws Exception {
    int width = 4;
    int height = 3;
    int nz = 3;
    RandomWalkOptions options = new RandomWalkOptions();
    options.algOptions.meanSource = 0.6;
    stack = TestDataGenerators.getTestStack(width, height, nz, "double");
    IncidenceMatrixGenerator obj = new IncidenceMatrixGenerator(stack, options.getAlgOptions());
    obj.computeSinkBox();
    Integer[] b = obj.getSinkBox();
    List<Integer> blist = Arrays.asList(b);
    boolean issorted = blist.stream().sorted().collect(Collectors.toList()).equals(blist);
    assertThat(issorted, is(true));
    // Arrays.sort(b);
    LOGGER.trace("BBA: " + ArrayTools
            .printArray(ArrayTools.array2Object(stack.getProcessor(1).getFloatArray())));
    LOGGER.trace("BBA: " + ArrayTools
            .printArray(ArrayTools.array2Object(stack.getProcessor(2).getFloatArray())));
    LOGGER.trace("BBA: " + ArrayTools
            .printArray(ArrayTools.array2Object(stack.getProcessor(3).getFloatArray())));
    LOGGER.trace("BB: " + ArrayUtils.toString(b));
    assertThat(blist, containsInAnyOrder(new Integer[] { 12, 21, 13, 22, 14, 23, 15, 17, 18, 20, 0,
        3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 24, 27, 30, 33, 25, 28, 31, 34, 26, 29, 32, 35 }));
  }

  /**
   * Compute new weights for known geometry.
   * 
   * <p>Input stack has 4 rows and 5 columns and 3 slices. Incidence matrix is generated traversing
   * along columns (rows first for specified column). E.G first three lines of incidence are:
   * 1 0 0 0 -1 0 0 0 0 0 ...
   * 1 -1 0 0 0 0 0.0 0.0 ..
   * 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1
   * That means right neighbour of pixel of index 0 is pixel of index 4, down neighbour is pixel of
   * index 1 and upper slice neighbour is pixel of index 20. Pixels are counted along columns:
   * 0 4 8 12 16
   * 1 5 9 13 17
   * 2 6 10 14 18
   * 3 7 11 15 19
   * 
   * <p>Which can be observed in weight vector: 0.011108992565355967, 0.6065306958646659,
   * 0.9950124886818522 which are computed for respective pixels
   * {@link TestDataGenerators#getTestStack_1()}
   * 
   * <p>The order of neighbours is R, D, U, next three rows of incidence matrix are for pixel of
   * number 1, which has Cartesian coordinates (1,0) (next row, the same column). Column index in
   * incidence matrix defines which pixels are taken to compute weight.
   *
   * @throws Exception the exception
   */
  @Test
  public void testAssignStack() throws Exception {
    RandomWalkOptions options = new RandomWalkOptions();
    options.algOptions.meanSource = 0.6;
    ImageStack ts = TestDataGenerators.getTestStack_1();
    IncidenceMatrixGenerator inc = new IncidenceMatrixGenerator(ts, options.getAlgOptions());
    inc.assignStack(ts);
    LOGGER.trace("Slice 1: "
            + ArrayTools.printArray(ArrayTools.array2Object(ts.getProcessor(1).getFloatArray())));
    // SparseMatrixDevice weights = SparseMatrixDevice.factory(inc.getWeights());
    // SparseMatrixDevice incidence = SparseMatrixDevice.factory(inc.getIncidence());
    // LOGGER.trace(ArrayTools.printArray(ArrayTools.array2Object(inicidence.full())));
    // LOGGER.trace(ArrayTools.printArray(ArrayTools.array2Object(weights.full())));
    LOGGER.trace("INC: " + inc.getIncidence());
    LOGGER.trace("WEI: " + inc.getWeights().toString());
    // Arrays.sort(inc.getSinkBox());
    LOGGER.trace("BBX: " + ArrayUtils.toString(inc.getSinkBox()));

    double[][] f = inc.getIncidence().full();
    LOGGER.debug(ArrayTools.printArray(ArrayTools.array2Object(f)));
    double[][] w = inc.getWeights().full();
    LOGGER.trace(ArrayTools.printArray(ArrayTools.array2Object(w)));

  }
}
