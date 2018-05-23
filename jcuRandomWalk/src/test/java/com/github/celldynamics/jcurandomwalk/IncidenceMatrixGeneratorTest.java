package com.github.celldynamics.jcurandomwalk;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.contains;

import java.io.File;
import java.util.Arrays;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
   * @throws Exception on error
   */
  @Test
  public void testIncidenceMatrix() throws Exception {
    LOGGER.debug(
            ArrayTools.printArray(ArrayTools.array2Object(stack.getProcessor(1).getFloatArray())));
    LOGGER.debug(
            ArrayTools.printArray(ArrayTools.array2Object(stack.getProcessor(2).getFloatArray())));
    IncidenceMatrixGenerator obj = new IncidenceMatrixGenerator(stack);
    double[][] f = obj.getIncidence().full();
    LOGGER.debug(ArrayTools.printArray(ArrayTools.array2Object(f)));
    double[][] w = obj.getWeights().full();
    LOGGER.trace(ArrayTools.printArray(ArrayTools.array2Object(w)));

    assertThat(f[0].length, is(IncidenceMatrixGenerator.getEdgesNumber(height, width, nz)));
    assertThat(f.length, is(IncidenceMatrixGenerator.getNodesNumber(height, width, nz)));
    // sum of each row is 0
    for (int r = 0; r < IncidenceMatrixGenerator.getEdgesNumber(height, width, nz); r++) {
      double s = 0;
      for (int c = 0; c < IncidenceMatrixGenerator.getNodesNumber(height, width, nz); c++) {
        s += f[c][r];
      }
      assertThat(s, closeTo(0.0, 1e-6));
    }

  }

  /**
   * Test of
   * {@link com.github.celldynamics.jcurandomwalk.IncidenceMatrixGenerator#lin20ind(int, int, int, int, int[])}.
   * 
   * <p>Address stack by linear index and get pixel value from obtained x,y,z coords which should be
   * the same as index.
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
  }
}
