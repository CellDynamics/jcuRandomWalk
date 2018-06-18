package com.github.celldynamics.jcurandomwalk;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;

import java.io.File;

import org.junit.Test;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.StackStatistics;

/**
 * @author baniuk
 *
 */
public class StackPreprocessorTest {

  /**
   * The tmpdir.
   */
  static String tmpdir = System.getProperty("java.io.tmpdir") + File.separator;

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.StackPreprocessor#processStack(ImageStack)}.
   * 
   * <p>Process stack, save it to tmp and check if minmax is in range 0-1
   *
   * @throws Exception the exception
   */
  @Test
  public void testProcessStack() throws Exception {
    ImagePlus teststack = IJ.openImage("src/test/test_data/Stack_cut.tif");
    ImageStack ret = new StackPreprocessor().processStack(teststack.getImageStack());
    IJ.saveAsTiff(new ImagePlus("", ret), tmpdir + "testProcessStack.tiff");
    StackStatistics st = new StackStatistics(new ImagePlus("", ret));
    assertThat(st.min, closeTo(0.0, 1e-8));
    assertThat(st.max, closeTo(1.0, 1e-8));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.StackPreprocessor#getMean(ImageStack)}.
   * 
   * @throws Exception
   */
  @Test
  public void testGetMean() throws Exception {
    ImageStack stack = ImageStack.create(5, 5, 3, 8);
    stack.setVoxel(0, 0, 0, 1);
    stack.setVoxel(0, 0, 1, 2);
    stack.setVoxel(0, 1, 2, 3);
    double ret = new StackPreprocessor().getMean(stack);
    assertThat(ret, closeTo(6.0 / (5 * 5 * 3), 1e-8));
  }

}
