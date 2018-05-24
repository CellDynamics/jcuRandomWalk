package com.github.celldynamics.jcurandomwalk;

import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

/**
 * @author baniu
 *
 */
public class TestDataGenerators {

  public int[] rowInd = new int[] { 0, 0, 1, 1, 2, 2, 2, 3, 3 };
  public int[] colInd = new int[] { 0, 1, 1, 2, 0, 3, 4, 2, 4 };
  public double[] val = new double[] { 1, 4, 2, 3, 5, 7, 8, 9, 6 };

  public int[] rowInd1 = new int[] { 0, 0, 1, 1, 2, 2, 3, 4, 4 };
  public int[] colInd1 = new int[] { 0, 2, 0, 1, 1, 3, 2, 2, 3 };
  public double[] val1 = new double[] { 1, 5, 4, 2, 3, 9, 7, 8, 6 };

  public double[] weights = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

  /**
   * Prepare test stack.
   * 
   * <p>Pixels are consecutive numbers from 0 column ordered. Column order is because of IP that
   * stores column in first dimension of 2D array.
   * 0 3 6 9
   * 1 4 7 10
   * 2 5 8 11
   * 2nd slice
   * 12 15 18 21
   * 13 16 19 22
   * 14 17 20 23
   * 
   * <p>There is 46 edges and 24 vertices in graph.
   * 
   * @param width number of columns
   * @param height number of rows
   * @param nz number of slices
   * @param type can be 'int' or 'double'
   * @return Stack of images.
   */
  public static ImageStack getTestStack(int width, int height, int nz, String type) {
    double l = 0;
    // IP is column ordered, array[0][] i column in 2D
    ImageStack stack = new ImageStack(width, height);
    ImageProcessor ip;
    for (int z = 0; z < nz; z++) {
      switch (type) {
        case "int":
          ip = new ShortProcessor(width, height);
          break;
        case "double":
          ip = new FloatProcessor(width, height);
          break;
        default:
          throw new IllegalArgumentException("Wrong type");
      }
      for (int c = 0; c < width; c++) {
        for (int r = 0; r < height; r++) {
          ip.putPixelValue(c, r, l);
          l++;
        }
      }
      stack.addSlice(ip);
    }
    return stack;
  }

}
