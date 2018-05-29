package com.github.celldynamics.jcurandomwalk;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

// TODO: Auto-generated Javadoc
/**
 * The Class TestDataGenerators.
 *
 * @author baniu
 */
public class TestDataGenerators {

  /** The row ind. */
  public int[] rowInd = new int[] { 0, 0, 1, 1, 2, 2, 2, 3, 3 };

  /** The col ind. */
  public int[] colInd = new int[] { 0, 1, 1, 2, 0, 3, 4, 2, 4 };

  /** The val. */
  public float[] val = new float[] { 1, 4, 2, 3, 5, 7, 8, 9, 6 };

  /** The row ind 1. */
  public int[] rowInd1 = new int[] { 0, 0, 1, 1, 2, 2, 3, 4, 4 };

  /** The col ind 1. */
  public int[] colInd1 = new int[] { 0, 2, 0, 1, 1, 3, 2, 2, 3 };

  /** The val 1. */
  public float[] val1 = new float[] { 1, 5, 4, 2, 3, 9, 7, 8, 6 };

  /** The weights. */
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

  /**
   * Load prepared stack of size 5x4x3
   * 
   * <p>Slice 1
   * 0.500 0.800 0.300 0.900 0.200
   * 0.400 0.500 0.300 0.800 0.900
   * 0.100 0.200 0.300 0.400 0.500
   * 0.900 0.700 0.500 0.300 0.200
   * 
   * Slice 2
   * 0.510 0.900 0.300 0.800 0.500
   * 0.900 0.800 0.300 0.500 0.400
   * 0.500 0.400 0.300 0.200 0.100
   * 0.200 0.300 0.500 0.700 0.900
   * 
   * Slice 3
   * 0.900 0.700 0.500 0.300 0.200
   * 0.100 0.200 0.300 0.400 0.500
   * 0.400 0.500 0.300 0.800 0.900
   * 0.500 0.800 0.300 0.900 0.200
   * 
   * @return stack
   */
  public static ImageStack getTestStack_1() {
    ImagePlus ret = IJ.openImage("src/test/test_data/TestStack.tif");
    return ret.getImageStack();
  }

  /**
   * Load prepared seed stack 5x4x3
   * 
   * Non zero pixels at positions (x,y column-wise index):
   * 0,0,0, 0
   * 0,2,0, 2
   * 3,1,0, 13
   * 
   * 2,2,1, 30
   * 
   * 1,3,2, 47
   * 2,1,2, 49
   * 4,3,2, 59.
   *
   * @return stack
   */
  public static ImageStack getSeedStack_1() {
    ImagePlus ret = IJ.openImage("src/test/test_data/seeds.tif");
    return ret.getImageStack();
  }

}
