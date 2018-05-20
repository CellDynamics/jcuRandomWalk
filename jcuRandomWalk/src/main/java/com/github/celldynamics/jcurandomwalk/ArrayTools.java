package com.github.celldynamics.jcurandomwalk;

import org.apache.commons.lang3.ArrayUtils;

/**
 * Tools for converting arrays.
 * 
 * @author baniu
 *
 */
public class ArrayTools {

  /**
   * Convert 2d array of integers to Number.
   * 
   * @param array 2D array to convert
   * @return 2D array of Number
   */
  public static Number[][] array2Object(int[][] array) {
    Number[][] ret = new Number[array.length][];
    for (int i = 0; i < array.length; i++) {
      ret[i] = ArrayUtils.toObject(array[i]);
    }
    return ret;
  }

  /**
   * Convert 2d array of doubles to Number.
   * 
   * @param array 2D array to convert
   * @return 2D array of Number
   */
  public static Number[][] array2Object(double[][] array) {
    Number[][] ret = new Number[array.length][];
    for (int i = 0; i < array.length; i++) {
      ret[i] = ArrayUtils.toObject(array[i]);
    }
    return ret;
  }

  /**
   * Convert 2d array of floats to Number.
   * 
   * @param array 2D array to convert
   * @return 2D array of Number
   */
  public static Number[][] array2Object(float[][] array) {
    Number[][] ret = new Number[array.length][];
    for (int i = 0; i < array.length; i++) {
      ret[i] = ArrayUtils.toObject(array[i]);
    }
    return ret;
  }

  /**
   * Print small arrays.
   * 
   * @param array array to print
   * @return String representation of array
   */
  public static String printArray(Number[][] array) {
    String ret = "\n";
    int ncols = array.length;
    int nrows = array[0].length;
    for (int r = 0; r < nrows; r++) {
      for (int c = 0; c < ncols; c++) {
        ret += array[c][r].toString() + '\t';
      }
      ret += '\n';
    }
    return ret;
  }
}
