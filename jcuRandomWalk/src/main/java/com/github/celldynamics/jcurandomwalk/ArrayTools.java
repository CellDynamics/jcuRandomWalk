package com.github.celldynamics.jcurandomwalk;

import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;

import org.apache.commons.lang3.ArrayUtils;

import com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVectorDevice;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;

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

  /**
   * Save all data from input matrix in specified folder
   * 
   * @param matrix matrix to save
   * @param folder folder (not file name!)
   */
  public static void dump(DenseVectorDevice matrix, Path folder) {
    matrix.toCpu(true);
    try {
      PrintWriter bout = new PrintWriter(
              new BufferedWriter(new FileWriter(folder.resolve("b.txt").toString())));
      for (int x = 0; x < matrix.getVal().length; x++) {
        bout.println(matrix.getVal()[x]);
      }
      bout.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  /**
   * Save all data from input matrix in specified folder
   * 
   * @param matrix matrix to save
   * @param folder folder (not file name!)
   */
  public static void dump(SparseMatrixDevice matrix, Path folder) {
    matrix.toCpu(true);
    try {
      PrintWriter rowIndout = new PrintWriter(
              new BufferedWriter(new FileWriter(folder.resolve("rowInd.txt").toString())));
      for (int x = 0; x < matrix.getRowInd().length; x++) {
        rowIndout.println(matrix.getRowInd()[x]);
      }
      rowIndout.close();
      PrintWriter colIndout = new PrintWriter(
              new BufferedWriter(new FileWriter(folder.resolve("colInd.txt").toString())));
      for (int x = 0; x < matrix.getColInd().length; x++) {
        colIndout.println(matrix.getColInd()[x]);
      }
      colIndout.close();
      PrintWriter valout = new PrintWriter(
              new BufferedWriter(new FileWriter(folder.resolve("val.txt").toString())));
      for (int x = 0; x < matrix.getVal().length; x++) {
        valout.println(matrix.getVal()[x]);
      }
      valout.close();

      // to coo
      SparseMatrixDevice coo = matrix.convert2coo();
      coo.toCpu(true);
      rowIndout = new PrintWriter(
              new BufferedWriter(new FileWriter(folder.resolve("rowIndCOO.txt").toString())));
      for (int x = 0; x < coo.getRowInd().length; x++) {
        rowIndout.println(coo.getRowInd()[x]);
      }
      rowIndout.close();
      colIndout = new PrintWriter(
              new BufferedWriter(new FileWriter(folder.resolve("colIndCOO.txt").toString())));
      for (int x = 0; x < coo.getColInd().length; x++) {
        colIndout.println(coo.getColInd()[x]);
      }
      colIndout.close();
      valout = new PrintWriter(
              new BufferedWriter(new FileWriter(folder.resolve("valCOO.txt").toString())));
      for (int x = 0; x < coo.getVal().length; x++) {
        valout.println(coo.getVal()[x]);
      }
      valout.close();

      throw new IllegalArgumentException("STOP");
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  /**
   * Allocate memory on GPU and copy array if it is not null.
   * 
   * @param array array to copy to GPU or null
   * @param size size of array
   * @return pointer to allocated memory
   */
  public static Pointer cudaMallocCopy(float[] array, int size) {
    Pointer tmp = new Pointer();
    JCuda.cudaMalloc(tmp, Sizeof.FLOAT * size);
    if (array != null && array.length > 0) {
      JCuda.cudaMemcpy(tmp, Pointer.to(array), Sizeof.FLOAT * size, cudaMemcpyHostToDevice);
    }
    return tmp;
  }

  /**
   * Allocate memory on GPU and copy array if it is not null.
   * 
   * @param array array to copy to GPU or null
   * @param size size of array
   * @return pointer to allocated memory
   */
  public static Pointer cudaMallocCopy(int[] array, int size) {
    Pointer tmp = new Pointer();
    JCuda.cudaMalloc(tmp, Sizeof.INT * size);
    if (array != null && array.length > 0) {
      JCuda.cudaMemcpy(tmp, Pointer.to(array), Sizeof.INT * size, cudaMemcpyHostToDevice);
    }
    return tmp;
  }
}
