package com.github.celldynamics.jcudarandomwalk.matrices;

import java.util.stream.IntStream;

/**
 * Structure for holding coordinates for sparse matrices.
 * 
 * <p>The coordinates are stored in three separate vectors for x, y and value. Coordinates are
 * 0-based.
 * 
 * @author baniu
 *
 */
class SparseMatrix {
  private int size; // number of coordinates
  private int i = 0; // counter

  /**
   * Get i-coordinates vector. Rows.
   * 
   * @return the vi
   */
  public int[] getCrows() {
    return vr;
  }

  /**
   * Get j-coordinates vector. Columns.
   * 
   * @return the vj
   */
  public int[] getCcols() {
    return vc;
  }

  /**
   * Get values vector.
   * 
   * @return the val
   */
  public double[] getCval() {
    return val;
  }

  private int[] vr; // rows
  private int[] vc; // cols
  private double[] val; // value

  /**
   * Create storage for specified number of sparse elements.
   * 
   * @param size size of the storage
   */
  public SparseMatrix(int size) {
    this.size = size;
    vr = new int[size];
    vc = new int[size];
    val = new double[size];
  }

  /**
   * Add entry (coordinates and value) to store.
   * 
   * @param r row coordinate
   * @param c column coordinate
   * @param v value stored under [r,c]
   */
  public void add(int r, int c, double v) {
    vr[i] = r;
    vc[i] = c;
    val[i] = v;
    i++;
  }

  /**
   * Convert sparse coordinates to full matrix.
   * 
   * <p>Only small matrixes. Column ordered.
   * 
   * @return full 2D matrix [cols][rows]
   */
  public double[][] full() {
    int ncols = IntStream.of(getCcols()).max().getAsInt() + 1; // assuming 0 based
    int nrows = IntStream.of(getCrows()).max().getAsInt() + 1;
    double[][] ret = new double[ncols][];
    for (int c = 0; c < ncols; c++) {
      ret[c] = new double[nrows];
    }
    for (int l = 0; l < size; l++) {
      ret[getCcols()[l]][getCrows()[l]] = getCval()[l];
    }
    return ret;

  }

}