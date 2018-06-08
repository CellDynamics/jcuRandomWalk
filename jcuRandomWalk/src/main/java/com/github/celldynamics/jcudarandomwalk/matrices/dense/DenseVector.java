package com.github.celldynamics.jcudarandomwalk.matrices.dense;

import java.io.Serializable;
import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Dense vecor either on GPU or CPU.
 * 
 * @author baniu
 *
 */
public abstract class DenseVector implements Serializable {

  /**
   * Default UID.
   */
  private static final long serialVersionUID = 371179784126818370L;

  /**
   * The Constant LOGGER.
   */
  static final Logger LOGGER = LoggerFactory.getLogger(DenseVector.class.getName());

  protected float[] val; // value
  protected int rowNumber; // number of rows
  protected int colNumber; // number of cols

  /**
   * Utility constructor.
   * 
   * @param rows number of rows, rows or cols should be 1
   * @param cols number of columns, rows or cols should be 1
   */
  DenseVector(int rows, int cols) {
    if (rows > 1 && cols > 1) {
      throw new IllegalArgumentException("For vector one of dimensions must be 1.");
    }
    this.rowNumber = rows;
    this.colNumber = cols;
  }

  public float[] getVal() {
    return val;
  }

  public int getRowNumber() {
    return rowNumber;
  }

  public int getColNumber() {
    return colNumber;
  }

  public int getElementNumber() {
    return val.length;
  }

  /*
   * (non-Javadoc)
   * 
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString() {
    return "DenseVector [rowNumber=" + rowNumber + ", colNumber=" + colNumber + ", val="
            + Arrays.toString(val) + "]";
  }

}
