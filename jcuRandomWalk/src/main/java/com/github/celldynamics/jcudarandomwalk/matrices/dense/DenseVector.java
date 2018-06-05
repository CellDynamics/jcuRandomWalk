package com.github.celldynamics.jcudarandomwalk.matrices.dense;

import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;

import org.apache.commons.lang3.NotImplementedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;

/**
 * Dense vecor either on GPU or CPU.
 * 
 * @author baniu
 *
 */
public abstract class DenseVector implements IDenseVector, Serializable {

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

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getVal()
   */
  @Override
  public float[] getVal() {
    if (val == null && this instanceof DenseVectorDevice) {
      ((DenseVectorDevice) this).retrieveFromDevice();
    }
    return val;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getRowNumber()
   */
  @Override
  public int getRowNumber() {
    return rowNumber;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getColNumber()
   */
  @Override
  public int getColNumber() {
    return colNumber;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getElementNumber()
   */
  @Override
  public int getElementNumber() {
    return val.length;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#removeRows(int[])
   */
  @Override
  public IMatrix removeRows(int[] rows) {
    throw new NotImplementedException("Not implemented");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#removeCols(int[])
   */
  @Override
  public IMatrix removeCols(int[] cols) {
    throw new NotImplementedException("Not implemented");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#multiply(com.github.celldynamics.
   * jcudarandomwalk.matrices.IMatrix)
   */
  @Override
  public IMatrix multiply(IMatrix in) {
    throw new NotImplementedException("Not implemented");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#transpose()
   */
  @Override
  public IMatrix transpose() {
    int tmp = colNumber;
    colNumber = rowNumber;
    rowNumber = tmp;
    return this;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#sumAlongRows()
   */
  @Override
  public IMatrix sumAlongRows() {
    throw new NotImplementedException("Not implemented");
  }

  /**
   * Produce sparse matrix of the same type as specified from raw data.
   * 
   * @param lap type of output matrix, can be even empty dummy object
   * @param rows number of rows, one of dimensions must be 1
   * @param cols number of columns, one of dimensions must be 1
   * @param val values
   * @return matrix of type 'type' from above arrays
   */
  public static IDenseVector denseVectorFactory(IMatrix lap, int rows, int cols, float[] val) {
    if (val.length == 0) {
      throw new IllegalArgumentException(
              "One or more arrays passed to sparseMatrixFactory are 0-sized");
    }
    Class<? extends IMatrix> classToLoad = lap.getClass();
    Class<?>[] carg = new Class[3]; // Our constructor has 3 arguments
    carg[0] = int.class;
    carg[1] = int.class;
    carg[2] = float[].class;
    try {
      return (IDenseVector) classToLoad.getDeclaredConstructor(carg).newInstance(rows, cols, val);
    } catch (InstantiationException | IllegalAccessException | IllegalArgumentException
            | InvocationTargetException | NoSuchMethodException | SecurityException e) {
      throw new IllegalArgumentException("Can not create object instance: " + e.getMessage());
    }
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

  /**
   * Default constructor for building dense vector from list of values. Must be implemented in
   * concrete classes.
   * 
   * @param rows number of rows, rows or cols should be 1
   * @param cols number of columns, rows or cols should be 1
   * @param val values
   */
  public DenseVector(int rows, int cols, double[] val) {
    throw new NotImplementedException("This constructor must be implemented in concrete classes.");
  }

  /**
   * Default constructor. Generally not used.
   */
  public DenseVector() {
  }

}
