package com.github.celldynamics.jcudarandomwalk.matrices.dense;

/**
 * Dense 1D vector on Host.
 * 
 * @author baniu
 *
 */
public class DenseVectorHost extends DenseVector {

  /**
   * Default UID.
   */
  private static final long serialVersionUID = 5888367409910667331L;

  /**
   * Empty vector.
   */
  public DenseVectorHost() {
    super(0, 0);
    val = new float[0];
  }

  /**
   * Default constructor for building dense vector from list of values. Must be implemented in
   * concrete classes.
   * 
   * @param rows number of rows, rows or cols should be 1
   * @param cols number of columns, rows or cols should be 1
   * @param val values
   */
  public DenseVectorHost(int rows, int cols, float[] val) {
    super(rows, cols);
    this.val = val;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#free()
   */
  @Override
  public void free() {
  }

}
