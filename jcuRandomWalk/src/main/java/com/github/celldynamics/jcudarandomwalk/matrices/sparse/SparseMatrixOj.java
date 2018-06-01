package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.ojalgo.access.ElementView2D;
import org.ojalgo.matrix.store.SparseStore;

import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.dense.IDenseVector;

/**
 * @author p.baniukiewicz
 *
 */
public class SparseMatrixOj implements ISparseMatrix {

  public final static OjSparseMatrixFactory FACTORY = new OjSparseMatrixFactory();

  SparseStore<Double> mat;

  SparseMatrixOj(SparseStore<Double> mat) {
    this.mat = mat;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getVal()
   */
  @Override
  public float[] getVal() {
    ElementView2D<Double, ?> nz = mat.nonzeros();
    List<Float> ret = new ArrayList<>();
    while (nz.hasNext()) {
      Number e = nz.next().get();
      ret.add(e.floatValue());
    }
    return ArrayUtils.toPrimitive(ret.toArray(new Float[0]));
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getRowNumber()
   */
  @Override
  public int getRowNumber() {
    return (int) mat.countRows();
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getColNumber()
   */
  @Override
  public int getColNumber() {
    return (int) mat.countColumns();
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getElementNumber()
   */
  @Override
  public int getElementNumber() {
    ElementView2D<Double, ?> nz = mat.nonzeros();
    return (int) nz.getExactSizeIfKnown();
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#removeRows(int[])
   */
  @Override
  public IMatrix removeRows(int[] rows) {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#removeCols(int[])
   */
  @Override
  public IMatrix removeCols(int[] cols) {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#multiply(com.github.celldynamics.
   * jcudarandomwalk.matrices.IMatrix)
   */
  @Override
  public IMatrix multiply(IMatrix in) {
    IMatrix ret;
    if (in instanceof SparseMatrixOj) {
      SparseStore<Double> tmpret = (SparseStore<Double>) mat.multiply(((SparseMatrixOj) in).mat);
      ret = new SparseMatrixOj(tmpret);
    } else {
      throw new IllegalArgumentException("This combination is not allowed");
    }
    return ret;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#transpose()
   */
  @Override
  public IMatrix transpose() {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#toGpu()
   */
  @Override
  public IMatrix toGpu() {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#toCpu()
   */
  @Override
  public IMatrix toCpu() {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#free()
   */
  @Override
  public void free() {
    // TODO Auto-generated method stub

  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#sumAlongRows()
   */
  @Override
  public IMatrix sumAlongRows() {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#getRowInd()
   */
  @Override
  public int[] getRowInd() {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#getColInd()
   */
  @Override
  public int[] getColInd() {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * 
   * @see
   * com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#getSparseMatrixType()
   */
  @Override
  public SparseMatrixType getSparseMatrixType() {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#convert2csr()
   */
  @Override
  public ISparseMatrix convert2csr() {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#convert2coo()
   */
  @Override
  public ISparseMatrix convert2coo() {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#full()
   */
  @Override
  public double[][] full() {
    // TODO Auto-generated method stub
    return null;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#luSolve(com.github.
   * celldynamics.jcudarandomwalk.matrices.dense.IDenseVector, boolean, int, float)
   */
  @Override
  public float[] luSolve(IDenseVector b_gpuPtr, boolean iLuBiCGStabSolve, int iter, float tol) {
    // TODO Auto-generated method stub
    return null;
  }

}
