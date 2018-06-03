package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.NotImplementedException;
import org.ojalgo.access.ElementView1D;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.SparseStore;

import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.dense.IDenseVector;

/**
 * @author p.baniukiewicz
 *
 */
public class SparseMatrixOj implements ISparseMatrix {

  public final static OjSparseMatrixFactory FACTORY = new OjSparseMatrixFactory();

  private int nrows;
  private int ncols;
  MatrixStore<Double> mat;

  SparseMatrixOj(MatrixStore<Double> mat) {
    this.mat = mat;
    this.nrows = (int) mat.countRows();
    this.ncols = (int) mat.countColumns();
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getVal()
   */
  @Override
  public float[] getVal() {
    // this returns column ordered nonzero indices. Matrix is ok, but only getVal returns in
    // different order.
    ElementView1D<Double, ?> nz = mat.nonzeros();
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
    return nrows;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getColNumber()
   */
  @Override
  public int getColNumber() {
    return ncols;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#getElementNumber()
   */
  @Override
  public int getElementNumber() {
    ElementView1D<Double, ?> nz = mat.nonzeros();
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
    SparseStore<Double> retm = SparseStore.PRIMITIVE.make(getColNumber(), getRowNumber());
    mat.transpose().supplyTo(retm);
    SparseMatrixOj ret = new SparseMatrixOj(retm);
    return ret;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#toGpu()
   */
  @Override
  public IMatrix toGpu() {
    return this;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#toCpu()
   */
  @Override
  public IMatrix toCpu() {
    return this;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#free()
   */
  @Override
  public void free() {
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
    ElementView1D<Double, ?> nz = mat.nonzeros();
    List<Integer> ret = new ArrayList<>();
    while (nz.hasNext()) {
      long i = nz.next().index();
      Number e = nz.get();
      ret.add((int) (i % nrows));
    }
    return ArrayUtils.toPrimitive(ret.toArray(new Integer[0]));
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#getColInd()
   */
  @Override
  public int[] getColInd() {
    ElementView1D<Double, ?> nz = mat.nonzeros();
    List<Integer> ret = new ArrayList<>();
    while (nz.hasNext()) {
      long i = nz.next().index();
      ret.add((int) (i % ncols));
    }
    return ArrayUtils.toPrimitive(ret.toArray(new Integer[0]));
  }

  /*
   * (non-Javadoc)
   * 
   * @see
   * com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#getSparseMatrixType()
   */
  @Override
  public SparseMatrixType getSparseMatrixType() {
    return SparseMatrixType.MATRIX_FORMAT_COO;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#convert2csr()
   */
  @Override
  public ISparseMatrix convert2csr() {
    throw new NotImplementedException("Conversions are not implemented for OjAlg");
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#convert2coo()
   */
  @Override
  public ISparseMatrix convert2coo() {
    throw new NotImplementedException("Conversions are not implemented for OjAlg");
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
