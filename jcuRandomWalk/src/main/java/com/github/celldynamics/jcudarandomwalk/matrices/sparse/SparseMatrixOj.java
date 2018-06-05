package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import static org.ojalgo.function.aggregator.Aggregator.SUM;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.NotImplementedException;
import org.ojalgo.RecoverableCondition;
import org.ojalgo.access.ElementView1D;
import org.ojalgo.matrix.decomposition.LU;
import org.ojalgo.matrix.store.ElementsSupplier;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;
import org.ojalgo.matrix.store.SparseStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVectorOj;
import com.github.celldynamics.jcudarandomwalk.matrices.dense.IDenseVector;

/**
 * Implementation of {@link ISparseMatrix} by OjAlg.
 * 
 * @author p.baniukiewicz
 *
 */
public class SparseMatrixOj implements ISparseMatrix {

  /**
   * The Constant LOGGER.
   */
  static final Logger LOGGER = LoggerFactory.getLogger(SparseMatrixOj.class.getName());

  private int nrows;
  private int ncols;
  /**
   * OjAlg store wrapped by this class.
   */
  MatrixStore<Double> mat;

  /**
   * Wrap existing Oj object.
   * 
   * @param mat oj object.
   */
  public SparseMatrixOj(MatrixStore<Double> mat) {
    this.mat = mat;
    this.nrows = (int) mat.countRows();
    this.ncols = (int) mat.countColumns();
  }

  /**
   * Create sparse matrix. Oj wrapper.
   * 
   * @param rowInd row indices
   * @param colInd column indices
   * @param val values
   * @param rowNumber number of rows
   * @param colNumber number of columns
   */
  public SparseMatrixOj(int[] rowInd, int[] colInd, float[] val, int rowNumber, int colNumber) {
    if ((rowInd.length != colInd.length) || (rowInd.length != val.length)) {
      throw new IllegalArgumentException("Input arrays should have the same length in COO format");
    }
    SparseStore<Double> mtrxA = SparseStore.PRIMITIVE.make(rowNumber, colNumber);
    for (int i = 0; i < rowInd.length; i++) {
      mtrxA.set(rowInd[i], colInd[i], val[i]);
    }
    mat = mtrxA;
  }

  /**
   * Create sparse matrix. Oj wrapper. Calculate number of rows and colums.
   * 
   * @param rowInd row indices
   * @param colInd column indices
   * @param val values
   */
  public SparseMatrixOj(int[] rowInd, int[] colInd, float[] val) {
    if ((rowInd.length != colInd.length) || (rowInd.length != val.length)) {
      throw new IllegalArgumentException("Input arrays should have the same length in COO format");
    }
    int colNumber = IntStream.of(colInd).parallel().max().getAsInt() + 1; // assuming 0 based
    int rowNumber = IntStream.of(rowInd).parallel().max().getAsInt() + 1;
    SparseStore<Double> mtrxA = SparseStore.PRIMITIVE.make(rowNumber, colNumber);
    for (int i = 0; i < rowInd.length; i++) {
      mtrxA.set(rowInd[i], colInd[i], val[i]);
    }
    mat = mtrxA;
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
    int[] rr = getRowInd();
    int[] cc = getColInd();
    float[] vv = getVal();
    SparseMatrixHost tmp =
            (SparseMatrixHost) new SparseMatrixHost(rr, cc, vv, getSparseMatrixType())
                    .removeRows(rows);

    return new SparseMatrixOj(tmp.getRowInd(), tmp.getColInd(), tmp.getVal());
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.IMatrix#removeCols(int[])
   */
  @Override
  public IMatrix removeCols(int[] cols) {
    int[] rr = getRowInd();
    int[] cc = getColInd();
    float[] vv = getVal();
    SparseMatrixHost tmp =
            (SparseMatrixHost) new SparseMatrixHost(rr, cc, vv, getSparseMatrixType())
                    .removeCols(cols);

    return new SparseMatrixOj(tmp.getRowInd(), tmp.getColInd(), tmp.getVal());
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
    ElementsSupplier<Double> rs = mat.reduceRows(SUM);
    PrimitiveDenseStore ret = PrimitiveDenseStore.FACTORY.makeZero(getRowNumber(), 1);
    rs.supplyTo(ret);
    return new DenseVectorOj(ret);

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
      ret.add((int) (i / nrows));
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
    return this.mat.toRawCopy2D();
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix#luSolve(com.github.
   * celldynamics.jcudarandomwalk.matrices.dense.IDenseVector, boolean, int, float)
   */
  @Override
  public float[] luSolve(IDenseVector b_gpuPtr, boolean iLuBiCGStabSolve, int iter, float tol) {
    LOGGER.info("Parameters iter and tol are ignored for OjAlg");
    final LU<Double> tmpA = LU.PRIMITIVE.make();
    tmpA.decompose(this.mat);
    try {
      MatrixStore<Double> ret = tmpA.solve(this.mat, ((DenseVectorOj) b_gpuPtr).mat);
      return new SparseMatrixOj(ret).getVal();
    } catch (RecoverableCondition e) {
      LOGGER.error("Solver failed with error: " + e.toString());
      throw new IllegalArgumentException(e);
    }
  }

  /*
   * (non-Javadoc)
   * 
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString() {
    return "SparseMatrixOj [nrows=" + nrows + ", ncols=" + ncols + ", mat=" + mat + "]";
  }

}
