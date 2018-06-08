package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import static org.ojalgo.function.aggregator.Aggregator.SUM;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import org.apache.commons.lang3.ArrayUtils;
import org.ojalgo.RecoverableCondition;
import org.ojalgo.access.ElementView1D;
import org.ojalgo.access.ElementView2D;
import org.ojalgo.matrix.decomposition.LU;
import org.ojalgo.matrix.store.ElementsSupplier;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;
import org.ojalgo.matrix.store.SparseStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVectorOj;

/**
 * Implementation of {@link ISparseMatrix} by OjAlg.
 * 
 * @author p.baniukiewicz
 *
 */
public class SparseMatrixOj {

  /**
   * The Constant LOGGER.
   */
  static final Logger LOGGER = LoggerFactory.getLogger(SparseMatrixOj.class.getName());

  protected int rowNumber; // number of rows
  protected int colNumber; // number of cols

  /**
   * OjAlg store wrapped by this class.
   */
  public MatrixStore<Double> mat;

  /**
   * Default empty constructor.
   */
  public SparseMatrixOj() {
  }

  /**
   * Wrap existing Oj object.
   * 
   * @param mat oj object.
   */
  public SparseMatrixOj(MatrixStore<Double> mat) {
    this.mat = mat;
    this.rowNumber = (int) mat.countRows();
    this.colNumber = (int) mat.countColumns();
  }

  /**
   * Wrap existing Oj object.
   * 
   * @param mat oj object.
   * @param rowNumber
   * @param colNumber
   */
  public SparseMatrixOj(MatrixStore<Double> mat, int rowNumber, int colNumber) {
    this.mat = mat;
    this.rowNumber = rowNumber;
    this.colNumber = colNumber;
  }

  /**
   * Create sparse matrix. Oj wrapper.
   * 
   * @param rowInd row indices
   * @param colInd column indices
   * @param val values
   * @param rowNumber number of rows
   * @param colNumber number of columns
   * @param matrixInputFormat for compatibility
   */
  public SparseMatrixOj(int[] rowInd, int[] colInd, float[] val, int rowNumber, int colNumber,
          SparseMatrixType matrixInputFormat) {
    if ((rowInd.length != colInd.length) || (rowInd.length != val.length)) {
      throw new IllegalArgumentException("Input arrays should have the same length in COO format");
    }
    this.rowNumber = rowNumber;
    this.colNumber = colNumber;
    SparseStore<Double> mtrxA = SparseStore.PRIMITIVE.make(rowNumber, colNumber);
    for (int i = 0; i < rowInd.length; i++) {
      mtrxA.set(rowInd[i], colInd[i], val[i]);
    }
    mat = mtrxA;
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
    this(rowInd, colInd, val, rowNumber, colNumber, SparseMatrixType.MATRIX_FORMAT_COO);
  }

  /**
   * Create sparse matrix. Oj wrapper. Calculate number of rows and colums.
   * 
   * @param rowInd row indices
   * @param colInd column indices
   * @param val values
   * @param matrixInputFormat for compatibility
   */
  public SparseMatrixOj(int[] rowInd, int[] colInd, float[] val,
          SparseMatrixType matrixInputFormat) {
    if ((rowInd.length != colInd.length) || (rowInd.length != val.length)) {
      throw new IllegalArgumentException("Input arrays should have the same length in COO format");
    }
    colNumber = IntStream.of(colInd).parallel().max().getAsInt() + 1; // assuming 0 based
    rowNumber = IntStream.of(rowInd).parallel().max().getAsInt() + 1;
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
    this(rowInd, colInd, val, SparseMatrixType.MATRIX_FORMAT_COO);
  }

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

  public int getRowNumber() {
    return rowNumber;
  }

  public int getColNumber() {
    return colNumber;
  }

  public int getElementNumber() {
    ElementView1D<Double, ?> nz = mat.nonzeros();
    return (int) nz.getExactSizeIfKnown();
  }

  public SparseMatrixOj removeRows(int[] rows) {
    int[] rr = getRowInd();
    int[] cc = getColInd();
    float[] vv = getVal();
    SparseCoordinates tmp = new SparseCoordinates(rr, cc, vv, getRowNumber(), getColNumber(),
            getSparseMatrixType());
    tmp.removeRowsIndices(rows);

    return new SparseMatrixOj(tmp.getRowInd(), tmp.getColInd(), tmp.getVal(), tmp.getRowNumber(),
            tmp.getColNumber());
  }

  public SparseMatrixOj removeCols(int[] cols) {
    int[] rr = getRowInd();
    int[] cc = getColInd();
    float[] vv = getVal();
    SparseCoordinates tmp = new SparseCoordinates(rr, cc, vv, getRowNumber(), getColNumber(),
            getSparseMatrixType());
    tmp.removeColsIndices(cols);

    return new SparseMatrixOj(tmp.getRowInd(), tmp.getColInd(), tmp.getVal(), tmp.getRowNumber(),
            tmp.getColNumber());
  }

  public SparseMatrixOj multiply(SparseMatrixOj weight) {
    SparseMatrixOj ret;
    if (weight instanceof SparseMatrixOj) {
      SparseStore<Double> tmpret =
              (SparseStore<Double>) mat.multiply(((SparseMatrixOj) weight).mat);
      ret = new SparseMatrixOj(tmpret);
    } else {
      throw new IllegalArgumentException("This combination is not allowed");
    }
    return ret;
  }

  public SparseMatrixOj transpose() {
    SparseStore<Double> retm = SparseStore.PRIMITIVE.make(getColNumber(), getRowNumber());
    mat.transpose().supplyTo(retm);
    SparseMatrixOj ret = new SparseMatrixOj(retm);
    return ret;
  }

  public DenseVectorOj sumAlongRows() {
    ElementsSupplier<Double> rs = mat.reduceRows(SUM);
    PrimitiveDenseStore ret = PrimitiveDenseStore.FACTORY.makeZero(getRowNumber(), 1);
    rs.supplyTo(ret);
    return new DenseVectorOj(ret);

  }

  public int[] getRowInd() {
    ElementView2D<Double, ?> nz = ((SparseStore<Double>) mat).nonzeros();
    List<Integer> ret = new ArrayList<>();
    while (nz.hasNext()) {
      long i = nz.next().row();
      ret.add((int) i);
    }
    return ArrayUtils.toPrimitive(ret.toArray(new Integer[0]));
  }

  public int[] getColInd() {
    ElementView2D<Double, ?> nz = ((SparseStore<Double>) mat).nonzeros();
    List<Integer> ret = new ArrayList<>();
    while (nz.hasNext()) {
      long i = nz.next().column();
      ret.add((int) i);
    }
    return ArrayUtils.toPrimitive(ret.toArray(new Integer[0]));
  }

  public SparseMatrixType getSparseMatrixType() {
    return SparseMatrixType.MATRIX_FORMAT_COO;
  }

  public double[][] full() {
    return this.mat.toRawCopy2D();
  }

  public float[] luSolve(DenseVectorOj b_gpuPtr, boolean iLuBiCGStabSolve, int iter, float tol) {
    LOGGER.info("Parameters iter and tol are ignored for OjAlg");
    final LU<Double> tmpA = LU.PRIMITIVE.make();
    tmpA.decompose(this.mat);
    try {
      // we need to *-1 here from unknown result. This breaks tests so they are adopted as well.
      MatrixStore<Double> ret = tmpA.solve(this.mat, b_gpuPtr.mat).multiply(-1.0);
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
    return "SparseMatrixOj [nrows=" + rowNumber + ", ncols=" + colNumber + ", mat=" + mat + "]";
  }

  public static SparseMatrixOj factory(SparseCoordinates raw) {
    return new SparseMatrixOj(raw.getRowInd(), raw.getColInd(), raw.getVal(), raw.getRowNumber(),
            raw.getColNumber(), SparseMatrixType.MATRIX_FORMAT_COO);
  }

}
