package com.github.celldynamics.jcurandomwalk;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.time.StopWatch;

import com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVectorOj;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseCoordinates;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj;

import ij.ImageStack;

/**
 * Main routine for RW segmentation.
 * 
 * @author p.baniukiewicz
 * @author t.bretschneider
 *
 */
public class RandomWalkAlgorithmOj extends RandomWalkSolver {

  SparseMatrixOj reducedLap; // reduced laplacian
  SparseMatrixOj lap; // full laplacian
  List<DenseVectorOj> bvector = new ArrayList<>(); // right vector

  private List<float[]> rawSoultions = new ArrayList<>(2); // in case getRawProbs is called
  private int[] mergedseeds; // Optimisation store

  /**
   * Constructor for tests.
   */
  RandomWalkAlgorithmOj() {

  }

  /**
   * Default constructor.
   * 
   * @param stack stack to be segmented
   * @param options options
   */
  public RandomWalkAlgorithmOj(ImageStack stack, RandomWalkOptions options) {
    super(stack, options);
  }

  /**
   * Compute Laplacian A'XA.
   * 
   * @see #getLap()
   */
  void computeLaplacian() {
    SparseMatrixOj incidence;
    SparseMatrixOj incidenceT;
    SparseMatrixOj weight;
    LOGGER.info("Computing Laplacian");
    StopWatch timer = new StopWatch();
    timer.start();
    // IMatrix atw = null;
    SparseCoordinates incTmp = img.getIncidence();
    SparseCoordinates weiTmp = img.getWeights();

    incidence = SparseMatrixOj.factory(incTmp);
    incidenceT = incidence.transpose();
    weight = SparseMatrixOj.factory(weiTmp);

    // ElementsSupplier<Double> aTw = ((SparseMatrixOj) weight).mat
    // .premultiply(((SparseMatrixOj) incidence).mat.transpose());
    // SparseStore<Double> mm =
    // SparseStore.PRIMITIVE.make(incidence.getColNumber(), incidence.getRowNumber());
    // aTw.supplyTo(mm);
    // atw = new SparseMatrixOj(mm, incidence.getColNumber(), incidence.getRowNumber());

    LOGGER.info("Base class: " + incidence.getClass().getSimpleName());
    // A'*W*A
    // ISparseMatrix ATW = incidenceGpuT.multiply(wGpu);
    // ISparseMatrix ATWA = ATW.multiply(incidenceGpu);
    SparseMatrixOj atw = incidenceT.multiply(weight);// .multiply(incidenceGpu);
    this.lap = atw.multiply(incidence);
    timer.stop();
    LOGGER.info("Laplacian computed in " + timer.toString());
  }

  /**
   * This method remove rows and columns defined in source and sink arrays from Laplacian lap and
   * computed B vector.
   * 
   * <p>Laplacian must be square. Indexes from source and sink are merged first to deal with
   * duplicates. There is no special checking implemented so largest index from sink and source
   * should be < length(lap)
   * 
   * <p>Source and sink can be swapped to get background oriented segmentation. B vector is computed
   * always from source.
   * 
   * @param source list of indices to remove - sources. Array must be sorted
   * @param sink second list of indices to remove, merged with the first one. Array must be sorted
   * @see #getReducedLap()
   */
  void computeReducedLaplacian(Integer[] source, Integer[] sink) {
    if (lap.getColNumber() != lap.getRowNumber()) {
      throw new IllegalArgumentException("Matrix should be square");
    }
    LOGGER.info("Reducing Laplacian");
    StopWatch timer = new StopWatch();
    timer.start();
    // ISparseMatrix lapCoo = lap.convert2coo();
    // ISparseMatrix cpuL;
    // cpuL = (ISparseMatrix) lapCoo.toCpu();
    // lapCoo.free();
    this.mergedseeds = mergeSeeds(source, sink);
    LOGGER.trace("Rows to be removed: " + this.mergedseeds.length);
    SparseMatrixOj lapRowsRem = lap.removeRows(this.mergedseeds); // return is on cpu

    this.bvector.add(computeB(lapRowsRem, source));
    this.bvector.add(computeB(lapRowsRem, sink));

    SparseMatrixOj reducedL = lapRowsRem.removeCols(this.mergedseeds);

    timer.stop();
    LOGGER.info("Laplacian reduced in " + timer.toString());
    this.reducedLap = reducedL;
  }

  /**
   * Compute B.
   * 
   * @param lapRowsRem Laplacian with removed edges (rows).
   * @param indexes indexes of either source or sink, sorted
   * @return B vector
   * @see #getB()
   */
  DenseVectorOj computeB(SparseMatrixOj lapRowsRem, Integer[] indexes) {
    LOGGER.info("Computing B");
    StopWatch timer = StopWatch.createStarted();
    // 49sec
    // List<Integer> ilist = Arrays.asList(ArrayUtils.toObject(indexes));
    // // all cols except indexes
    // int[] colsRemove = IntStream.range(0, lap.getColNumber()).parallel()
    // .filter(x -> !ilist.contains(x)).toArray();

    // int[] indexescp = Arrays.copyOf(indexes, indexes.length);
    // Arrays.sort(indexescp); // TODO sort on output and remove copy
    List<Integer> colsRemovea = new ArrayList<Integer>(lapRowsRem.getColNumber());
    for (int i = 0; i < lapRowsRem.getColNumber(); i++) {
      int a = Arrays.binarySearch(indexes, i); // assuming indexes sorted
      if (a < 0) { // not found
        colsRemovea.add(i);
      }
    }
    int[] colsRemove = ArrayUtils.toPrimitive(colsRemovea.toArray(new Integer[0]));

    SparseMatrixOj tmp = lapRowsRem.removeCols(colsRemove);
    DenseVectorOj ret = tmp.sumAlongRows();
    int eln = ret.getElementNumber();
    for (int i = 0; i < eln; i++) {
      ret.getVal()[i] *= -1;
    }
    timer.stop();
    LOGGER.info("B computed in " + timer.toString());
    return ret;
  }

  /**
   * Main routine.
   * 
   * <p>Require incidence matrix computed by {@link #computeIncidence()}.
   * 
   * @param seed stack of size of segmented stack with pixel>0 for seeds.
   * @param seedVal value of seed in seed stack to solve for. Define seed pixels.
   * @return Segmented stack
   */
  @Override
  public ImageStack solve(ImageStack seed, int seedVal) throws Exception {
    computeIncidence();
    computeLaplacian(); // here there is first matrix created, decides CPU/GPU
    Integer[] seedIndices = getSourceIndices(seed, seedVal);
    computeReducedLaplacian(seedIndices, getIncidenceMatrix().getSinkBox());
    SparseMatrixOj reducedLapGpu = getReducedLap();
    SparseMatrixOj reducedLapGpuCsr = reducedLapGpu;
    // reducedLapGpu.free();
    LOGGER.info("Forward");
    float[] solvedFw = reducedLapGpuCsr.luSolve(bvector.get(0), true, options.getAlgOptions().maxit,
            options.getAlgOptions().tol);
    float[] solvedSeedsFw = incorporateSeeds(solvedFw, seedIndices,
            getIncidenceMatrix().getSinkBox(), lap.getColNumber());

    LOGGER.info("Backward");
    float[] solvedBw = reducedLapGpuCsr.luSolve(bvector.get(1), true, options.getAlgOptions().maxit,
            options.getAlgOptions().tol);
    float[] solvedSeedsBw = incorporateSeeds(solvedBw, getIncidenceMatrix().getSinkBox(),
            seedIndices, lap.getColNumber());

    float[] solvedSeeds = new float[solvedSeedsBw.length];
    for (int i = 0; i < solvedSeeds.length; i++) {
      solvedSeeds[i] = solvedSeedsFw[i] > solvedSeedsBw[i] ? 1.0f : 0.0f;
    }

    ImageStack ret = getSegmentedStack(solvedSeeds);// solvedSeeds
    // store raw solutions in case. Do not convert here as we do not know if getRawProbs will be
    // called.
    rawSoultions.add(solvedSeedsFw);
    rawSoultions.add(solvedSeedsBw);
    return ret;
  }

  /**
   * Set 1.0 in vector x at positions from seeds and 0 at positions from sink. Extend reduces
   * solution to full.
   * 
   * <p>Source is always 1.0, sink 0.0. Both vectors can be swapped.
   * 
   * @param x vector of solution
   * @param source indices of seeds (column ordered, got from
   *        {@link #getSourceIndices(ImageStack, int)}
   * @param sink sink indices, taken from e.g {@link IncidenceMatrixGenerator#computeSinkBox()}
   * @param verNumber
   * @return copy of vector x with incorporated seeds (1.0 set on positions from seeds, 0.0 at
   *         indexes from sink)
   */
  float[] incorporateSeeds(float[] x, Integer[] source, Integer[] sink, int verNumber) {
    LOGGER.info("Incorporating seeds into solution");
    final float valSeed = 1.0f;
    final float valSink = 0.0f;
    // sanity check
    if (x.length + source.length + sink.length != verNumber) {
      throw new IllegalArgumentException(
              "Number of vertices does not match. Maybe seeds and sink cotntain the same indices?");
    }
    StopWatch timer = StopWatch.createStarted();

    // all indices that are not in seed or sink, merged source and sink (1:27)
    // List<Integer> listSeedSink = Arrays.asList(ArrayUtils.toObject(this.mergedseeds));
    // all indices that are not in seed or sink
    // int[] ar = IntStream.range(0, verNumber).parallel().filter(i -> !listSeedSink.contains(i))
    // .toArray();

    if (this.mergedseeds == null) {
      LOGGER.trace("Perhaps debug mode?");
      this.mergedseeds = mergeSeeds(source, sink);
    }
    List<Integer> ara = new ArrayList<Integer>(verNumber);
    for (int i = 0; i < verNumber; i++) {
      int a = Arrays.binarySearch(this.mergedseeds, i); // assuming mergeseed sorted
      if (a < 0) { // not found
        ara.add(i);
      }
    }

    float[] ret = new float[verNumber];
    Stream.of(source).parallel().forEach(i -> ret[i] = valSeed);
    Stream.of(sink).parallel().forEach(i -> ret[i] = valSink);

    int l = 0;
    for (Integer arI : ara) {
      ret[arI] = x[l++];
    }
    timer.stop();
    LOGGER.info("Seeds incorporated into solution in " + timer.toString());
    return ret;
  }

  /**
   * Get computed reduced Laplacian.
   * 
   * @return the reducedLap
   */
  SparseMatrixOj getReducedLap() {
    return reducedLap;
  }

  /**
   * Get computed B vector.
   * 
   * @return the b
   */
  List<DenseVectorOj> getB() {
    return bvector;
  }

  /**
   * Change solution of the problem into ImageStack.
   * 
   * @param solution Solution vector, must
   * @return Stack
   */
  ImageStack getSegmentedStack(float[] solution) {
    int nrows = stack.getHeight();
    int ncols = stack.getWidth();
    int nz = stack.getSize();

    StopWatch timer = StopWatch.createStarted();
    if (solution.length != IncidenceMatrixGenerator.getNodesNumber(nrows, ncols, nz)) {
      throw new IllegalArgumentException(
              "length of sulution different than number of expected pixels.");
    }
    ImageStack ret = ImageStack.create(ncols, nrows, nz, 32);
    int[] ind = new int[3];
    for (int lin = 0; lin < solution.length; lin++) {
      IncidenceMatrixGenerator.lin20ind(lin, nrows, ncols, nz, ind);
      int y = ind[0]; // row
      int x = ind[1];
      int z = ind[2];
      ret.setVoxel(x, y, z, solution[lin]);
    }
    timer.stop();
    LOGGER.info("Result converted to stack in " + timer.toString());
    return ret;
  }

  /**
   * Return {@link IncidenceMatrixGenerator} object.
   * 
   * @return the img
   */
  IncidenceMatrixGenerator getIncidenceMatrix() {
    return img;
  }

  /**
   * Destroy all private CUDA objects.
   */
  @Override
  public void free() {
  }

  /**
   * Return GPU laplacian.
   * 
   * @return the lap
   */
  SparseMatrixOj getLap() {
    return lap;
  }

  @Override
  public List<ImageStack> getRawProbs() {
    List<ImageStack> rawProbs = new ArrayList<>(2);
    rawProbs.add(getSegmentedStack(rawSoultions.get(0)));
    rawProbs.add(getSegmentedStack(rawSoultions.get(1))); // last is BG

    return rawProbs;
  }

  @Override
  public void initilize() {
    // TODO Auto-generated method stub

  }
}
