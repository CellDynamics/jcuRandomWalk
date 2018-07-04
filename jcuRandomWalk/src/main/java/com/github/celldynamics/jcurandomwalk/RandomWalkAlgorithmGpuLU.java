package com.github.celldynamics.jcurandomwalk;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.time.StopWatch;

import com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVectorDevice;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseCoordinates;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice;

import ij.ImageStack;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;
import jcuda.runtime.JCuda;

/**
 * Main routine for RW segmentation.
 * 
 * @author p.baniukiewicz
 * @author t.bretschneider
 *
 */
public class RandomWalkAlgorithmGpuLU extends RandomWalkSolver {

  private cusparseHandle handle = null;
  private cublasHandle cublasHandle = null;
  SparseMatrixDevice reducedLap; // reduced laplacian
  SparseMatrixDevice lap; // full laplacian
  List<DenseVectorDevice> bvector = new ArrayList<>(); // right vector

  protected List<float[]> rawSoultions = new ArrayList<>(2); // in case getRawProbs is called
  private int[] mergedseeds; // Optimisation store

  /**
   * Constructor for tests.
   */
  RandomWalkAlgorithmGpuLU() {

  }

  /**
   * Default constructor.
   * 
   * @param stack stack to be segmented
   * @param options options
   */
  public RandomWalkAlgorithmGpuLU(ImageStack stack, RandomWalkOptions options) {
    super(stack, options);
  }

  /**
   * Compute Laplacian A'XA.
   * 
   * @see #getLap()
   */
  void computeLaplacian() {
    SparseMatrixDevice incidence;
    SparseMatrixDevice incidenceT;
    SparseMatrixDevice weight;
    LOGGER.info("Computing Laplacian");
    StopWatch timer = new StopWatch();
    timer.start();
    // IMatrix atw = null;
    SparseCoordinates incTmp = img.getIncidence();
    // FIXME no chaining
    incidence = SparseMatrixDevice.factory(incTmp, handle, cublasHandle).convert2csr();
    incidenceT = incidence.transpose();
    SparseCoordinates weiTmp = img.getWeights();
    weight = SparseMatrixDevice.factory(weiTmp, handle, cublasHandle).convert2csr();

    // A'*W*A
    // ISparseMatrix ATW = incidenceGpuT.multiply(wGpu);
    // ISparseMatrix ATWA = ATW.multiply(incidenceGpu);
    SparseMatrixDevice atw = incidenceT.multiply(weight);// .multiply(incidenceGpu);
    incidenceT.free();
    weight.free();
    this.lap = atw.multiply(incidence);
    atw.free();
    incidence.free();
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
    SparseMatrixDevice lapRowsRem = lap.removeRows(this.mergedseeds); // return is on cpu

    this.bvector.add(computeB(lapRowsRem, source));
    this.bvector.add(computeB(lapRowsRem, sink));

    SparseMatrixDevice reducedL = lapRowsRem.removeCols(this.mergedseeds);
    lapRowsRem.free();
    timer.stop();
    LOGGER.info("Laplacian reduced in " + timer.toString());
    this.reducedLap = reducedL.convert2csr();
  }

  /**
   * Compute B.
   * 
   * @param lapRowsRem Laplacian with removed edges (rows).
   * @param indexes indexes of either source or sink, sorted
   * @return B vector
   * @see #getB()
   */
  DenseVectorDevice computeB(SparseMatrixDevice lapRowsRem, Integer[] indexes) {
    LOGGER.info("Computing B");
    StopWatch timer = StopWatch.createStarted();
    // on gpu it could be completely different

    // List<Integer> colsRemovea = new ArrayList<Integer>(lapRowsRem.getColNumber());
    // for (int i = 0; i < lapRowsRem.getColNumber(); i++) {
    // int a = Arrays.binarySearch(indexes, i); // assuming indexes sorted
    // if (a < 0) { // not found
    // colsRemovea.add(i);
    // }
    // }

    List<Integer> colsRemovea = IntStream.range(0, lapRowsRem.getColNumber()).boxed().parallel()
            .filter(i -> Arrays.binarySearch(indexes, i) < 0).collect(Collectors.toList());

    int[] colsRemove = ArrayUtils.toPrimitive(colsRemovea.toArray(new Integer[0]));

    SparseMatrixDevice tmp = lapRowsRem.removeCols(colsRemove);
    float[] sum = tmp.sumAlongRowsIndices();
    for (int i = 0; i < sum.length; i++) {
      sum[i] *= -1;
    }
    DenseVectorDevice ret = new DenseVectorDevice(tmp.getRowNumber(), 1, sum);

    // DenseVectorDevice ret = lapRowsRem.sumAlongRows(indexes);
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
    StopWatch timer = new StopWatch();
    timer.start();
    computeIncidence();
    readTimer("INCIDENCE[ms]", timer);
    timer.start();
    computeLaplacian(); // here there is first matrix created, decides CPU/GPU
    readTimer("LAPLACIAN[ms]", timer);
    Integer[] seedIndices = getSourceIndices(seed, seedVal);
    timer.start();
    computeReducedLaplacian(seedIndices, getIncidenceMatrix().getSinkBox());
    readTimer("R-LAPLACIAN[ms]", timer);
    SparseMatrixDevice reducedLapGpuCsr = getReducedLap();
    Long sTmp = Long.valueOf(reducedLapGpuCsr.getRowNumber())
            * Long.valueOf(reducedLapGpuCsr.getColNumber());
    times.put("R-SIZE[N]", sTmp);
    times.put("R-NNZ[N]", Long.valueOf(reducedLapGpuCsr.getElementNumber()));
    times.put("S-SIZE[N]", (long) (stack.getHeight() * stack.getWidth() * stack.getSize()));
    reducedLapGpuCsr.setUseCheating(options.useCheating);

    LOGGER.info("Forward");
    timer.start();
    float[] solvedFw = reducedLapGpuCsr.luSolve(bvector.get(0), true, options.getAlgOptions().maxit,
            options.getAlgOptions().tol);
    readTimer("F-SOLVE[ms]", timer);
    float[] solvedSeedsFw = incorporateSeeds(solvedFw, seedIndices,
            getIncidenceMatrix().getSinkBox(), lap.getColNumber());

    LOGGER.info("Backward");
    timer.start();
    float[] solvedBw = reducedLapGpuCsr.luSolve(bvector.get(1), true, options.getAlgOptions().maxit,
            options.getAlgOptions().tol);
    readTimer("B-SOLVE[ms]", timer);
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
    reducedLapGpuCsr.free();
    saveRecord();// save stats
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
  SparseMatrixDevice getReducedLap() {
    return reducedLap;
  }

  /**
   * Get computed B vector.
   * 
   * @return the b
   */
  List<DenseVectorDevice> getB() {
    return bvector;
  }

  /**
   * Return {@link IncidenceMatrixGenerator} object.
   * 
   * @return the img
   */
  IncidenceMatrixGenerator getIncidenceMatrix() {
    return img;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcurandomwalk.IRandomWalkSolver#free()
   */
  @Override
  public void free() {
    if (reducedLap != null) {
      reducedLap.free();
    }
    if (lap != null) {
      lap.free();
    }
    for (DenseVectorDevice bv : this.bvector) {
      if (bv != null) {
        bv.free();
      }
    }
    finish();
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcurandomwalk.IRandomWalkSolver#initilize()
   */
  @Override
  public void initilize() {
    if (handle == null) {
      handle = new cusparseHandle();
      JCusparse.setExceptionsEnabled(true);
      JCuda.setExceptionsEnabled(true);
      JCusparse.cusparseCreate(handle);
    }
    if (cublasHandle == null) {
      cublasHandle = new cublasHandle();
      JCublas2.setExceptionsEnabled(true);
      JCublas2.cublasCreate(cublasHandle);
    }
  }

  /**
   * Must be called at very end.
   */
  public void finish() {
    try {
      if (handle != null) {
        JCusparse.setExceptionsEnabled(false);
        JCuda.setExceptionsEnabled(false);
        JCusparse.cusparseDestroy(handle);
        handle = null;
      }
      if (cublasHandle != null) {
        JCublas2.setExceptionsEnabled(false);
        JCublas2.cublasDestroy(cublasHandle);
        cublasHandle = null;
      }
    } catch (Error e) {
      LOGGER.error("Exception caugh during cleaning: " + e.getMessage().split("\n")[0]);
    }
  }

  /**
   * Return GPU laplacian.
   * 
   * @return the lap
   */
  SparseMatrixDevice getLap() {
    return lap;
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcurandomwalk.IRandomWalkSolver#getRawProbs()
   */
  @Override
  public List<ImageStack> getRawProbs() {
    List<ImageStack> rawProbs = new ArrayList<>(2);
    rawProbs.add(getSegmentedStack(rawSoultions.get(0)));
    rawProbs.add(getSegmentedStack(rawSoultions.get(1))); // last is BG

    return rawProbs;
  }

}
