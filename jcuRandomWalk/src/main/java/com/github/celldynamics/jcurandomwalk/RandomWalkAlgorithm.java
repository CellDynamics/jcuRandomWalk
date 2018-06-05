package com.github.celldynamics.jcurandomwalk;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.time.StopWatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.ICudaLibHandles;
import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.dense.IDenseVector;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixOj;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.StackConverter;
import ij.process.StackProcessor;
import ij.process.StackStatistics;
import jcuda.jcusparse.JCusparse;
import jcuda.runtime.JCuda;

/**
 * Main routine for RW segmentation.
 * 
 * @author p.baniukiewicz
 * @author t.bretschneider
 *
 */
public class RandomWalkAlgorithm {

  static final Logger LOGGER = LoggerFactory.getLogger(RandomWalkAlgorithm.class.getName());

  ImageStack stack;
  IncidenceMatrixGenerator img;
  RandomWalkOptions options;
  ISparseMatrix reducedLap; // reduced laplacian
  ISparseMatrix lap; // full laplacian
  List<IDenseVector> b = new ArrayList<>(); // right vector

  private int[] mergedseeds; // Optimisation store

  /**
   * Constructor for tests.
   */
  RandomWalkAlgorithm() {

  }

  /**
   * Default constructor.
   * 
   * @param stack stack to be segmented
   * @param options options
   */
  public RandomWalkAlgorithm(ImageStack stack, RandomWalkOptions options) {
    this.options = options;
    this.stack = stack;
  }

  /**
   * Compute/reload incidence matrix for stack. Depending on options.
   * 
   * @throws Exception file can not be read or deserialised
   */
  public void computeIncidence() throws Exception {
    String filename = options.configBaseName + "_" + stack.toString() + options.configBaseExt;
    Path fullToFilename = options.configFolder.resolve(filename);
    if (options.ifComputeIncidence) {
      LOGGER.info("Computing new incidence matrix");
      img = new IncidenceMatrixGenerator(stack, options.getAlgOptions());
      if (options.ifSaveIncidence) {
        img.saveObject(fullToFilename.toString());
      }
    } else {
      if (fullToFilename.toFile().exists()) { // try to load if exists
        try {
          img = IncidenceMatrixGenerator.restoreObject(fullToFilename.toString(), stack,
                  options.getAlgOptions()); // load
        } catch (Exception e) {
          LOGGER.error(
                  "Incidence file could not be restored (" + filename + "): " + e.getMessage());
          LOGGER.debug(e.getMessage(), e);
          throw e;
        }
      } else { // if does not exist generate new one and save
        LOGGER.info("Computing new incidence matrix");
        img = new IncidenceMatrixGenerator(stack, options.getAlgOptions());
        img.saveObject(fullToFilename.toString());
      }
    }
  }

  /**
   * Compute Laplacian A'XA.
   * 
   * @see #getLap()
   */
  void computeLaplacian() {
    IMatrix incidence;
    IMatrix incidenceT;
    IMatrix weight;
    LOGGER.info("Computing Laplacian");
    StopWatch timer = new StopWatch();
    timer.start();
    // IMatrix atw = null;
    if (options.cpuOnly == false) {
      incidence = img.getIncidence().toSparse(new SparseMatrixDevice()).convert2csr();
      incidenceT = incidence.transpose();
      weight = img.getWeights().toSparse(new SparseMatrixDevice()).convert2csr();
    } else {
      incidence = img.getIncidence().toSparse(new SparseMatrixOj());
      incidenceT = incidence.transpose();
      weight = img.getWeights().toSparse(new SparseMatrixOj());

      // ElementsSupplier<Double> aTw = ((SparseMatrixOj) weight).mat
      // .premultiply(((SparseMatrixOj) incidence).mat.transpose());
      // SparseStore<Double> mm =
      // SparseStore.PRIMITIVE.make(incidence.getColNumber(), incidence.getRowNumber());
      // aTw.supplyTo(mm);
      // atw = new SparseMatrixOj(mm, incidence.getColNumber(), incidence.getRowNumber());

    }
    LOGGER.info("Base class: " + incidence.getClass().getSimpleName());
    // A'*W*A
    // ISparseMatrix ATW = incidenceGpuT.multiply(wGpu);
    // ISparseMatrix ATWA = ATW.multiply(incidenceGpu);
    IMatrix atw = incidenceT.multiply(weight);// .multiply(incidenceGpu);
    incidenceT.free();
    weight.free();
    this.lap = (ISparseMatrix) atw.multiply(incidence);
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
    IMatrix lapRowsRem = lap.removeRows(this.mergedseeds); // return is on cpu

    this.b.add(computeB(lapRowsRem, source));
    this.b.add(computeB(lapRowsRem, sink));

    ISparseMatrix reducedL = (ISparseMatrix) lapRowsRem.removeCols(this.mergedseeds);

    lapRowsRem.free();
    timer.stop();
    LOGGER.info("Laplacian reduced in " + timer.toString());
    this.reducedLap = reducedL;
  }

  /**
   * Compute B.
   * 
   * @param lap Laplacian with removed edges (rows).
   * @param indexes indexes of either source or sink, sorted
   * @return B vector
   * @see #getB()
   */
  IDenseVector computeB(IMatrix lap, Integer[] indexes) {
    LOGGER.info("Computing B");
    StopWatch timer = StopWatch.createStarted();
    if (lap instanceof SparseMatrixDevice) {
      // on gpu it could be completely different
      timer.stop();
      throw new NotImplementedException("not implemented");
    } else {
      // 49sec
      // List<Integer> ilist = Arrays.asList(ArrayUtils.toObject(indexes));
      // // all cols except indexes
      // int[] colsRemove = IntStream.range(0, lap.getColNumber()).parallel()
      // .filter(x -> !ilist.contains(x)).toArray();

      // int[] indexescp = Arrays.copyOf(indexes, indexes.length);
      // Arrays.sort(indexescp); // TODO sort on output and remove copy
      List<Integer> colsRemovea = new ArrayList<Integer>(lap.getColNumber());
      for (int i = 0; i < lap.getColNumber(); i++) {
        int a = Arrays.binarySearch(indexes, i); // assuming indexes sorted
        if (a < 0) { // not found
          colsRemovea.add(i);
        }
      }
      int[] colsRemove = ArrayUtils.toPrimitive(colsRemovea.toArray(new Integer[0]));

      IMatrix tmp = lap.removeCols(colsRemove);
      IMatrix ret = tmp.sumAlongRows();
      int eln = ret.getElementNumber();
      for (int i = 0; i < eln; i++) {
        ret.getVal()[i] *= -1;
      }
      timer.stop();
      LOGGER.info("B computed in " + timer.toString());
      return (IDenseVector) ret;
    }
  }

  /**
   * Merge indexes from source and sink into one array and removes duplicates.
   * 
   * @param source indexes (column ordered) of pixels that are the source
   * @param sink indexes (column ordered) of pixels that are the sink
   * @return merged two input arrays without duplicates. Sorted.
   */
  private int[] mergeSeeds(Integer[] source, Integer[] sink) {
    LOGGER.debug("Merging seeds");
    int[] mergedseeds = Stream.concat(Stream.of(source).parallel(), Stream.of(sink).parallel())
            .distinct().mapToInt(i -> i).toArray();
    Arrays.sort(mergedseeds);
    return mergedseeds;
  }

  /**
   * Apply default processing to stack. Assumes 8-bit imput.
   * 
   * <p>Apply 3x3 median filer 2D in each slice and normalisation.
   * 
   */
  public void processStack() {
    StopWatch timer = new StopWatch();
    timer.start();
    LOGGER.info("Processing stack");
    ImageStack filterOut =
            ImageStack.create(stack.getWidth(), stack.getHeight(), stack.getSize(), 8);
    new StackProcessor(this.stack).filter3D(filterOut, 1, 1, 1, 0, stack.getSize(),
            StackProcessor.FILTER_MEDIAN);
    ImagePlus ip = new ImagePlus("", filterOut);

    StackStatistics stats = new StackStatistics(ip);
    double min = stats.min;
    double max = stats.max; // test if computed for the whole stack
    // convert stack to Float
    StackConverter stc = new StackConverter(ip);
    stc.convertToGray32();

    for (int z = 1; z <= ip.getImageStackSize(); z++) {
      ip.getStack().getProcessor(z).subtract(min);
      ip.getStack().getProcessor(z).sqrt();
    }
    stats = new StackStatistics(ip);
    max = 1 / stats.max;
    for (int z = 1; z <= ip.getImageStackSize(); z++) {
      ip.getStack().getProcessor(z).multiply(max);
    }
    timer.stop();
    this.stack = ip.getStack();
    LOGGER.info("Stack normalised in " + timer.toString());
  }

  /**
   * Main routine.
   * 
   * <p>Require incidence matrix computed by {@link #computeIncidence(boolean)}.
   * 
   * @param seed stack of size of segmented stack with pixel>0 for seeds.
   * @param seedVal value of seed in seed stack to solve for. Define seed pixels.
   * @return Segmented stack
   */
  public ImageStack solve(ImageStack seed, int seedVal) {
    if (getIncidenceMatrix() == null) {
      throw new IllegalStateException("Incidence matrix should be computed first");
    }
    computeLaplacian(); // here there is first matrix created, decides CPU/GPU
    Integer[] seedIndices = getSourceIndices(seed, seedVal);
    computeReducedLaplacian(seedIndices, getIncidenceMatrix().getSinkBox());
    ISparseMatrix reducedLapGpu = (ISparseMatrix) getReducedLap().toGpu();
    ISparseMatrix reducedLapGpuCsr = reducedLapGpu.convert2csr();
    // reducedLapGpu.free();
    LOGGER.info("Forward");
    float[] solved_fw = reducedLapGpuCsr.luSolve(b.get(0), true, options.getAlgOptions().maxit,
            options.getAlgOptions().tol);
    float[] solvedSeeds_fw = incorporateSeeds(solved_fw, seedIndices,
            getIncidenceMatrix().getSinkBox(), lap.getColNumber());

    LOGGER.info("Backward");
    float[] solved_bw = reducedLapGpuCsr.luSolve(b.get(1), true, options.getAlgOptions().maxit,
            options.getAlgOptions().tol);
    float[] solvedSeeds_bw = incorporateSeeds(solved_bw, getIncidenceMatrix().getSinkBox(),
            seedIndices, lap.getColNumber());

    float[] solvedSeeds = new float[solvedSeeds_bw.length];
    for (int i = 0; i < solvedSeeds.length; i++) {
      solvedSeeds[i] = solvedSeeds_fw[i] > solvedSeeds_bw[i] ? 1.0f : 0.0f;
    }

    ImageStack ret = getSegmentedStack(solvedSeeds);// solvedSeeds
    reducedLapGpuCsr.free();
    reducedLapGpu.free();
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
  public ISparseMatrix getReducedLap() {
    return reducedLap;
  }

  /**
   * Get computed B vector.
   * 
   * @return the b
   */
  public List<IDenseVector> getB() {
    return b;
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
   * Obtain indices of seed pixels from stack. Indices are computed in column-ordered manner.
   * 
   * <p>Ordering:
   * 0 3
   * 1 4
   * 2 5
   * 
   * <p>It is possible to use this method for obtaining sink indices by reversing stack. Note that
   * output is sorted.
   * 
   * @param seedStack Pixels with intensity >0 are considered as seed. Expect 8-bit binary stacks.
   * @param value value of seed pixels
   * @return sorted array of indices.
   */
  Integer[] getSourceIndices(ImageStack seedStack, int value) {
    if (!seedStack.getProcessor(1).isBinary()) {
      throw new IllegalArgumentException("Seed stack is not binary");
    }
    LOGGER.info("Converting seed stack to indices");
    StopWatch timer = new StopWatch();
    timer.start();
    int l = 0;
    StackStatistics stat = new StackStatistics(new ImagePlus("", seedStack));
    int[] ret = new int[stat.histogram[stat.histogram.length - 1]];
    int[] ind = new int[3];
    for (int z = 0; z < seedStack.getSize(); z++) {
      for (int x = 0; x < seedStack.getWidth(); x++) {
        for (int y = 0; y < seedStack.getHeight(); y++) {
          if (seedStack.getVoxel(x, y, z) == value) {
            ind[0] = y; // row
            ind[1] = x;
            ind[2] = z;
            ret[l++] = IncidenceMatrixGenerator.ind20lin(ind, seedStack.getHeight(),
                    seedStack.getWidth(), seedStack.getSize());
          }
        }
      }
    }
    Integer[] retob = ArrayUtils.toObject(ret);
    Arrays.sort(retob);
    timer.stop();
    LOGGER.info("Seeds processed in " + timer.toString());
    return retob;
  }

  /**
   * Return {@link IncidenceMatrixGenerator} object.
   * 
   * @return the img
   */
  public IncidenceMatrixGenerator getIncidenceMatrix() {
    return img;
  }

  /**
   * Destroy all private CUDA objects.
   */
  public void free() {
    if (reducedLap != null) {
      reducedLap.free();
    }
    if (lap != null) {
      lap.free();
    }
    for (IDenseVector bv : this.b) {
      if (bv != null) {
        bv.free();
      }
    }
  }

  /**
   * Must be called on very beginning.
   */
  public static void initilizeGpu() {
    JCusparse.setExceptionsEnabled(true);
    JCuda.setExceptionsEnabled(true);
    JCusparse.cusparseCreate(ICudaLibHandles.handle);
  }

  /**
   * Must be called at very end.
   */
  public static void finish() {
    try {
      JCusparse.setExceptionsEnabled(false);
      JCuda.setExceptionsEnabled(false);
      JCusparse.cusparseDestroy(ICudaLibHandles.handle);
    } catch (Exception e) {
      LOGGER.error("Exception caugh during cleaning: " + e.getMessage());
    }
  }

  /**
   * @return the lap
   */
  public ISparseMatrix getLap() {
    return lap;
  }
}
