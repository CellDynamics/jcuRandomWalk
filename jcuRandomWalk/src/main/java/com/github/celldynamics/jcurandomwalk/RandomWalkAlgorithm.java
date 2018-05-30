package com.github.celldynamics.jcurandomwalk;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.time.StopWatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.ICudaLibHandles;
import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVector;
import com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVectorHost;
import com.github.celldynamics.jcudarandomwalk.matrices.dense.IDenseVector;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice;

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
 *
 */
public class RandomWalkAlgorithm {

  static final Logger LOGGER = LoggerFactory.getLogger(RandomWalkAlgorithm.class.getName());

  ImageStack stack;
  IncidenceMatrixGenerator img;
  RandomWalkOptions options;
  ISparseMatrix reducedLap; // reduced laplacian
  ISparseMatrix lap; // full laplacian
  IDenseVector b; // right vector

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
   * Compute incidence matrix for stack.
   * 
   * @param always if true compute always if false try to load first. If file has not been found
   *        compute new incidence matrix and save it.
   * @throws Exception file can not be read or deserialised
   */
  public void computeIncidence(boolean always) throws Exception {
    if (always) {
      LOGGER.info("Computing new incidence matrix");
      img = new IncidenceMatrixGenerator(stack);
    } else {
      String filename = options.configBaseName + "_" + stack.toString() + options.configBaseExt;
      Path fullToFilename = options.configFolder.resolve(filename);
      if (fullToFilename.toFile().exists()) { // try to load if exists
        try {
          img = IncidenceMatrixGenerator.restoreObject(fullToFilename.toString(), stack); // load
        } catch (Exception e) {
          LOGGER.error(
                  "Incidence file could not be restored (" + filename + "): " + e.getMessage());
          LOGGER.debug(e.getMessage(), e);
          throw e;
        }
      } else { // if does not exist generate new one and save
        LOGGER.info("Computing new incidence matrix");
        img = new IncidenceMatrixGenerator(stack);
        img.saveObject(fullToFilename.toString());
      }
    }
  }

  /**
   * Compute Laplacian A'XA.
   * 
   * @return Laplacian on GPU
   */
  public ISparseMatrix computeLaplacian() {
    IMatrix incidence;
    IMatrix incidenceT;
    IMatrix weight;
    StopWatch timer = new StopWatch();
    timer.start();
    if (options.useGPU) {
      incidence = ((ISparseMatrix) img.getIncidence().toGpu()).convert2csr();
      incidenceT = incidence.transpose();

      weight = ((ISparseMatrix) img.getWeights().toGpu()).convert2csr();
    } else {
      incidence = ((ISparseMatrix) img.getIncidence()).convert2csr();
      incidenceT = incidence.transpose();

      weight = ((ISparseMatrix) img.getWeights()).convert2csr();
    }
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
    return this.lap;
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
   * @param source list of indices to remove - sources
   * @param sink second list of indices to remove, merged with the first one
   */
  public void computeReducedLaplacian(int[] source, int[] sink) {
    if (lap.getColNumber() != lap.getRowNumber()) {
      throw new IllegalArgumentException("Matrix should be square");
    }
    StopWatch timer = new StopWatch();
    timer.start();
    // ISparseMatrix lapCoo = lap.convert2coo();
    // ISparseMatrix cpuL;
    // cpuL = (ISparseMatrix) lapCoo.toCpu();
    // lapCoo.free();
    int[] merged = mergeSeeds(source, sink);
    LOGGER.trace("Rows to be removed: " + merged.length);
    IMatrix lapRowsRem = lap.removeRows(merged); // return is on cpu

    this.b = computeB(lapRowsRem, source);
    ISparseMatrix reducedL = (ISparseMatrix) lapRowsRem.removeCols(merged);
    // if (reducedL instanceof SparseMatrixHost) {
    // ((SparseMatrixHost) reducedL).compressSparseIndices(); // FIXME is necessary?
    // }
    lapRowsRem.free();
    timer.stop();
    LOGGER.info("Laplacian reduced in " + timer.toString());
    this.reducedLap = reducedL;
  }

  /**
   * Compute B.
   * 
   * @param lap Laplacian with removed edges (rows).
   * @param indexes indexes of either source or sink
   * @return B vector
   */
  IDenseVector computeB(IMatrix lap, int[] indexes) {
    StopWatch timer = StopWatch.createStarted();
    if (lap instanceof SparseMatrixDevice) {
      // on gpu it could be completely different
      timer.stop();
      throw new NotImplementedException("not implemented");
    } else {
      List<Integer> ilist = Arrays.asList(ArrayUtils.toObject(indexes));
      // all cols except indexes
      int[] colsRemove =
              IntStream.range(0, lap.getColNumber()).filter(x -> !ilist.contains(x)).toArray();
      IMatrix tmp = lap.removeCols(colsRemove);
      // if (tmp instanceof SparseMatrixHost) {
      // ((SparseMatrixHost) tmp).compressSparseIndices(); // FIXME is necessary?
      // }
      ISparseMatrix ret = (ISparseMatrix) tmp.sumAlongRows();
      for (int i = 0; i < ret.getVal().length; i++) {
        ret.getVal()[i] *= -1;
      }
      IDenseVector dwret = DenseVector.denseVectorFactory(new DenseVectorHost(), ret.getRowNumber(),
              1, ret.getVal());
      timer.stop();
      LOGGER.info("B computed in " + timer.toString());
      return dwret;
    }
  }

  /**
   * Merge indexes from source and sink into one array and removes duplicates.
   * 
   * @param source indexes (column ordered) of pixels that are the source
   * @param sink indexes (column ordered) of pixels that are the sink
   * @return merged two input arrays without duplicates.
   */
  public int[] mergeSeeds(int[] source, int[] sink) {
    int[] ret = Stream
            .concat(IntStream.of(source).parallel().boxed(), IntStream.of(sink).parallel().boxed())
            .distinct().mapToInt(i -> i).toArray();
    Arrays.sort(ret);
    return ret;
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
   * @param seed
   * @return Segmented stack
   */
  public ImageStack solve(ImageStack seed) {
    computeLaplacian();
    int[] seedIndices = getSourceIndices(seed);
    computeReducedLaplacian(seedIndices, getImg().getSinkBox());
    ISparseMatrix reducedLapGpu = (ISparseMatrix) getReducedLap().toGpu();
    ISparseMatrix reducedLapGpuCsr = reducedLapGpu.convert2csr();
    // reducedLapGpu.free();
    float[] solved = reducedLapGpuCsr.luSolve(b, true);
    float[] solvedSeeds =
            incorporateSeeds(solved, seedIndices, getImg().getSinkBox(), lap.getColNumber());

    ImageStack ret = getSegmentedStack(solvedSeeds);
    reducedLapGpuCsr.free();
    reducedLapGpu.free();
    return ret;
  }

  /**
   * Set 1.0 in vector x at positions from seeds and 0 at positions from sink. Extend reduces
   * solution to full.
   * 
   * <p>Seeds are always marked by 1.0, sinks by 0.0. Both vectors can be swapped.
   * 
   * @param x vector of solution
   * @param source indices of seeds (column ordered, got from {@link #getSourceIndices(ImageStack)}
   * @param sink sink indices, taken from e.g {@link IncidenceMatrixGenerator#computeSinkBox()}
   * @param verNumber
   * @return copy of vector x with incorporated seeds (1.0 set on positions from seeds, 0.0 at
   *         indexes from sink)
   */
  float[] incorporateSeeds(float[] x, int[] source, int[] sink, int verNumber) {
    final float valSeed = 1.0f;
    final float valSink = 0.0f;
    // sanity check
    if (x.length + source.length + sink.length != verNumber) {
      throw new IllegalArgumentException(
              "Number of vertices does not match. Maybe seeds and sink cotntain the same indices?");
    }
    StopWatch timer = StopWatch.createStarted();
    // all indices that are not in seed or sink
    List<Integer> listSeedSink = Stream
            .concat(IntStream.of(source).parallel().boxed(), IntStream.of(sink).parallel().boxed())
            .distinct().collect(Collectors.toList()); // merged source and sink
    // all indices that are not in seed or sink
    IntStream botSeedSink = IntStream.range(0, verNumber).filter(i -> !listSeedSink.contains(i));

    float[] ret = new float[verNumber];
    IntStream.of(source).forEach(i -> ret[i] = valSeed);
    IntStream.of(sink).forEach(i -> ret[i] = valSink);
    AtomicInteger counter = new AtomicInteger(0);
    botSeedSink.forEach(i -> ret[i] = x[counter.getAndIncrement()]);
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
  public IDenseVector getB() {
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
   * <p>It is possible to use this method for obtaining sink indices by reversing stack.
   * 
   * @param seedStack Pixels with intensity >0 are considered as seed. Expect 8-bit binary stacks.
   * @return array of indices.
   */
  int[] getSourceIndices(ImageStack seedStack) {
    if (!seedStack.getProcessor(1).isBinary()) {
      throw new IllegalArgumentException("Seed stack is not binary");
    }
    StopWatch timer = new StopWatch();
    timer.start();
    int l = 0;
    StackStatistics stat = new StackStatistics(new ImagePlus("", seedStack));
    int[] ret = new int[stat.histogram[stat.histogram.length - 1]];
    int[] ind = new int[3];
    for (int z = 0; z < seedStack.getSize(); z++) {
      for (int x = 0; x < seedStack.getWidth(); x++) {
        for (int y = 0; y < seedStack.getHeight(); y++) {
          if (seedStack.getVoxel(x, y, z) > 0) {
            ind[0] = y; // row
            ind[1] = x;
            ind[2] = z;
            ret[l++] = IncidenceMatrixGenerator.ind20lin(ind, seedStack.getHeight(),
                    seedStack.getWidth(), seedStack.getSize());
          }
        }
      }
    }
    timer.stop();
    LOGGER.info("Seeds processed in " + timer.toString());
    return ret;
  }

  /**
   * Return {@link IncidenceMatrixGenerator} object.
   * 
   * @return the img
   */
  public IncidenceMatrixGenerator getImg() {
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
    if (b != null) {
      b.free();
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
}
