package com.github.celldynamics.jcurandomwalk;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.stream.Stream;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.StackConverter;
import ij.process.StackProcessor;
import ij.process.StackStatistics;

/**
 * Core of the solver.
 * 
 * @author baniu
 *
 */
public abstract class RandomWalkSolver implements IRandomWalkSolver {

  static final Logger LOGGER = LoggerFactory.getLogger(RandomWalkSolver.class.getName());

  ImageStack stack;
  IncidenceMatrixGenerator img;
  RandomWalkOptions options;

  /**
   * Empty constructor.
   */
  public RandomWalkSolver() {
  }

  /**
   * Default constructor.
   * 
   * @param stack stack to be segmented
   * @param options options
   */
  public RandomWalkSolver(ImageStack stack, RandomWalkOptions options) {
    this.options = options;
    this.stack = stack;
  }

  /**
   * Change solution of the problem into ImageStack.
   * 
   * @param solution Solution vector
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
    LOGGER.info("Vector converted to stack in " + timer.toString());
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
   * @param seedStack Pixels with intensity >0 are considered as seed.
   * @param value value of seed pixels
   * @return sorted array of indices.
   */
  Integer[] getSourceIndices(ImageStack seedStack, int value) {
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
   * Compute/reload incidence matrix for stack. Depending on options.
   * 
   * @throws Exception file can not be read or deserialised
   */
  protected void computeIncidence() throws Exception {
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
   * Apply default processing to stack. Assumes 8-bit imput.
   * 
   * <p>Apply 3x3 median filer 2D in each slice and normalisation.
   * 
   */
  @Override
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
   * Merge indexes from source and sink into one array and removes duplicates.
   * 
   * @param source indexes (column ordered) of pixels that are the source
   * @param sink indexes (column ordered) of pixels that are the sink
   * @return merged two input arrays without duplicates. Sorted.
   */
  protected int[] mergeSeeds(Integer[] source, Integer[] sink) {
    LOGGER.debug("Merging seeds");
    int[] mergedseeds = Stream.concat(Stream.of(source).parallel(), Stream.of(sink).parallel())
            .distinct().mapToInt(i -> i).toArray();
    Arrays.sort(mergedseeds);
    return mergedseeds;
  }

}
