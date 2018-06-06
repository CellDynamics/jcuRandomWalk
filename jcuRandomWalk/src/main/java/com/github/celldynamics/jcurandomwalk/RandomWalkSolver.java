package com.github.celldynamics.jcurandomwalk;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.stream.Stream;

import org.apache.commons.lang3.time.StopWatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.StackConverter;
import ij.process.StackProcessor;
import ij.process.StackStatistics;

public abstract class RandomWalkSolver implements IRandomWalkSolver {

  static final Logger LOGGER = LoggerFactory.getLogger(RandomWalkSolver.class.getName());

  ImageStack stack;
  IncidenceMatrixGenerator img;
  RandomWalkOptions options;

  public RandomWalkSolver() {
    // TODO Auto-generated constructor stub
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

  @Override
  public abstract ImageStack solve(ImageStack seed, int seedVal) throws Exception;

  @Override
  public abstract void free();

}
