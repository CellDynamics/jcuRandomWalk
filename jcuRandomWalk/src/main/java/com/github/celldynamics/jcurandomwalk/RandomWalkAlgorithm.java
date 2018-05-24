package com.github.celldynamics.jcurandomwalk;

import java.nio.file.Path;

import org.apache.commons.lang3.time.StopWatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixDevice;
import com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixHost;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.StackStatistics;
import jcuda.jcusparse.JCusparse;
import jcuda.runtime.JCuda;

/**
 * @author p.baniukiewicz
 *
 */
public class RandomWalkAlgorithm {

  static final Logger LOGGER = LoggerFactory.getLogger(RandomWalkAlgorithm.class.getName());

  ImageStack stack;
  IncidenceMatrixGenerator img;
  RandomWalkOptions options;

  /**
   * Default constructor,
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
   * @throws Exception
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
          img = IncidenceMatrixGenerator.restoreObject(fullToFilename.toString()); // load
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
    StopWatch timer = new StopWatch();
    timer.start();
    ISparseMatrix incidenceGpu = ((SparseMatrixHost) img.getIncidence()).toGpu().convert2csr();
    ISparseMatrix incidenceGpuT = incidenceGpu.transpose();

    ISparseMatrix wGpu = ((SparseMatrixHost) img.getWeights()).toGpu().convert2csr();
    // A'*W*A
    // ISparseMatrix ATW = incidenceGpuT.multiply(wGpu);
    // ISparseMatrix ATWA = ATW.multiply(incidenceGpu);
    ISparseMatrix lap = incidenceGpuT.multiply(wGpu).multiply(incidenceGpu);
    timer.stop();
    LOGGER.info("Laplacian computed in " + timer.toString());
    return lap;
  }

  /**
   * Apply default processing to stack.
   * 
   * <p>Apply 3x3 median filer 2D in each slice
   * 
   */
  public void processStack() {
    StopWatch timer = new StopWatch();
    timer.start();
    for (int z = 1; z <= stack.getSize(); z++) {
      stack.getProcessor(z).medianFilter();
    }
    StackStatistics stats = new StackStatistics(new ImagePlus("", stack));
    double min = stats.min;
    for (int z = 1; z <= stack.getSize(); z++) {
      stack.getProcessor(z).subtract(min);
      stack.getProcessor(z).sqrt();
    }
    stats = new StackStatistics(new ImagePlus("", stack));
    double max = 1 / stats.max;
    for (int z = 1; z <= stack.getSize(); z++) {
      stack.getProcessor(z).multiply(max);
    }
    timer.stop();
    LOGGER.info("Stack normalised in " + timer.toString());
  }

  /**
   * Must be called on very beginning.
   */
  public static void initilizeGpu() {
    JCusparse.setExceptionsEnabled(true);
    JCuda.setExceptionsEnabled(true);
    JCusparse.cusparseCreate(SparseMatrixDevice.handle);
  }

  /**
   * Must be called at very end.
   */
  public static void finish() {
    JCusparse.setExceptionsEnabled(false);
    JCuda.setExceptionsEnabled(false);
    JCusparse.cusparseDestroy(SparseMatrixDevice.handle);
  }
}
