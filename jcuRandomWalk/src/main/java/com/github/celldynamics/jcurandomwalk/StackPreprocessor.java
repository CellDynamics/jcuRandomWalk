package com.github.celldynamics.jcurandomwalk;

import org.apache.commons.lang3.time.StopWatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.StackConverter;
import ij.process.StackProcessor;
import ij.process.StackStatistics;

/**
 * Pre-process stack.
 * 
 * @author baniuk
 *
 */
public class StackPreprocessor {

  static final Logger LOGGER = LoggerFactory.getLogger(StackPreprocessor.class.getName());

  /**
   * Apply default processing to stack. Assumes 8-bit imput.
   * 
   * <p>Apply 3x3 median filer 2D in each slice and normalisation.
   * 
   */
  public ImageStack processStack(ImageStack stack) {
    StopWatch timer = new StopWatch();
    timer.start();
    LOGGER.info("Processing stack");
    ImageStack filterOut =
            ImageStack.create(stack.getWidth(), stack.getHeight(), stack.getSize(), 8);
    new StackProcessor(stack).filter3D(filterOut, 1, 1, 1, 0, stack.getSize(),
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
    LOGGER.info("Stack normalised in " + timer.toString());
    return ip.getStack();
  }

}
