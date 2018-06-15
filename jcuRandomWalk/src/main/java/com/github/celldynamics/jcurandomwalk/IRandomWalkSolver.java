package com.github.celldynamics.jcurandomwalk;

import java.util.List;

import ij.ImageStack;

/**
 * RW solver interface. All algorithms should implement this.
 * 
 * @author baniu
 *
 */
public interface IRandomWalkSolver {

  /**
   * Perform RW segmentation.
   * 
   * @param seed stack of seeds. Pixels of seedVal stand for seeds.
   * @param seedVal value of pixels of seed to be considered as seeds.
   * @return segmented stack.
   * @throws Exception on eny error.
   */
  public ImageStack solve(ImageStack seed, int seedVal) throws Exception;

  /**
   * Release resources (if needed).
   */
  public void free();

  /**
   * Process input stack. Called if specified option is active.
   */
  public void processStack();

  /**
   * Return raw probability map for all initial seeds.
   * 
   * <p>At least two for FG and BG should be returned. Background map always last.
   * 
   * @return probability maps. BG always last.
   */
  public List<ImageStack> getRawProbs();

  /**
   * Check if there is at least non-zero pixel in seed map.
   * 
   * @param seed seed to test
   * @throws IllegalArgumentException if seed map is empty
   */
  public void validateSeeds(ImageStack seed);

  /**
   * Return stack.
   * 
   * @return stack hold by this object
   * @see #processStack()
   */
  public ImageStack getStack();
}
