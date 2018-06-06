package com.github.celldynamics.jcurandomwalk;

import ij.ImageStack;

/**
 * @author baniu
 *
 */
public interface IRandomWalkSolver {

  public ImageStack solve(ImageStack seed, int seedVal) throws Exception;

  public void free();

  public void processStack();
}
