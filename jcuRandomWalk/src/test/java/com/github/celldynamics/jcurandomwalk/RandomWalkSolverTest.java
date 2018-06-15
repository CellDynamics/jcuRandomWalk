package com.github.celldynamics.jcurandomwalk;

import java.util.List;

import org.junit.Test;

import ij.ImageStack;

/**
 * @author baniuk
 *
 */
public class RandomWalkSolverTest {

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.JcuRandomWalkCli#validateSeeds(ij.ImageStack)}.
   */
  @Test(expected = IllegalArgumentException.class)
  public void testValidateSeeds() throws Exception {
    ImageStack stack = ImageStack.create(10, 5, 5, 8);
    RandomWalkSolver cli = new RandomWalkSolver() {

      @Override
      public ImageStack solve(ImageStack seed, int seedVal) throws Exception {
        // TODO Auto-generated method stub
        return null;
      }

      @Override
      public void free() {
        // TODO Auto-generated method stub

      }

      @Override
      public List<ImageStack> getRawProbs() {
        // TODO Auto-generated method stub
        return null;
      }

    };
    cli.validateSeeds(stack);
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.JcuRandomWalkCli#validateSeeds(ij.ImageStack)}.
   */
  @Test
  public void testValidateSeeds_1() throws Exception {
    ImageStack stack = ImageStack.create(10, 5, 5, 8);
    stack.setVoxel(5, 2, 1, 255);
    RandomWalkSolver cli = new RandomWalkSolver() {

      @Override
      public ImageStack solve(ImageStack seed, int seedVal) throws Exception {
        // TODO Auto-generated method stub
        return null;
      }

      @Override
      public void free() {
        // TODO Auto-generated method stub

      }

      @Override
      public List<ImageStack> getRawProbs() {
        // TODO Auto-generated method stub
        return null;
      }

    };
    cli.validateSeeds(stack);
  }
}
