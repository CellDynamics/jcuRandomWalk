package com.github.celldynamics.jcurandomwalk;

import com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVectorDevice;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice;

import ij.ImageStack;

/**
 * Main routine for RW segmentation based on incomplete Cholesky factorization.
 * 
 * @author p.baniukiewicz
 * @author t.bretschneider
 *
 */
public class RandomWalkAlgorithmGpuChol extends RandomWalkAlgorithmGpuLU {

  /**
   * Constructor for tests.
   */
  public RandomWalkAlgorithmGpuChol() {
    super();
  }

  /**
   * Default constructor.
   * 
   * @param stack stack to be segmented
   * @param options options
   */
  public RandomWalkAlgorithmGpuChol(ImageStack stack, RandomWalkOptions options) {
    super(stack, options);
  }

  /*
   * (non-Javadoc)
   * 
   * @see com.github.celldynamics.jcurandomwalk.RandomWalkAlgorithmGpuLU#solve(ij.ImageStack, int)
   */
  @Override
  public ImageStack solve(ImageStack seed, int seedVal) throws Exception {
    computeIncidence();
    computeLaplacian(); // here there is first matrix created, decides CPU/GPU
    Integer[] seedIndices = getSourceIndices(seed, seedVal);
    computeReducedLaplacian(seedIndices, getIncidenceMatrix().getSinkBox());
    SparseMatrixDevice reducedLapGpuCsr = getReducedLap();

    SparseMatrixDevice triangle = reducedLapGpuCsr.getLowerTriangle();
    DenseVectorDevice chol = triangle.getCholesky();
    triangle.setUseCheating(true); // use always

    LOGGER.info("Forward");
    float[] solvedFw = triangle.luSolveSymmetric(bvector.get(0), chol,
            options.getAlgOptions().maxit, options.getAlgOptions().tol);
    float[] solvedSeedsFw = incorporateSeeds(solvedFw, seedIndices,
            getIncidenceMatrix().getSinkBox(), lap.getColNumber());

    LOGGER.info("Backward");
    float[] solvedBw = triangle.luSolveSymmetric(bvector.get(1), chol,
            options.getAlgOptions().maxit, options.getAlgOptions().tol);
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
    triangle.free();
    chol.free();
    return ret;
  }

}
