package com.github.celldynamics.jcurandomwalk;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice;

import ch.qos.logback.classic.Level;

/**
 * Pre-conditioner.
 * 
 * @author baniuk
 *
 */
enum Preconds {
  /**
   * Cholesky - symmetric.
   */
  CHOL,
  /**
   * LU.
   */
  LU
}

/**
 * Program and segmentation options.
 * 
 * @author p.baniukiewicz
 *
 */
public class RandomWalkOptions {

  /**
   * Default options.
   */
  public RandomWalkOptions() {

  }

  /**
   * Copy constructor.
   * 
   * @param src source
   */
  public RandomWalkOptions(RandomWalkOptions src) {
    this.thLevel = src.thLevel;
    this.algOptions = new AlgOptions(src.algOptions);
    this.configBaseName = src.configBaseName;
    this.configBaseExt = src.configBaseExt;
    this.seedSuffix = src.seedSuffix;
    this.outSuffix = src.outSuffix;
    this.configFolder = src.configFolder;
    this.device = src.device;
    this.cpuOnly = src.cpuOnly;
    this.rawProbMaps = src.rawProbMaps;
    this.stack = src.stack;
    this.seeds = src.seeds;
    this.output = src.output;
    this.ifComputeIncidence = src.ifComputeIncidence;
    this.ifSaveIncidence = src.ifSaveIncidence;
    this.ifApplyProcessing = src.ifApplyProcessing;
    this.gammaVal = src.gammaVal;
    this.debugLevel = src.debugLevel;
    this.useCheating = src.useCheating;
    this.preconditioner = src.preconditioner;
    this.bench = src.bench;
  }

  /**
   * The tmpdir.
   */
  static String tmpdir = System.getProperty("java.io.tmpdir") + File.separator;

  /**
   * Level of threshold. Negative value = option inactive.
   */
  Double thLevel = -1.0;
  /**
   * Options specific to algorithm.
   */
  public AlgOptions algOptions = new AlgOptions();
  /**
   * Name of incidence file.
   */
  public String configBaseName = "incidence";
  /**
   * Extension of incidence file.
   */
  public String configBaseExt = ".ser";
  /**
   * Suffix of seed image (if not specified).
   */
  public String seedSuffix = "_seeds.tif";
  /**
   * Suffix of output image (if not specified).
   */
  public String outSuffix = "_segm.tif";
  /**
   * Default path to save incidence.
   */
  public Path configFolder = Paths.get(System.getProperty("user.dir"));
  /**
   * GPU device to use.
   */
  public int device = 0;
  /**
   * If use cpu.
   */
  public boolean cpuOnly = false;
  /**
   * Save probability maps.
   */
  public boolean rawProbMaps = false;
  /**
   * Path to stack to segment.
   */
  public Path stack;
  /**
   * Path to seeds.
   */
  public Path seeds;
  /**
   * Path to output image.
   */
  public Path output;
  /**
   * Compute incidence always.
   */
  boolean ifComputeIncidence = true;
  /**
   * Save incidence ?.
   */
  boolean ifSaveIncidence = false;
  /**
   * Apply stack processing ?.
   */
  boolean ifApplyProcessing = false;
  /**
   * Gamma for processing stack with defaultprocessing option.
   * 
   * @see #ifApplyProcessing
   */
  public double gammaVal = 0.5;
  /**
   * Default debug level.
   */
  public Level debugLevel = Level.INFO;
  /**
   * Compute LU analysis only once per
   * {@link SparseMatrixDevice#luSolve(com.github.celldynamics.jcudarandomwalk.matrices.dense.DenseVectorDevice, boolean, int, float)}
   */
  public boolean useCheating = false;
  /**
   * Preconditioner used.
   */
  public Preconds preconditioner = Preconds.CHOL;

  /**
   * Path to benchmark results. If null no benchmark is computed.
   */
  public Path bench = null;

  /**
   * Set preconditioner from string.
   * 
   * @param preconditioner the preconditioner to set
   */
  public void setPreconditioner(String preconditioner) {
    this.preconditioner = Preconds.valueOf(preconditioner.trim().toUpperCase());
  }

  /**
   * Specific options for algorithm.
   * 
   * @author p.baniukiewicz
   *
   */
  public class AlgOptions {

    /**
     * Default options.
     */
    public AlgOptions() {

    }

    /**
     * Copy constructor.
     * 
     * @param src source
     */
    public AlgOptions(AlgOptions src) {
      this.maxit = src.maxit;
      this.tol = src.tol;
      this.sigmaGrad = src.sigmaGrad;
      this.sigmaMean = src.sigmaMean;
      this.meanSource = src.meanSource;
    }

    /**
     * Maximum number of iterations.
     */
    public int maxit = 200;
    /**
     * Tolerance.
     */
    public float tol = 1e-3f;
    /**
     * sigmaGrad.
     */
    public double sigmaGrad = 0.1;
    /**
     * sigmaMean.
     * 
     */
    public Double sigmaMean = 1e6;
    /**
     * meanSource.
     * 
     * <p>If null it is computed from current input image.
     */
    public Double meanSource = null;

    /*
     * (non-Javadoc)
     * 
     * @see java.lang.Object#toString()
     */
    @Override
    public String toString() {
      return "AlgOptions [maxit=" + maxit + ", tol=" + tol + ", sigmaGrad=" + sigmaGrad
              + ", sigmaMean=" + sigmaMean + ", meanSource=" + meanSource + "]";
    }

  }

  /**
   * Return options specific to algorithm.
   * 
   * @return the algOptions
   */
  public AlgOptions getAlgOptions() {
    return algOptions;
  }

  @Override
  public String toString() {
    return "RandomWalkOptions [thLevel=" + thLevel + ", algOptions=" + algOptions
            + ", configBaseName=" + configBaseName + ", configBaseExt=" + configBaseExt
            + ", seedSuffix=" + seedSuffix + ", outSuffix=" + outSuffix + ", configFolder="
            + configFolder + ", device=" + device + ", cpuOnly=" + cpuOnly + ", rawProbMaps="
            + rawProbMaps + ", stack=" + stack + ", seeds=" + seeds + ", output=" + output
            + ", ifComputeIncidence=" + ifComputeIncidence + ", ifSaveIncidence=" + ifSaveIncidence
            + ", ifApplyProcessing=" + ifApplyProcessing + ", gammaVal=" + gammaVal
            + ", debugLevel=" + debugLevel + ", useCheating=" + useCheating + ", preconditioner="
            + preconditioner + ", bench=" + bench + "]";
  }

}
