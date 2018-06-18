package com.github.celldynamics.jcurandomwalk;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

import ch.qos.logback.classic.Level;

/**
 * Program and segmentation options.
 * 
 * @author p.baniukiewicz
 *
 */
public class RandomWalkOptions {

  /**
   * The tmpdir.
   */
  static String tmpdir = System.getProperty("java.io.tmpdir") + File.separator;

  /**
   * Level of threshold. Negative value = option inacive.
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
   * Default debug level.
   */
  Level debugLevel = Level.INFO;

  /**
   * Specific options for algorithm.
   * 
   * @author p.baniukiewicz
   *
   */
  public class AlgOptions {
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

  /*
   * (non-Javadoc)
   * 
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString() {
    return "RandomWalkOptions [thLevel=" + thLevel + ", algOptions=" + algOptions.toString()
            + ", configBaseName=" + configBaseName + ", configBaseExt=" + configBaseExt
            + ", seedSuffix=" + seedSuffix + ", outSuffix=" + outSuffix + ", configFolder="
            + configFolder + ", device=" + device + ", cpuOnly=" + cpuOnly + ", rawProbMaps="
            + rawProbMaps + ", stack=" + stack + ", seeds=" + seeds + ", output=" + output
            + ", ifComputeIncidence=" + ifComputeIncidence + ", ifSaveIncidence=" + ifSaveIncidence
            + ", ifApplyProcessing=" + ifApplyProcessing + ", debugLevel=" + debugLevel + "]";
  }

}
