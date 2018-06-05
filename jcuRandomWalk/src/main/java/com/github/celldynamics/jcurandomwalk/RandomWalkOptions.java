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
    public double sigmaGrad = 0.1; // TODO expose
    /**
     * sigmaMean.
     */
    public double sigmaMean = 1e6;
    /**
     * meanSource.
     */
    public double meanSource = 0.6;

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
    return "RandomWalkOptions [algOptions=" + algOptions + ", configBaseName=" + configBaseName
            + ", configBaseExt=" + configBaseExt + ", configFolder=" + configFolder + ", device="
            + device + ", useGPU=" + cpuOnly + ", stack=" + stack + ", seeds=" + seeds + ", output="
            + output + ", ifComputeIncidence=" + ifComputeIncidence + ", ifSaveIncidence="
            + ifSaveIncidence + ", ifApplyProcessing=" + ifApplyProcessing + ", debugLevel="
            + debugLevel + "]";
  }

}
