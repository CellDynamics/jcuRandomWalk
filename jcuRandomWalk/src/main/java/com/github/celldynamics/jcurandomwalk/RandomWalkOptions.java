package com.github.celldynamics.jcurandomwalk;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Segmentation options.
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
  public Path configFolder = Paths.get(tmpdir);
  /**
   * GPU device to use.
   */
  public int device = 0;
  /**
   * If use gpu.
   */
  public boolean useGPU = true;
  /**
   * Path to stack to segment.
   */
  public Path stack;
  /**
   * Path to seeds.
   */
  public Path seeds;
}
