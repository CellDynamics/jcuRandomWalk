/**
 * 
 */
package com.github.celldynamics.jcurandomwalk;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * @author p.baniukiewicz
 *
 */
public class RandomWalkOptions {
  /**
   * The tmpdir.
   */
  static String tmpdir = System.getProperty("java.io.tmpdir") + File.separator;

  public String configBaseName = "incidence";
  public String configBaseExt = ".ser";
  public Path configFolder = Paths.get(tmpdir);
}
