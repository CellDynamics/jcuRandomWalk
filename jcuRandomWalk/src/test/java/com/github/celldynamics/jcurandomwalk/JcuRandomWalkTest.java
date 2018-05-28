package com.github.celldynamics.jcurandomwalk;

import org.junit.Test;

/**
 * Show options.
 * 
 * @author baniu
 *
 */
public class JcuRandomWalkTest {

  /**
   * Test jcu random walk.
   *
   * @throws Exception the exception
   */
  @Test
  public void testJcuRandomWalk() throws Exception {
    String[] args = new String[] { "app", "-h" };
    JcuRandomWalk app = new JcuRandomWalk(args);
  }

}
