package com.github.celldynamics.jcurandomwalk;

import org.junit.Test;

/**
 * Show options.
 * 
 * @author baniu
 *
 */
//!>
public class JcuRandomWalkTest {

  /**
   * Test jcu random walk.
   *
   * @throws Exception the exception
   */
  @Test
  public void testJcuRandomWalk() throws Exception {
    // String[] args = "app -s xx -i xxx -o yy -d".split(" ");
     String[] args = "app -h".split(" ");
//     String[] args =
//     "app -i src/test/test_data/segment_test_normalised.tif -s src/test/test_data/segment_test_seeds.tif -o /tmp/solution.tif -dd"
//     .split(" ");

//    String[] args =
//            "app -i src/test/test_data/img_original_normalised.tif -s src/test/test_data/img_original_seeds.tif -o /tmp/test.tif -dd"
//                    .split(" ");
    
    
    JcuRandomWalk app = new JcuRandomWalk(args);
  }
  //!<
}
