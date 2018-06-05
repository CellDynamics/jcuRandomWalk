package com.github.celldynamics.jcurandomwalk;

/**
 * Test runner.
 * 
 * @author baniu
 *
 */
//!>
public class JcuRandomWalkRun {

 
  /**
   * Main.
   * 
   * @param args args
   */
  public static void main(String[] args) {
    // String[] args = "app -s xx -i xxx -o yy -d".split(" ");
    //String[] args1 = "app -h".split(" ");
//     String[] args1 =
//     "app -i src/test/test_data/segment_test_normalised.tif -s src/test/test_data/segment_test_seeds.tif -o /tmp/solution.tif -dd -cpuonly"
//     .split(" ");

    String[] args1 =
            "app -i src/test/test_data/img_original_normalised.tif -s src/test/test_data/img_original_seeds.tif -o /tmp/test.tif -dd --cpuonly"
                    .split(" ");
    
    JcuRandomWalk.main(args1);

  }
  //!<
}
