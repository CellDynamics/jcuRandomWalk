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
//    String args1 = "app -h"; 
//    String args1 =
//           "app -i src/test/test_data/img_original_normalised_small.tif "
//           + "-s src/test/test_data/img_original_seeds_small.tif "
//           + "-o C:\\Users\\baniu\\AppData\\Local\\Temp\\solution_small.tif "
//           + "-dd "
//           + "-cpuonly "
//           + "-probmaps";

  String args1 =
          "app -i src/test/test_data/img_original_normalised_small.tif "
          + "-o C:\\Users\\baniu\\AppData\\Local\\Temp\\solution_small.tif "
          + "-dd "
          + "-cpuonly "
          + "-probmaps "
          + "--autoth 0.5";
      
  //   + "-s src/test/test_data/img_original_seeds_small.tif "
//    String[] args1 =
//            "app -i src/test/test_data/img_original_normalised.tif -s src/test/test_data/img_original_seeds.tif -o /tmp/test.tif"
//                    .split(" ");
    
    JcuRandomWalkCli.main(args1.split(" "));

  }
  //!<
}
