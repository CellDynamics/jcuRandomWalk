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
//    String args1 = "app -v"; 
    String args1 =
           "app -i src/test/test_data/img_original.tif "
           + "-o /tmp/solution_small.tif "
           + "-dd "
           + "--autoth 0.9 "
           + "--defaultprocessing";

//  String args1 =
//          "app -i src/test/test_data/ "
//          //+ "-o C:\\Users\\baniu\\AppData\\Local\\Temp\\solution_small.tif "
//          + "-dd "
//          + "-cpuonly "
//          + "-probmaps "
//          + "--autoth 0.5";
      
  //   + "-s src/test/test_data/img_original_seeds_small.tif "
//    String[] args1 =
//            "app -i src/test/test_data/img_original_normalised.tif -s src/test/test_data/img_original_seeds.tif -o /tmp/test.tif"
//                    .split(" ");
//    
//    String args1 =
//            "app -i src/test/test_data/sc/img_10.tif "
//            + "-s src/test/test_data/sc/img_10s.tif "
//            + "-o /tmp/solution_small.tif ";
    JcuRandomWalkCli.main(args1.split(" "));

  }
  //!<
}
