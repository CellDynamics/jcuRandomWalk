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
    
    String args1 =
           "app -i /local/internal_tmp/baniuk/rw/z_stack_deconvolved_t7.tif "
           + "-o /home/baniuk/external/rw/folder/res/solution_t7.tif "
           + "-dd "
           + "--autoth 0.95 "
           + "--defaultprocessing 0.4 "
           + "--sigmaMean 0.05 "
           + "--sigmaGrad 0.1 "
           + "--probmaps";
//
//    String args1 =
//            "app -i /local/internal_tmp/baniuk/rw/folder "
//            + "-o /local/internal_tmp/baniuk/rw/folder/res "
//            + "-dd "
//            + "--autoth 0.95 "
//            + "--device -2 "
//            + "--defaultprocessing 0.4 "
//            + "--sigmaMean 0.05 "
//            + "--sigmaGrad 0.1 "
//            + "--usecheating";
    
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
