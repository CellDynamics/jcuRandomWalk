package com.github.celldynamics.jcurandomwalk;

import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaSetDevice;

import java.nio.file.Paths;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.lang3.time.StopWatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.versioning.ToolVersion;
import com.github.celldynamics.versioning.ToolVersionStruct;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;

/**
 * Main application class.
 */
public class JcuRandomWalk {
  static final Logger LOGGER = LoggerFactory.getLogger(JcuRandomWalk.class.getName());
  private int cliErrorStatus = 0; // returned to main
  private RandomWalkOptions rwOptions;
  private Options cliOptions = null;

  /**
   * Default constructor using default options.
   */
  public JcuRandomWalk() {
    rwOptions = new RandomWalkOptions();
  }

  /**
   * Constructor - parser initialised by cli. Main runner
   * 
   * @param args args passed from cli.
   */
  public JcuRandomWalk(String[] args) {
    this();
    CommandLineParser parser = new DefaultParser();

    Option helpOption = Option.builder("h").desc("print this message").longOpt("help").build();

    Option verOption = Option.builder("v").desc("Show version").longOpt("version").build();

    Option saveIncOption =
            Option.builder().desc("Save incdence matrix for this size of the stack. Default is "
                    + rwOptions.ifSaveIncidence).longOpt("saveincidence").build();

    Option loadIncOption = Option.builder()
            .desc("Load incidence matrix for this size of stack or compute new one and save it"
                    + " if relevant file has not been found. Default is "
                    + !rwOptions.ifComputeIncidence)
            .longOpt("loadincidence").build();

    Option defProcessOption = Option.builder()
            .desc("Apply default processing to stack. Default is " + rwOptions.ifApplyProcessing)
            .longOpt("defaultprocessing").build();

    Option deviceOption = Option.builder("d").argName("device").hasArg()
            .desc("Select CUDA device. Default is " + rwOptions.device).type(Integer.class)
            .longOpt("device").build();

    Option imageOption = Option.builder("i").argName("image").hasArg().required()
            .desc("Stack to process").longOpt("image").build();

    Option seedOption = Option.builder("s").argName("seeds").hasArg().required()
            .desc("Seeds as binary image of size of \"image\"").longOpt("seed").build();

    cliOptions = new Options();
    cliOptions.addOption(seedOption);
    cliOptions.addOption(loadIncOption);
    cliOptions.addOption(defProcessOption);
    cliOptions.addOption(deviceOption);
    cliOptions.addOption(saveIncOption);
    cliOptions.addOption(imageOption);
    cliOptions.addOption(verOption);
    cliOptions.addOption(helpOption);

    Options helpOptions = new Options(); // 2nd group of options
    helpOptions.addOption(verOption); // these are repeated here to have showHelp working
    helpOptions.addOption(helpOption);
    try {
      if (parseForHelp(helpOptions, args)) { // check only for selected options
        return; // and finish
      }
      // process all but do not handle those from 2nd group
      CommandLine cmd = parser.parse(cliOptions, args);

      if (cmd.hasOption("device")) {
        rwOptions.device = Integer.parseInt(cmd.getOptionValue("device").trim());
      }
      if (cmd.hasOption('i')) {
        rwOptions.stack = Paths.get(cmd.getOptionValue('i'));
      }
      if (cmd.hasOption('s')) {
        rwOptions.seeds = Paths.get(cmd.getOptionValue('s'));
      }
      if (cmd.hasOption("loadincidence")) {
        rwOptions.ifComputeIncidence = false;
      }
      if (cmd.hasOption("saveincidence")) {
        rwOptions.ifSaveIncidence = true;
      }
      if (cmd.hasOption("defaultprocessing")) {
        rwOptions.ifApplyProcessing = true;
      }

      // run();

    } catch (org.apache.commons.cli.ParseException pe) {
      System.err.println("Parsing failed: " + pe.getMessage());
      showHelp();
      cliErrorStatus = 1; // finish with error
      return;
    } catch (NumberFormatException ne) {
      System.err.println("One of numeric parameters could not be parsed: " + ne.getMessage());
    } catch (Exception e) {
      System.err.println("Other error: " + e.getMessage());
    } finally { // TODO any from cuda, add better handling
      RandomWalkAlgorithm.finish();
    }
  }

  /**
   * Print help string.
   */
  private void showHelp() {
    String[] authors = new String[] { "Till Bretschneider (Till.Bretschneider@warwick.ac.uk)",
        "Piotr Baniukiewicz (P.Baniukiewicz@warwick.ac.uk)" };
    ToolVersion tv =
            new ToolVersion("/" + this.getClass().getPackage().getName().replaceAll("\\.", "/")
                    + "/jcurandomwalk.properties");
    HelpFormatter formatter = new HelpFormatter();
    String header = "\nRandom Walk segemntaion on GPU.\n";
    header = header.concat(tv.getToolversion(authors));
    String footer = "\n\n";
    formatter.printHelp("JcuRandomWalk", header, cliOptions, footer, true);
  }

  /**
   * @throws Exception
   * 
   */
  public void run() throws Exception {
    selectGpu();
    StopWatch timer = StopWatch.createStarted();
    ImageStack stack = IJ.openImage(rwOptions.stack.toString()).getImageStack();
    ImageStack seed = IJ.openImage(rwOptions.seeds.toString()).getImageStack();
    timer.stop();
    LOGGER.info("Stacks loaded in " + timer.toString());
    // create main object
    timer = StopWatch.createStarted();
    RandomWalkAlgorithm rwa = new RandomWalkAlgorithm(stack, rwOptions);
    // compute or load incidence
    rwa.computeIncidence(rwOptions.ifComputeIncidence);
    if (rwOptions.ifApplyProcessing) {
      rwa.processStack();
    }

    ImageStack segmented = rwa.solve(seed);
    ImagePlus tmp = new ImagePlus("", segmented);
    IJ.saveAsTiff(tmp, "/tmp/solution.tif");

    // TODO finish
    timer.stop();
    LOGGER.info("Solved in " + timer.toString());
    rwa.free();
  }

  /**
   * Pick GPU selected in {@link RandomWalkOptions}.
   */
  private void selectGpu() {
    try {
      if (rwOptions.useGPU) {
        RandomWalkAlgorithm.initilizeGpu();
        int[] devicecount = new int[1];
        cudaGetDeviceCount(devicecount);
        cudaSetDevice(rwOptions.device);
        RandomWalkAlgorithm.initilizeGpu();
        LOGGER.info(String.format("Using device %d/%d", rwOptions.device, devicecount[0]));
      }
    } catch (UnsatisfiedLinkError | NoClassDefFoundError e) {
      LOGGER.error(e.getMessage());
    }
  }

  /**
   * Parse for selected options only, exclusive to other group.
   * 
   * @param options options to look for
   * @param args cli arguments
   * @return true if options parsed successfully, false otherwise.
   */
  private boolean parseForHelp(Options options, String[] args) {
    CommandLineParser parser = new DefaultParser();
    CommandLine cmd = null;
    try {
      cmd = parser.parse(options, args);
    } catch (ParseException e) {
      return false;
    }
    if (cmd.hasOption("help")) { // show help and finish
      showHelp();
      return true;
    }
    if (cmd.hasOption('v')) {
      ToolVersionStruct ver =
              new ToolVersion("/" + this.getClass().getPackage().getName().replaceAll("\\.", "/")
                      + "/jcurandomwalk.properties").getQuimPBuildInfo();
      System.out.println("Version: " + ver.getVersion());
      return true;
    }
    return false;
  }

  /**
   * Main entry point.
   * 
   * <p>See cmd help for options.
   * 
   * @param args cmd args
   */
  public static void main(String[] args) {
    JcuRandomWalk app = new JcuRandomWalk(args);
    System.exit(app.cliErrorStatus);
  }

}
