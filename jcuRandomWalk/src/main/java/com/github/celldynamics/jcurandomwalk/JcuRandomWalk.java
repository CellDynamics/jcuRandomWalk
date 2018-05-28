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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.versioning.ToolVersion;
import com.github.celldynamics.versioning.ToolVersionStruct;

import ij.IJ;
import ij.ImageStack;

/**
 * Main application class.
 */
public class JcuRandomWalk {
  static final Logger LOGGER = LoggerFactory.getLogger(JcuRandomWalk.class.getName());
  private int cliErrorStatus = 0; // returned to main
  private RandomWalkOptions rwo;
  private Options options = null;

  /**
   * Default constructor using default options.
   */
  public JcuRandomWalk() {
    rwo = new RandomWalkOptions();
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

    Option saveIncOption = Option.builder().desc("Save incdence matrix for this size of the stack")
            .longOpt("saveincidence").build();

    Option loadIncOption = Option.builder()
            .desc("Load incidence matrix for this size of stack or compute new one and save it"
                    + " if relevant file has not been found.")
            .longOpt("loadincidence").build();

    Option defProcessOption = Option.builder().desc("Apply default processing to stack.")
            .longOpt("defaultprocessing").build();

    Option deviceOption = Option.builder("d").argName("device").hasArg().desc("Select CUDA device")
            .type(Integer.class).longOpt("device").build();

    Option imageOption = Option.builder("i").argName("image").hasArg().required()
            .desc("Stack to process").longOpt("image").build();

    Option seedOption = Option.builder("s").argName("seeds").hasArg().required()
            .desc("Seeds as binary image of size of \"image\"").longOpt("seed").build();

    options = new Options();
    options.addOption(seedOption);
    options.addOption(loadIncOption);
    options.addOption(defProcessOption);
    options.addOption(deviceOption);
    options.addOption(saveIncOption);
    options.addOption(imageOption);
    options.addOption(verOption);
    options.addOption(helpOption);

    Options helpOptions = new Options(); // 2nd group of options
    helpOptions.addOption(verOption); // these are repeated here to have showHelp working
    helpOptions.addOption(helpOption);
    try {
      if (parseForHelp(helpOptions, args)) { // check only for selected options
        return; // and finish
      }
      // process all but do not handle those from 2nd group
      CommandLine cmd = parser.parse(options, args);

      if (cmd.hasOption("device")) {
        rwo.device = Integer.parseInt(cmd.getOptionValue("device").trim());
      }
      if (cmd.hasOption('i')) {
        rwo.stack = Paths.get(cmd.getOptionValue('i'));
      }
      if (cmd.hasOption('s')) {
        rwo.seeds = Paths.get(cmd.getOptionValue('s'));
      }
      selectGpu(); // initilaise gpu (if selected)

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
    formatter.printHelp("JcuRandomWalk", header, options, footer, true);
  }

  /**
   * 
   */
  public void run() {
    selectGpu();
    ImageStack stack = IJ.openImage(rwo.stack.toString()).getImageStack();
  }

  private void selectGpu() {
    if (rwo.useGPU) {
      RandomWalkAlgorithm.initilizeGpu();
      int[] devicecount = new int[1];
      cudaGetDeviceCount(devicecount);
      cudaSetDevice(rwo.device);
      LOGGER.info(String.format("Using device %d/%d", rwo.device, devicecount[0]));
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
