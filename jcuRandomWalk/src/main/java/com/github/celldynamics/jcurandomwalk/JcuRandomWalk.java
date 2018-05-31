package com.github.celldynamics.jcurandomwalk;

import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaSetDevice;

import java.nio.file.Paths;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.lang3.time.StopWatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.versioning.ToolVersion;
import com.github.celldynamics.versioning.ToolVersionStruct;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.encoder.PatternLayoutEncoder;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.ConsoleAppender;
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

    Option deviceOption = Option.builder().argName("device").hasArg()
            .desc("Select CUDA device. Default is " + rwOptions.device).type(Integer.class)
            .longOpt("device").build();

    Option imageOption = Option.builder("i").argName("image").hasArg().required()
            .desc("Stack to process").longOpt("image").build();
    Option seedOption = Option.builder("s").argName("seeds").hasArg().required()
            .desc("Seeds as binary image of size of \"image\"").longOpt("seed").build();
    Option outputOption = Option.builder("o").argName("output").hasArg().required()
            .desc("Output image").longOpt("output").build();

    Option quietOption = Option.builder("q").desc("Mute output").longOpt("quiet").build();
    Option debugOption = Option.builder("d").desc("Debug stream").longOpt("debug").build();
    Option ddebugOption =
            Option.builder("dd").desc("Even more debug streams").longOpt("superdebug").build();
    OptionGroup qd = new OptionGroup();
    qd.addOption(debugOption);
    qd.addOption(quietOption);
    qd.addOption(ddebugOption);

    // alg options
    Option maxitOption = Option.builder().argName("iter").hasArg()
            .desc("Maximum number of iterations. Default is " + rwOptions.getAlgOptions().maxit)
            .longOpt("maxit").build();
    Option tolOption = Option.builder().argName("tol").hasArg()
            .desc("Tolerance. Default is " + rwOptions.getAlgOptions().tol).longOpt("tol").build();
    Option sigmaGradOption = Option.builder().argName("num").hasArg()
            .desc("sigmaGrad. Default is " + rwOptions.getAlgOptions().sigmaGrad)
            .longOpt("sigmaGrad").build();
    Option sigmaMeanOption = Option.builder().argName("num").hasArg()
            .desc("sigmaMean. Default is " + rwOptions.getAlgOptions().sigmaMean)
            .longOpt("sigmaMean").build();
    Option meanSourceOption = Option.builder().argName("num").hasArg()
            .desc("meanSource. Default is " + rwOptions.getAlgOptions().meanSource)
            .longOpt("meanSource").build();

    cliOptions = new Options();
    cliOptions.addOption(seedOption);
    cliOptions.addOption(loadIncOption);
    cliOptions.addOption(defProcessOption);
    cliOptions.addOption(deviceOption);
    cliOptions.addOption(saveIncOption);
    cliOptions.addOption(imageOption);
    cliOptions.addOption(verOption);
    cliOptions.addOption(helpOption);
    cliOptions.addOption(outputOption);
    cliOptions.addOptionGroup(qd);

    cliOptions.addOption(maxitOption);
    cliOptions.addOption(tolOption);
    cliOptions.addOption(sigmaGradOption);
    cliOptions.addOption(sigmaMeanOption);
    cliOptions.addOption(meanSourceOption);

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
      if (cmd.hasOption('o')) {
        rwOptions.output = Paths.get(cmd.getOptionValue('o'));
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
      if (cmd.hasOption("q")) {
        rwOptions.debugLevel = Level.WARN;
      }
      if (cmd.hasOption("d")) {
        rwOptions.debugLevel = Level.DEBUG;
      }
      if (cmd.hasOption("dd")) {
        rwOptions.debugLevel = Level.TRACE;
      }

      setLogging();
      run();

    } catch (org.apache.commons.cli.ParseException pe) {
      System.err.println("Parsing failed: " + pe.getMessage());
      showHelp();
      cliErrorStatus = 1; // finish with error
      return;
    } catch (NumberFormatException ne) {
      System.err.println("One of numeric parameters could not be parsed: " + ne.getMessage());
    } catch (Exception e) {
      System.err.println("Program failed with exception: " + e.getClass() + " : " + e.getMessage());
      if (rwOptions.debugLevel.toInt() < Level.INFO_INT) {
        e.printStackTrace();
      }
    } finally {
      RandomWalkAlgorithm.finish();
    }
  }

  /**
   * Set debug level for com.github.celldynamics.jcurandomwalk.
   */
  private void setLogging() {
    LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();
    PatternLayoutEncoder logEncoder = new PatternLayoutEncoder();
    logEncoder.setContext(loggerContext);
    logEncoder.setPattern("%highlight([%-5level]) %gray(%-25logger{0}) - %msg%n");
    logEncoder.start();
    ConsoleAppender<ILoggingEvent> logConsoleAppender = new ConsoleAppender<ILoggingEvent>();
    logConsoleAppender.setContext(loggerContext);
    logConsoleAppender.setName("stdout");
    logConsoleAppender.setEncoder(logEncoder);
    logConsoleAppender.setTarget("System.out");
    logConsoleAppender.start();

    Logger rootLogger = loggerContext.getLogger("com.github.celldynamics");
    ((ch.qos.logback.classic.Logger) rootLogger).setLevel(rwOptions.debugLevel);
    if (rwOptions.debugLevel.toInt() >= Level.INFO_INT) {
      ((ch.qos.logback.classic.Logger) rootLogger).setAdditive(false);
      ((ch.qos.logback.classic.Logger) rootLogger).addAppender(logConsoleAppender);
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
   * Run segmentation.
   * 
   * @throws Exception
   * 
   */
  public void run() throws Exception {
    final int seedVal = 255; // value of seed to look for, TODO multi seed option
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

    ImageStack segmented = rwa.solve(seed, seedVal);
    ImagePlus segmentedImage = new ImagePlus("", segmented);
    IJ.saveAsTiff(segmentedImage, rwOptions.output.toString());
    timer.stop();
    LOGGER.info("Solved in " + timer.toString());
    rwa.free();
  }

  /**
   * Pick GPU selected in {@link RandomWalkOptions}.
   */
  private void selectGpu() {
    // try {
    if (rwOptions.useGPU) {
      RandomWalkAlgorithm.initilizeGpu();
      int[] devicecount = new int[1];
      cudaGetDeviceCount(devicecount);
      cudaSetDevice(rwOptions.device);
      RandomWalkAlgorithm.initilizeGpu();
      LOGGER.info(String.format("Using device %d/%d", rwOptions.device, devicecount[0]));
    }
    // } catch (UnsatisfiedLinkError | NoClassDefFoundError e) {
    // LOGGER.error(e.getMessage());
    // }
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
