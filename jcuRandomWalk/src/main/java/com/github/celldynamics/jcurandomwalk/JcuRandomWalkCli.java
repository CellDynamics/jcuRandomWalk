package com.github.celldynamics.jcurandomwalk;

import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaSetDevice;

import java.awt.Window;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.CountDownLatch;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.io.FilenameUtils;
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
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;

/**
 * Main application class. CLI frontend.
 * 
 * @author p.baniukiewicz
 * @author t.bretschneider
 */
public class JcuRandomWalkCli {

  ImageJ ij;
  static final Logger LOGGER = LoggerFactory.getLogger(JcuRandomWalkCli.class.getName());
  private int cliErrorStatus = 0; // returned to main
  private RandomWalkOptions rwOptions;
  private Options cliOptions = null;
  private CommandLine cmd;
  private ImageStack seed;
  private ImageStack stack;

  /**
   * Default constructor using default options.
   */
  public JcuRandomWalkCli() {
    rwOptions = new RandomWalkOptions();
  }

  /**
   * Constructor - parser initialised by cli. Main runner
   * 
   * @param args args passed from cli.
   */
  public JcuRandomWalkCli(String[] args) {
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
    OptionGroup gl = new OptionGroup();
    gl.addOption(saveIncOption);
    gl.addOption(loadIncOption);

    Option defProcessOption = Option.builder()
            .desc("Apply default processing to stack. Default is " + rwOptions.ifApplyProcessing)
            .longOpt("defaultprocessing").build();
    Option autoThOption = Option.builder("t").argName("level").hasArg().type(Integer.class)
            .desc("Apply thresholding to stack and produce seeds.").longOpt("autoth").build();

    Option deviceOption = Option.builder().argName("device").hasArg()
            .desc("Select CUDA device. Default is " + rwOptions.device).type(Integer.class)
            .longOpt("device").build();
    Option cpuOption = Option.builder().desc("Use CPU only. Default is " + rwOptions.cpuOnly)
            .longOpt("cpuonly").build();

    Option imageOption = Option.builder("i").argName("input").hasArg().required()
            .desc("Stack to process. Must be in range 0-1.").longOpt("image").build();
    Option seedOption =
            Option.builder("s").argName("seeds").hasArg()
                    .desc("Seeds as binary image of size of \"input\". Default is \"input\""
                            + rwOptions.seedSuffix + " (if there is no -t option)")
                    .longOpt("seed").build();
    Option outputOption = Option.builder("o").argName("output").hasArg()
            .desc("Output image name. Default is input" + rwOptions.outSuffix).longOpt("output")
            .build();

    Option rawProbOption =
            Option.builder().desc("Save raw probability maps. Default is " + rwOptions.rawProbMaps)
                    .longOpt("probmaps").build();

    Option quietOption = Option.builder("q").desc("Mute output").longOpt("quiet").build();
    Option debugOption = Option.builder("d").desc("Debug stream").longOpt("debug").build();
    Option ddebugOption =
            Option.builder("dd").desc("Even more debug streams").longOpt("superdebug").build();
    OptionGroup qd = new OptionGroup();
    qd.addOption(debugOption);
    qd.addOption(quietOption);
    qd.addOption(ddebugOption);
    OptionGroup seeds = new OptionGroup();
    seeds.addOption(seedOption);
    seeds.addOption(autoThOption);

    Option showOption = Option.builder().desc("Show resulting image").longOpt("show").build();

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
    // cliOptions.addOption(seedOption);
    cliOptions.addOptionGroup(seeds);
    cliOptions.addOptionGroup(gl);
    cliOptions.addOption(defProcessOption);
    // cliOptions.addOption(autoThOption);
    cliOptions.addOption(deviceOption);
    cliOptions.addOption(cpuOption);
    cliOptions.addOption(imageOption);
    cliOptions.addOption(verOption);
    cliOptions.addOption(helpOption);
    cliOptions.addOption(outputOption);
    cliOptions.addOptionGroup(qd);
    cliOptions.addOption(showOption);

    cliOptions.addOption(maxitOption);
    cliOptions.addOption(tolOption);
    cliOptions.addOption(sigmaGradOption);
    cliOptions.addOption(sigmaMeanOption);
    cliOptions.addOption(meanSourceOption);
    cliOptions.addOption(rawProbOption);

    Options helpOptions = new Options(); // 2nd group of options
    helpOptions.addOption(verOption); // these are repeated here to have showHelp working
    helpOptions.addOption(helpOption);
    try {
      if (parseForHelp(helpOptions, args)) { // check only for selected options
        rwOptions.cpuOnly = true; // fake overriding to comply with finally
        return; // and finish
      }
      // process all but do not handle those from 2nd group
      cmd = parser.parse(cliOptions, args);

      if (cmd.hasOption("device")) {
        rwOptions.device = Integer.parseInt(cmd.getOptionValue("device").trim());
      }
      if (cmd.hasOption("cpuonly")) {
        rwOptions.cpuOnly = true;
      }
      if (cmd.hasOption('i')) {
        rwOptions.stack = Paths.get(cmd.getOptionValue('i'));
        loadImageAction();
      }
      if (cmd.hasOption('s')) {
        rwOptions.seeds = Paths.get(cmd.getOptionValue('s'));
        loadSeedAction();
      } else if (!cmd.hasOption("t")) { // use default seed source if no -t option
        rwOptions.seeds = Paths.get(
                FilenameUtils.removeExtension(rwOptions.stack.toString()) + rwOptions.seedSuffix);
        LOGGER.warn("No -s option specified, assuming " + rwOptions.seeds.toString());
        loadSeedAction();
      }
      if (cmd.hasOption('o')) {
        rwOptions.output = Paths.get(cmd.getOptionValue('o'));
      } else { // default output
        rwOptions.output = Paths.get(
                FilenameUtils.removeExtension(rwOptions.stack.toString()) + rwOptions.outSuffix);
        LOGGER.warn("No -o option specified, assuming " + rwOptions.output.toString());
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
      if (cmd.hasOption("probmaps")) {
        rwOptions.rawProbMaps = true;
      }
      if (cmd.hasOption("t")) {
        rwOptions.thLevel = Double.parseDouble(cmd.getOptionValue("t").trim());
        thresholdSeedAction();
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
      cliErrorStatus = 1; // finish with error
    } catch (Exception e) {
      System.err.println("Program failed with exception: " + e.getClass() + " : " + e.getMessage());
      if (rwOptions.debugLevel.toInt() < Level.INFO_INT) {
        e.printStackTrace();
      }
      cliErrorStatus = 1; // finish with error
    } finally { // GPU clean up
      if (rwOptions.cpuOnly == false) {
        RandomWalkAlgorithmGpu.finish();
      }
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
    IRandomWalkSolver rwa = null;
    final int seedVal = 255; // value of seed to look for, TODO multi seed option
    StopWatch timer = StopWatch.createStarted();
    if (rwOptions.cpuOnly == false) {
      deviceAction();
      // create main object
      rwa = new RandomWalkAlgorithmGpu(stack, rwOptions);
    } else {
      rwa = new RandomWalkAlgorithmOj(stack, rwOptions);
    }
    // compute or load incidence or save, depending on options
    if (rwOptions.ifApplyProcessing) {
      applyProcessingAction(rwa);
    }
    ImageStack segmented = rwa.solve(seed, seedVal);
    ImagePlus segmentedImage = new ImagePlus("", segmented);
    if (rwOptions.output.getParent() == null
            || !rwOptions.output.getParent().toFile().isDirectory()) { // overcome IJ.savetiff
      throw new IOException("Output folder does not exist");
    }
    IJ.saveAsTiff(segmentedImage, rwOptions.output.toString());
    LOGGER.info("File " + rwOptions.output.toString() + " saved");
    timer.stop();
    LOGGER.info("Solved in " + timer.toString());
    rwa.free();
    // actions
    if (cmd != null && cmd.hasOption("show")) {
      showResultAction(segmentedImage);
    }
    if (rwOptions.rawProbMaps) {
      rawProbMapsAction(rwa);
    }
  }

  /**
   * Prepare seeds of -t option used.
   */
  private void thresholdSeedAction() {
    StopWatch timer = StopWatch.createStarted();
    seed = ImageStack.create(stack.getWidth(), stack.getHeight(), stack.getSize(), 8);
    for (int x = 0; x < seed.getWidth(); x++) {
      for (int y = 0; y < seed.getHeight(); y++) {
        for (int z = 0; z < seed.size(); z++) {
          if (seed.getVoxel(x, y, z) <= rwOptions.thLevel) {
            seed.setVoxel(x, y, z, 0);
          } else {
            seed.setVoxel(x, y, z, 255);
          }
        }
      }
    }
    LOGGER.info("Seeds created in " + timer.toString());
  }

  /**
   * Action for -s option.
   * 
   * @throws IOException if file not found
   */
  private void loadSeedAction() throws IOException {
    StopWatch timer = StopWatch.createStarted();
    if (!rwOptions.seeds.toFile().exists()) {
      throw new IOException("Seeds file not found");
    } else {
      seed = IJ.openImage(rwOptions.seeds.toString()).getImageStack();
    }
    timer.stop();
    LOGGER.info("Seeds loaded in " + timer.toString());
  }

  /**
   * Action for -s option.
   * 
   * @throws IOException if file not found
   */
  private void loadImageAction() throws IOException {
    StopWatch timer = StopWatch.createStarted();
    if (!rwOptions.stack.toFile().exists()) {
      throw new IOException("Stack file not found");
    } else {
      stack = IJ.openImage(rwOptions.stack.toString()).getImageStack();
      timer.stop();
      LOGGER.info("image loaded in " + timer.toString());
    }
  }

  /**
   * Action of apply processing to stack.
   * 
   * @param rwa instance of solver.
   */
  private void applyProcessingAction(IRandomWalkSolver rwa) {
    rwa.processStack();
  }

  /**
   * Action for show result.
   * 
   * @param segmentedImage result to show.
   */
  private void showResultAction(ImagePlus segmentedImage) {
    ij = new ImageJ();
    segmentedImage.show();
  }

  /**
   * Action for rawProbMap.
   * 
   * @param rwa instance of solver.
   * @throws IOException any error
   */
  private void rawProbMapsAction(IRandomWalkSolver rwa) throws IOException {
    LOGGER.info("Saving probability maps");
    List<ImageStack> maps = rwa.getRawProbs();
    Path parent = rwOptions.output.getParent();
    if (parent == null) {
      parent = Paths.get("/");
    }
    if (!parent.toFile().isDirectory()) { // overcome IJ.savetiff
      throw new IOException("Output folder does not exist");
    }
    Path name = rwOptions.output.getFileName();
    int i = 1;
    for (ImageStack is : maps) {
      ImagePlus im = new ImagePlus("", is);
      String nameToSave = parent.resolve(name + "Map_" + i + ".tif").toString();
      IJ.saveAsTiff(im, nameToSave);
      LOGGER.info("Map " + nameToSave + " saved");
      i++;
    }
  }

  /**
   * Pick GPU selected in {@link RandomWalkOptions}.
   */
  private void deviceAction() {
    RandomWalkAlgorithmGpu.initilizeGpu();
    int[] devicecount = new int[1];
    cudaGetDeviceCount(devicecount);
    cudaSetDevice(rwOptions.device);
    LOGGER.info(String.format("Using device %d/%d", rwOptions.device, devicecount[0]));
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
    StopWatch timer = StopWatch.createStarted();
    CountDownLatch startSignal = new CountDownLatch(1);
    JcuRandomWalkCli app = new JcuRandomWalkCli(args);
    if (app.ij != null) {
      app.ij.addWindowListener(new WindowAdapter() {

        @Override
        public void windowClosing(WindowEvent e) {
          Window[] windows = Window.getWindows();
          for (Window w : windows) {
            w.dispose();
          }
          startSignal.countDown();
        }
      });
      try {
        LOGGER.info("Waiting for IJ to close.");
        startSignal.await();
      } catch (InterruptedException e1) {
        e1.printStackTrace();
      }
    }
    if (app.cliErrorStatus == 0) {
      timer.stop();
      LOGGER.info("Bye! Total time spent: " + timer.toString());
    }
    System.exit(app.cliErrorStatus);
  }

}
