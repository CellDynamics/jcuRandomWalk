package com.github.celldynamics.jcurandomwalk;

import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaSetDevice;

import java.awt.Window;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.stream.Collectors;
import java.util.stream.Stream;

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
  // private boolean folderMode = false; // true if -i points to folder

  /**
   * Default constructor using default options.
   */
  public JcuRandomWalkCli() {
    rwOptions = new RandomWalkOptions();
  }

  /**
   * Constructor - parser initialized by cli. Main runner
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

    Option defProcessOption = Option.builder().argName("gamma").hasArg().type(Double.class)
            .desc("Apply default processing to stack. Input stack should be 8-bit. Default is "
                    + rwOptions.ifApplyProcessing)
            .longOpt("defaultprocessing").build();
    Option autoThOption = Option.builder("t").argName("level").hasArg().type(Double.class)
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
            .desc("Output image name or folder where image will be stored. Default is input"
                    + rwOptions.outSuffix)
            .longOpt("output").build();

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
    Option maxitOption = Option.builder().argName("iter").hasArg().type(Integer.class)
            .desc("Maximum number of iterations. Default is " + rwOptions.getAlgOptions().maxit)
            .longOpt("maxit").build();
    Option tolOption = Option.builder().argName("tol").hasArg().type(Double.class)
            .desc("Tolerance. Default is " + rwOptions.getAlgOptions().tol).longOpt("tol").build();
    Option sigmaGradOption = Option.builder().argName("num").hasArg().type(Double.class)
            .desc("sigmaGrad. Default is " + rwOptions.getAlgOptions().sigmaGrad)
            .longOpt("sigmaGrad").build();
    Option sigmaMeanOption = Option.builder().argName("num").hasArg().type(Double.class)
            .desc("sigmaMean. Default is "
                    + (rwOptions.getAlgOptions().sigmaMean == null ? "computed"
                            : rwOptions.getAlgOptions().sigmaMean))
            .longOpt("sigmaMean").build();
    Option meanSourceOption = Option.builder().argName("num").hasArg().type(Double.class)
            .desc("meanSource. Default is "
                    + (rwOptions.getAlgOptions().meanSource == null ? "computed"
                            : rwOptions.getAlgOptions().meanSource))
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
      }
      if (cmd.hasOption('s')) {
        rwOptions.seeds = Paths.get(cmd.getOptionValue('s'));
      } else {
        rwOptions.seeds = Paths.get(
                FilenameUtils.removeExtension(rwOptions.stack.toString()) + rwOptions.seedSuffix);
        LOGGER.warn("No -s option specified, assuming "
                + (cmd.hasOption("t") ? "generating seeds from input (-t)"
                        : rwOptions.seeds.toString()));
      }
      if (cmd.hasOption('o')) {
        rwOptions.output = Paths.get(cmd.getOptionValue('o'));
      } else { // default output
        rwOptions.output = Paths.get(
                FilenameUtils.removeExtension(rwOptions.stack.toString()) + rwOptions.outSuffix);
        LOGGER.warn("No -o option specified, assuming pattern inputFile_" + rwOptions.outSuffix);
      }
      if (cmd.hasOption("loadincidence")) {
        rwOptions.ifComputeIncidence = false;
      }
      if (cmd.hasOption("saveincidence")) {
        rwOptions.ifSaveIncidence = true;
      }
      if (cmd.hasOption("defaultprocessing")) {
        rwOptions.ifApplyProcessing = true;
        rwOptions.gammaVal = Double.parseDouble(cmd.getOptionValue("defaultprocessing").trim());
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
      }
      if (cmd.hasOption("maxit")) {
        rwOptions.getAlgOptions().maxit = Integer.parseInt(cmd.getOptionValue("maxit").trim());
      }
      if (cmd.hasOption("tol")) {
        rwOptions.getAlgOptions().tol =
                (float) Double.parseDouble(cmd.getOptionValue("tol").trim());
      }
      if (cmd.hasOption("sigmaGrad")) {
        rwOptions.getAlgOptions().sigmaGrad =
                Double.parseDouble(cmd.getOptionValue("sigmaGrad").trim());
      }
      if (cmd.hasOption("sigmaMean")) {
        rwOptions.getAlgOptions().sigmaMean =
                Double.parseDouble(cmd.getOptionValue("sigmaMean").trim());
      }
      if (cmd.hasOption("meanSource")) {
        rwOptions.getAlgOptions().meanSource =
                Double.parseDouble(cmd.getOptionValue("meanSource").trim());
      }

      setLogging();
      runner();

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
   * Distribute jobs in parallel.
   * 
   * @throws Exception
   */
  private void runner() throws Exception {
    ImageStack seed;
    Path outputFolder;
    List<Path> stacks = loadImagesAction();
    // set output paths
    // if no -o option, output is already filled with default name in input folder
    // if only folder given - set default name in this folder
    if (rwOptions.output.toFile().isDirectory()) { // if only folder
      outputFolder = rwOptions.output; // prepare default output in this folder
      rwOptions.output = outputFolder
              .resolve(FilenameUtils.removeExtension(rwOptions.stack.getFileName().toString())
                      + rwOptions.outSuffix);
    } else {
      // output is filled with file name, take its path in case we need to enter in folderMode
      // (file name will be ignored then)
      outputFolder = rwOptions.output.getParent();
    }

    for (Path stackPath : stacks) {
      ImageStack stack = loadImage(stackPath); // load and process stack if option selected
      if (stacks.size() > 1) { // more than one? we need to adapt output to each separatelly
        // for folder mode take only root of path -o and set automatic output name
        // for each image
        rwOptions.output = outputFolder
                .resolve(FilenameUtils.removeExtension(stackPath.getFileName().toString())
                        + rwOptions.outSuffix);
      }
      // if no folder mode, output should contain filename given or automatic (automatic set in
      // handling options)
      if (rwOptions.getAlgOptions().meanSource == null) {
        rwOptions.getAlgOptions().meanSource = new StackPreprocessor().getMean(stack);
      }
      if (rwOptions.thLevel >= 0) { // so -t was used
        seed = thresholdSeedAction(stack); // stack already processed
      } else { // load seed given or default
        seed = loadSeedAction();
      }
      LOGGER.info("Processing file " + stackPath.getFileName().toString());
      run(seed, stack, rwOptions); // TODO Options can be copied and run in parallel
      if (stacks.size() > 1) {
        LOGGER.info(
                "-----------------------------------------------------------------------------");
      }
    }
  }

  /**
   * Iterate over image or folder and return list of them.
   * 
   * <p>Set also folderMode.
   * 
   * @return list of images to process.
   * @throws IOException if wrong folder or file
   */
  private List<Path> loadImagesAction() throws IOException {
    List<Path> ret = new ArrayList<>();
    if (rwOptions.stack.toFile().isDirectory()) {
      File dir = rwOptions.stack.toFile();
      File[] inFolder = dir.listFiles(new FilenameFilter() {

        @Override
        public boolean accept(File dir, String name) {
          return name.endsWith(".tif");
        }
      });
      ret.addAll(Stream.of(inFolder).map(f -> f.toPath()).collect(Collectors.toList()));
      LOGGER.info("Discovered " + ret.size() + " files in " + rwOptions.stack.toString());
      // folderMode = true;
    } else if (rwOptions.stack.toFile().isFile()) {
      ret.add(rwOptions.stack); // just add this one
    } else {
      throw new IOException("Stack file not found");
    }
    return ret;
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
   * @param seed seeds
   * @param stack stack
   * @param rwOptions own copy of options
   * 
   * @throws Exception
   * 
   */
  public void run(ImageStack seed, ImageStack stack, RandomWalkOptions rwOptions) throws Exception {
    LOGGER.trace(rwOptions.toString());
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
    rwa.validateSeeds(seed);
    ImageStack segmented = rwa.solve(seed, seedVal);
    ImagePlus segmentedImage = new ImagePlus("", segmented);
    if (rwOptions.output.getParent() == null
            || !rwOptions.output.getParent().toFile().isDirectory()) { // overcome IJ.savetiff
      throw new IOException("Output folder does not exist");
    }
    IJ.saveAsTiff(segmentedImage, rwOptions.output.toString());
    LOGGER.info("Output " + rwOptions.output.toString() + " saved");
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
   * 
   * @param stack stack to threshold
   * @return seeds
   */
  private ImageStack thresholdSeedAction(ImageStack stack) {
    ImageStack seed;
    StopWatch timer = StopWatch.createStarted();
    seed = ImageStack.create(stack.getWidth(), stack.getHeight(), stack.getSize(), 8);
    for (int x = 0; x < seed.getWidth(); x++) {
      for (int y = 0; y < seed.getHeight(); y++) {
        for (int z = 0; z < seed.size(); z++) {
          if (stack.getVoxel(x, y, z) <= rwOptions.thLevel) {
            seed.setVoxel(x, y, z, 0);
          } else {
            seed.setVoxel(x, y, z, 255);
          }
        }
      }
    }
    LOGGER.info("Seeds created in " + timer.toString());
    if (rwOptions.debugLevel.toInt() < Level.INFO_INT) {
      ImagePlus tmpImg = new ImagePlus("", seed);
      String tmpdir = System.getProperty("java.io.tmpdir") + File.separator;
      IJ.saveAsTiff(tmpImg, tmpdir + "seeds.tif");
      LOGGER.debug("Saved seeds in " + tmpdir + "seeds.tif");
    }
    return seed;
  }

  /**
   * Action for -s option.
   * 
   * @return loaded seeds
   * 
   * @throws IOException if file not found
   */
  private ImageStack loadSeedAction() throws IOException {
    ImageStack seed;
    StopWatch timer = StopWatch.createStarted();
    if (!rwOptions.seeds.toFile().exists()) {
      throw new IOException("Seeds file not found");
    } else {
      seed = IJ.openImage(rwOptions.seeds.toString()).getImageStack();
    }
    timer.stop();
    LOGGER.info("Seeds loaded in " + timer.toString());
    return seed;
  }

  /**
   * Action for -s option and --defaultprocessing.
   * 
   * @param image Path to image to load
   * 
   * @return loaded stack
   * @throws IOException if file not found
   */
  private ImageStack loadImage(Path image) throws IOException {
    ImageStack stack;
    StopWatch timer = StopWatch.createStarted();
    if (!image.toFile().exists()) {
      throw new IOException("Stack file not found");
    } else {
      stack = IJ.openImage(image.toString()).getImageStack();
      timer.stop();
      LOGGER.info("image " + image.getFileName() + " loaded in " + timer.toString());
      return applyProcessingAction(stack);
    }
  }

  /**
   * Action of apply processing to stack.
   * 
   * @param rwa instance of solver.
   */
  private ImageStack applyProcessingAction(ImageStack stack) {
    if (rwOptions.ifApplyProcessing == false) {
      return stack;
    }
    ImageStack ret = new StackPreprocessor().processStack(stack, rwOptions.gammaVal);
    if (rwOptions.debugLevel.toInt() < Level.INFO_INT) {
      ImagePlus tmpImg = new ImagePlus("", ret);
      String tmpdir = System.getProperty("java.io.tmpdir") + File.separator;
      IJ.saveAsTiff(tmpImg, tmpdir + "object_stack.tif");
      LOGGER.debug("Saved stack in " + tmpdir + "object_stack.tif");
    }
    return ret;
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
