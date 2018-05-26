package com.github.celldynamics.jcurandomwalk;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.contains;

import java.io.File;
import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixDevice;
import com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixHost;
import com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixType;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.StackStatistics;

/**
 * @author p.baniukiewicz
 *
 */
public class RandomWalkAlgorithmTest {

  static final Logger LOGGER = LoggerFactory.getLogger(RandomWalkAlgorithmTest.class.getName());

  /**
   * The tmpdir.
   */
  static String tmpdir = System.getProperty("java.io.tmpdir") + File.separator;

  @Rule
  public TemporaryFolder folder = new TemporaryFolder();
  @Rule
  public MockitoRule mockitoRule = MockitoJUnit.rule();

  // @Mock
  // IncidenceMatrixGenerator img;
  // @InjectMocks
  // private RandomWalkAlgorithm objMocked;

  static {
    LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();
    Logger rootLogger = loggerContext.getLogger(RandomWalkAlgorithmTest.class.getName());
    ((ch.qos.logback.classic.Logger) rootLogger).setLevel(Level.DEBUG);
  }

  // dimensions of test stack
  private int width = 3;
  private int height = 4;
  private int nz = 2;
  private ImageStack stack;

  private TestDataGenerators tdg;

  /**
   * @throws java.lang.Exception
   */
  @Before
  public void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    stack = TestDataGenerators.getTestStack(width, height, nz, "double");
    tdg = new TestDataGenerators();
  }

  /**
   * @throws java.lang.Exception
   */
  @After
  public void tearDown() throws Exception {
  }

  /**
   * Enable exceptions.
   */
  @BeforeClass
  public static void before() {
    try {
      RandomWalkAlgorithm.initilizeGpu();
    } catch (UnsatisfiedLinkError | NoClassDefFoundError e) {
      LOGGER.error(e.getMessage());
    }
  }

  /**
   * Disable exceptions.
   */
  @AfterClass
  public static void after() {
    try {
      RandomWalkAlgorithm.finish();
    } catch (UnsatisfiedLinkError | NoClassDefFoundError e) {
      LOGGER.error(e.getMessage());
    }
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.RandomWalkAlgorithm#computeIncidence(boolean)}.
   * 
   * @throws Exception
   */
  @Test
  public void testComputeIncidence() throws Exception {
    RandomWalkOptions options = new RandomWalkOptions();
    options.configFolder = folder.newFolder().toPath();
    RandomWalkAlgorithm obj = new RandomWalkAlgorithm(stack, options);
    obj.computeIncidence(false);
    assertThat(options.configFolder.resolve("incidence_stack[3x4x2].ser").toFile().exists(),
            is(true));
  }

  /**
   * Test of {@link RandomWalkAlgorithm#computeLaplacian()}.
   * 
   * <p>Compute laplacean for {@link TestDataGenerators#getTestStack(int, int, int, String)} and
   * mocked weights.
   * 
   * @throws Exception
   */
  @Test
  public void testComputeLaplacian_1() throws Exception {
    RandomWalkOptions options = new RandomWalkOptions();
    options.configFolder = folder.newFolder().toPath();
    // mocked IncidenceMatrixGenerator that return fixed weights
    IncidenceMatrixGenerator img = Mockito.spy(Mockito.spy(new IncidenceMatrixGenerator(stack)));
    // return 2.0 for each weight
    Mockito.doReturn(2.0).when(img).computeWeight(Mockito.any(ImageStack.class),
            Mockito.any(int[].class), Mockito.any(int[].class), Mockito.anyDouble(),
            Mockito.anyDouble(), Mockito.anyDouble());
    img.computeIncidence(); // repeated to use mocked (first is in constructor)

    // final object
    RandomWalkAlgorithm obj = Mockito.spy(new RandomWalkAlgorithm(stack, options));
    // assign mocked generator
    obj.img = img;
    ISparseMatrix lap = obj.computeLaplacian();
    // A' [24 46]
    // W [46 46]
    // A [46 24]
    // L = A'*W*A [24 24]
    assertThat(lap.getRowNumber(), is(24));
    assertThat(lap.getColNumber(), is(24));
    LOGGER.debug("Incidence:" + obj.img.getIncidence().toString());
    LOGGER.debug("Weights:" + obj.img.getWeights().toString());
    SparseMatrixDevice lapcoo = (SparseMatrixDevice) lap.convert2coo();
    lapcoo.retrieveFromDevice();
    // compare with jcuRandomWalk/JCudaMatrix/Matlab/tests.java
    LOGGER.debug("Laplacean" + lapcoo.toString());
    LOGGER.debug(ArrayTools.printArray(ArrayTools.array2Object(lapcoo.full())));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.RandomWalkAlgorithm#processStack()}.
   * 
   * <p>Process stack, save it to tmp and check if minmax is in range 0-1
   * 
   * @throws Exception
   */
  @Test
  public void testProcessStack() throws Exception {
    ImagePlus test_stack = IJ.openImage("src/test/test_data/Stack_cut.tif");
    RandomWalkOptions options = new RandomWalkOptions();
    RandomWalkAlgorithm obj = new RandomWalkAlgorithm(test_stack.getImageStack(), options);
    obj.processStack();
    IJ.saveAsTiff(new ImagePlus("", obj.stack), tmpdir + "testProcessStack.tiff");
    StackStatistics st = new StackStatistics(new ImagePlus("", obj.stack));
    assertThat(st.min, closeTo(0.0, 1e-8));
    assertThat(st.max, closeTo(1.0, 1e-8));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.RandomWalkAlgorithm#getSourceIndices(ij.ImageStack)}.
   * 
   * <p>Output indexes are column-ordered
   * 
   * @throws Exception
   */
  @Test
  public void testGetSourceIndices() throws Exception {
    ImageStack seed = TestDataGenerators.getSeedStack_1();
    RandomWalkAlgorithm obj = new RandomWalkAlgorithm();
    int[] ret = obj.getSourceIndices(seed);
    assertThat(Arrays.asList(ret), contains(new int[] { 0, 2, 13, 30, 47, 49, 59 }));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.RandomWalkAlgorithm#computeReducedLaplacian(com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrix, int[], int[])}.
   * 
   * @throws Exception
   */
  @Test
  public void testComputeReducedLaplacian() throws Exception {
    // Laplacian is square, assume diagonal only
    int[] rI = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] cI = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    double[] v = new double[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };
    ISparseMatrix testL = new SparseMatrixHost(rI, cI, v, SparseMatrixType.MATRIX_FORMAT_COO);
    LOGGER.debug("Laplacean" + testL.toString());
    RandomWalkAlgorithm obj = new RandomWalkAlgorithm();

    // remove row/co 1,2,3
    int[] source = new int[] { 1, 3 };
    int[] sink = new int[] { 1, 2 };

    ISparseMatrix ret = obj.computeReducedLaplacian(testL, source, sink).convert2coo();
    LOGGER.debug("Reduced" + ret.toString());
    assertThat(ret.getColNumber(), is(3));
    assertThat(ret.getRowNumber(), is(3));
    assertThat(ret.getElementNumber(), is(5));
    assertThat(Arrays.asList(ret.getVal()),
            contains(new double[] { 10.0, 102.0, 131.0, 14.0, 15.0 }));

  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.RandomWalkAlgorithm#mergeSeeds(int[], int[])}.
   * 
   * @throws Exception
   */
  @Test
  public void testMergeSeeds() throws Exception {
    RandomWalkAlgorithm obj = new RandomWalkAlgorithm();
    int[] a1 = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    int[] a2 = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    int[] ret = obj.mergeSeeds(a1, a2);
    LOGGER.debug("CS: " + ArrayUtils.toString(ret));

  }

}
