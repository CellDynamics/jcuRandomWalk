package com.github.celldynamics.jcurandomwalk;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsInAnyOrder;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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

import com.github.celldynamics.jcudarandomwalk.matrices.IMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixDevice;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixHost;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixType;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.StackStatistics;

/**
 * The Class RandomWalkAlgorithmTest.
 *
 * @author p.baniukiewicz
 */
public class RandomWalkAlgorithmTest {

  static {
    LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();
    Logger rootLogger = loggerContext.getLogger("com.github.celldynamics.jcurandomwalk");
    ((ch.qos.logback.classic.Logger) rootLogger).setLevel(Level.DEBUG);
  }

  /** The Constant LOGGER. */
  static final Logger LOGGER = LoggerFactory.getLogger(RandomWalkAlgorithmTest.class.getName());

  /**
   * The tmpdir.
   */
  static String tmpdir = System.getProperty("java.io.tmpdir") + File.separator;

  /** The folder. */
  @Rule
  public TemporaryFolder folder = new TemporaryFolder();

  /** The mockito rule. */
  @Rule
  public MockitoRule mockitoRule = MockitoJUnit.rule();

  // @Mock
  // IncidenceMatrixGenerator img;
  // @InjectMocks
  // private RandomWalkAlgorithm objMocked;

  /** The width. */
  // dimensions of test stack
  private int width = 3;

  /** The height. */
  private int height = 4;

  /** The nz. */
  private int nz = 2;

  /** The stack. */
  private ImageStack stack;

  /**
   * Sets the up.
   *
   * @throws Exception the exception
   */
  @Before
  public void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    stack = TestDataGenerators.getTestStack(width, height, nz, "double");
  }

  /**
   * Tear down.
   *
   * @throws Exception the exception
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
   * @throws Exception the exception
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
   * <p>Compute laplacian for {@link TestDataGenerators#getTestStack(int, int, int, String)} and
   * mocked weights.
   *
   * @throws Exception the exception
   */
  @Test
  public void testComputeLaplacian_1() throws Exception {
    RandomWalkOptions options = new RandomWalkOptions();
    options.configFolder = folder.newFolder().toPath();
    // mocked IncidenceMatrixGenerator that return fixed weights
    IncidenceMatrixGenerator img =
            Mockito.spy(Mockito.spy(new IncidenceMatrixGenerator(stack, options.getAlgOptions())));
    // return 2.0 for each weight
    Mockito.doReturn(2.0).when(img).computeWeight(Mockito.any(ImageStack.class),
            Mockito.any(int[].class), Mockito.any(int[].class), Mockito.anyDouble(),
            Mockito.anyDouble(), Mockito.anyDouble());
    img.computeIncidence(); // repeated to use mocked (first is in constructor)

    // final object
    RandomWalkAlgorithm obj = Mockito.spy(new RandomWalkAlgorithm(stack, options));
    // assign mocked generator
    obj.img = img;
    obj.computeLaplacian();
    ISparseMatrix lap = obj.getLap();
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
   * @throws Exception the exception
   */
  @Test
  public void testProcessStack() throws Exception {
    ImagePlus teststack = IJ.openImage("src/test/test_data/Stack_cut.tif");
    RandomWalkOptions options = new RandomWalkOptions();
    RandomWalkAlgorithm obj = new RandomWalkAlgorithm(teststack.getImageStack(), options);
    obj.processStack();
    IJ.saveAsTiff(new ImagePlus("", obj.stack), tmpdir + "testProcessStack.tiff");
    StackStatistics st = new StackStatistics(new ImagePlus("", obj.stack));
    assertThat(st.min, closeTo(0.0, 1e-8));
    assertThat(st.max, closeTo(1.0, 1e-8));
  }

  /**
   * Test method for
   * {@link RandomWalkAlgorithm#getSourceIndices(ImageStack, int)}.
   * 
   * <p>Output indexes are column-ordered
   *
   * @throws Exception the exception
   */
  @Test
  public void testGetSourceIndices() throws Exception {
    ImageStack seed = TestDataGenerators.getSeedStack_1();
    RandomWalkAlgorithm obj = new RandomWalkAlgorithm();
    Integer[] ret = obj.getSourceIndices(seed, 255);
    List<Integer> blist = Arrays.asList(ret);
    boolean issorted = blist.stream().sorted().collect(Collectors.toList()).equals(blist);
    assertThat(issorted, is(true));
    assertThat(blist, containsInAnyOrder(new Integer[] { 0, 2, 13, 30, 47, 49, 59 }));
  }

  /**
   * Test method for
   * {@link RandomWalkAlgorithm#computeReducedLaplacian(Integer[], Integer[])}.
   *
   * @throws Exception the exception
   */
  @Test
  public void testComputeReducedLaplacian() throws Exception {
    // Laplacian is square, assume diagonal only
    int[] ri = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] ci = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    float[] v = new float[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };
    ISparseMatrix testL = new SparseMatrixHost(ri, ci, v, SparseMatrixType.MATRIX_FORMAT_COO);
    LOGGER.debug("Laplacean" + testL.toString());
    RandomWalkAlgorithm obj = new RandomWalkAlgorithm();

    // remove row/co 1,2,3
    Integer[] source = new Integer[] { 1, 3 };
    Integer[] sink = new Integer[] { 1, 2 };
    obj.lap = testL;
    obj.computeReducedLaplacian(source, sink);
    ISparseMatrix ret = obj.reducedLap.convert2coo();
    LOGGER.debug("Reduced" + ret.toString());
    assertThat(ret.getColNumber(), is(3));
    assertThat(ret.getRowNumber(), is(3));
    assertThat(ret.getElementNumber(), is(5));
    assertThat(Arrays.asList(ret.getVal()),
            contains(new float[] { 10.0f, 102.0f, 131.0f, 14.0f, 15.0f }));

  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.RandomWalkAlgorithm#mergeSeeds(int[], int[])}.
   *
   * @throws Exception the exception
   */
  // @Test
  // public void testMergeSeeds() throws Exception {
  // RandomWalkAlgorithm obj = new RandomWalkAlgorithm();
  // int[] a1 = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  // int[] a2 = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
  // int[] ret = obj.mergeSeeds(a1, a2);
  // LOGGER.debug("CS: " + ArrayUtils.toString(ret));
  // assertThat(Arrays.asList(ret), contains(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }));
  // }

  /**
   * Test compute B.
   * 
   * <p>Matrix:
   * 10 101 0 0 0 102
   * 0 11 0 0 0 0
   * 0 0 12 0 0 0
   * 0 0 0 13 0 0
   * 131 0 0 0 14 0
   * 0 0 0 0 0 15
   * 
   * @throws Exception the exception
   */
  @Test
  public void testComputeB() throws Exception {
    RandomWalkAlgorithm obj = new RandomWalkAlgorithm();
    int[] ri = new int[] { 0, 0, 0, 1, 2, 3, 4, 4, 5 };
    int[] ci = new int[] { 0, 1, 5, 1, 2, 3, 0, 4, 5 };
    float[] v = new float[] { 10, 101, 102, 11, 12, 13, 131, 14, 15 };
    ISparseMatrix testL = new SparseMatrixHost(ri, ci, v, SparseMatrixType.MATRIX_FORMAT_COO);
    IMatrix ret = obj.computeB(testL, new Integer[] { 0, 1 });
    assertThat(Arrays.asList(ret.getVal()),
            contains(new float[] { -111.0f, -11.0f, -0f, -0f, -131.0f, -0.0f }));
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.RandomWalkAlgorithm#getSegmentedStack(float[])}.
   * 
   * <p>Check if output stack converted from linear solution has proper orientation - column wise.
   * 
   * @throws Exception
   */
  @Test
  public void testGetSegmentedStack() throws Exception {
    RandomWalkAlgorithm obj = new RandomWalkAlgorithm();
    ImageStack mockStack = Mockito.mock(ImageStack.class);
    obj.stack = mockStack;
    Mockito.doReturn(5).when(mockStack).getWidth();
    Mockito.doReturn(4).when(mockStack).getHeight();
    Mockito.doReturn(2).when(mockStack).getSize();
    List<Float> solution =
            IntStream.range(0, 5 * 4 * 2).mapToObj(x -> new Float(x)).collect(Collectors.toList());
    ImageStack ret = obj.getSegmentedStack(ArrayUtils.toPrimitive(solution.toArray(new Float[0])));
    LOGGER.debug("Segmented stack: " + ret.toString());
    LOGGER.trace("S1: "
            + ArrayTools.printArray(ArrayTools.array2Object(ret.getProcessor(2).getFloatArray())));
    // check column order
    assertThat(ret.getVoxel(0, 0, 0), closeTo(0.0, 1e-6));
    assertThat(ret.getVoxel(0, 1, 0), closeTo(1.0, 1e-6));
    assertThat(ret.getVoxel(1, 0, 0), closeTo(4.0, 1e-6));

    assertThat(ret.getVoxel(0, 0, 1), closeTo(20.0, 1e-6));
    assertThat(ret.getVoxel(0, 1, 1), closeTo(21.0, 1e-6));
    assertThat(ret.getVoxel(1, 0, 1), closeTo(24.0, 1e-6));

  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.RandomWalkAlgorithm#solve(ImageStack, int)}.
   * 
   * @throws Exception
   */
  @Test
  public void testSolve() throws Exception {
    RandomWalkOptions options = new RandomWalkOptions();
    ImageStack org = IJ.openImage("src/test/test_data/segment_test_normalised.tif").getImageStack();
    ImageStack seeds = IJ.openImage("src/test/test_data/segment_test_seeds.tif").getImageStack();
    RandomWalkAlgorithm obj = new RandomWalkAlgorithm(org, options);
    obj.computeIncidence(options.ifComputeIncidence);
    ImageStack segmented = obj.solve(seeds, 255);
    ImagePlus tmp = new ImagePlus("", segmented);
    IJ.saveAsTiff(tmp, "/tmp/solution.tif");
  }

  /**
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.RandomWalkAlgorithm#incorporateSeeds(float[], Integer[], Integer[], int)}.
   * 
   * @throws Exception
   */
  @Test
  public void testIncorporateSeeds() throws Exception {
    // full size is 10 elements
    Integer[] source = new Integer[] { 0, 3, 4 }; // source pixels
    Integer[] sink = new Integer[] { 1, 9 }; // sink pixels
    // 5 pixels were given, 5 were calculated from reduced lap
    float[] solution = new float[] { 10, 11, 12, 13, 14 };
    RandomWalkAlgorithm obj = new RandomWalkAlgorithm();
    float[] ret = obj.incorporateSeeds(solution, source, sink, 10);
    // expected are 1.0 at positions from source
    assertThat((double) ret[0], closeTo(1.0, 1e-6));
    assertThat((double) ret[3], closeTo(1.0, 1e-6));
    assertThat((double) ret[4], closeTo(1.0, 1e-6));

    // expected are 0.0 at positions from source
    assertThat((double) ret[1], closeTo(0.0, 1e-6));
    assertThat((double) ret[9], closeTo(0.0, 1e-6));

    // expected are solution values at remaining positions
    assertThat((double) ret[2], closeTo(10.0, 1e-6));
    assertThat((double) ret[5], closeTo(11.0, 1e-6));
    assertThat((double) ret[6], closeTo(12.0, 1e-6));
    assertThat((double) ret[7], closeTo(13.0, 1e-6));
    assertThat((double) ret[8], closeTo(14.0, 1e-6));
  }

}
