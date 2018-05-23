package com.github.celldynamics.jcurandomwalk;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixHost;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import ij.ImageStack;

/**
 * @author p.baniukiewicz
 *
 */
public class RandomWalkAlgorithmTest {

  static final Logger LOGGER = LoggerFactory.getLogger(RandomWalkAlgorithmTest.class.getName());

  @Rule
  public TemporaryFolder folder = new TemporaryFolder();
  @Rule
  public MockitoRule mockitoRule = MockitoJUnit.rule();

  @Mock
  IncidenceMatrixGenerator img;
  @InjectMocks
  private RandomWalkAlgorithm objMocked;

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

  /**
   * @throws java.lang.Exception
   */
  @Before
  public void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    stack = TestDataGenerators.getTestStack(width, height, nz, "double");
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
    RandomWalkAlgorithm.initilizeGpu();
  }

  /**
   * Disable exceptions.
   */
  @AfterClass
  public static void after() {
    RandomWalkAlgorithm.finish();
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
   * Test method for
   * {@link com.github.celldynamics.jcurandomwalk.RandomWalkAlgorithm#computeLaplacean()}.
   */
  @Test
  public void testComputeLaplacean() throws Exception {
    RandomWalkOptions options = new RandomWalkOptions();
    options.configFolder = folder.newFolder().toPath();
    RandomWalkAlgorithm obj = new RandomWalkAlgorithm(stack, options);
    obj.computeIncidence(true);
    ISparseMatrix lap = obj.computeLaplacean();
  }

  @Test
  public void testComputeLaplacean_1() throws Exception {
    RandomWalkOptions options = new RandomWalkOptions();
    options.configFolder = folder.newFolder().toPath();
    // RandomWalkAlgorithm obj = new RandomWalkAlgorithm(stack, options);
    // obj.img = Mockito.spy(new IncidenceMatrixGenerator(stack));
    // Mockito.doReturn(new double[] { 1.0, 2.0 }).when(objMocked.img).getIncidence();
    Mockito.when(img.getIncidence()).thenReturn(new SparseMatrixHost(2));
    LOGGER.debug("" + objMocked.img.getIncidence());
  }

}
