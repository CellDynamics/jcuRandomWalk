package com.github.celldynamics.jcurandomwalk;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.IncidenceMatrixGeneratorTest;

import ij.ImageStack;

/**
 * Test class.
 * 
 * @author baniu
 *
 */
public class ArrayToolsTest {

  /**
   * The Constant LOGGER.
   */
  static final Logger LOGGER = LoggerFactory.getLogger(ArrayToolsTest.class.getName());

  /**
   * Test of {@link ArrayTools#printArray(Number[][])}.
   * 
   * @throws Exception on error
   */
  @Test
  public void testArray2Object() throws Exception {
    ImageStack stack = IncidenceMatrixGeneratorTest.getTestStack(4, 3, 2, "int");
    Number[][] ret = ArrayTools.array2Object(stack.getProcessor(1).getIntArray());
    LOGGER.debug(ArrayTools.printArray(ret));

    stack = IncidenceMatrixGeneratorTest.getTestStack(4, 3, 2, "double");
    ret = ArrayTools.array2Object(stack.getProcessor(1).getFloatArray());
    LOGGER.debug(ArrayTools.printArray(ret));
  }

}
