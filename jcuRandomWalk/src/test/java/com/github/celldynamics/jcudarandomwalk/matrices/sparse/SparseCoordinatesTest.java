package com.github.celldynamics.jcudarandomwalk.matrices.sparse;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;

import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.Test;

/**
 * SparseCoordinates base.
 * 
 * @author baniu
 *
 */
public class SparseCoordinatesTest {

  /**
   * Test of add.
   * 
   * @throws Exception Exception
   */
  @Test
  public void testAdd() throws Exception {
    SparseCoordinates test = new SparseCoordinates(9);
    test.add(0, 0, 1);
    test.add(0, 1, 4);
    test.add(1, 1, 2);
    test.add(1, 2, 3);
    test.add(2, 0, 5);
    test.add(2, 3, 7);
    test.add(2, 4, 8);
    test.add(3, 2, 9);
    test.add(3, 4, 6);

    assertThat(test.getColInd().length, is(9));
    assertThat(Arrays.asList(ArrayUtils.toObject(test.getVal())),
            contains(1.0f, 4.0f, 2.0f, 3.0f, 5.0f, 7.0f, 8.0f, 9.0f, 6.0f)); // order from page 16
  }

}
