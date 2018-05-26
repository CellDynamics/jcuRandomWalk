package com.github.celldynamics.versioning;

import org.junit.Test;
import org.mockito.Mockito;

/**
 * @author baniu
 *
 */
public class ToolVersionTest {

  /**
   * Test method for
   * {@link com.github.celldynamics.versioning.ToolVersion#getFormattedToolVersion(com.github.celldynamics.versioning.ToolVersionStruct, java.lang.String[])}.
   */
  @Test
  public void testGetFormattedToolVersion() throws Exception {
    String[] aut = new String[] { "Till Bretschneider (Till.Bretschneider@warwick.ac.uk)",
        "Piotr Baniukiewicz (P.Baniukiewicz@warwick.ac.uk)" };
    ToolVersionStruct version = Mockito.mock(ToolVersionStruct.class);
    Mockito.when(version.getVersion()).thenReturn("v1.2.3");
    Mockito.when(version.getBuildstamp()).thenReturn("10:32:56");
    Mockito.when(version.getName()).thenReturn("jcuRandomWalk");
    String ret = ToolVersion.getFormattedToolVersion(version, aut);
    System.out.println(ret);
  }

}
