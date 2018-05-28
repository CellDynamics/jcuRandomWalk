package com.github.celldynamics.jcurandomwalk;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import jcuda.Pointer;
import jcuda.runtime.JCuda;

/**
 * General notebook for jCuda tests.
 * 
 * @author p.baniukiewicz
 *
 */
public class CudaTest {

  static {
    LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();
    Logger rootLogger = loggerContext.getLogger(CudaTest.class.getName());
    ((ch.qos.logback.classic.Logger) rootLogger).setLevel(Level.DEBUG);
  }
  /**
   * The Constant LOGGER.
   */
  static final Logger LOGGER = LoggerFactory.getLogger(CudaTest.class.getName());

  @Test
  public void test() {
    JCuda.setExceptionsEnabled(true);
    Pointer pointer = new Pointer();
    JCuda.cudaMalloc(pointer, 4);
    LOGGER.debug("Pointer: " + pointer);
    JCuda.cudaFree(pointer);
    JCuda.cudaFree(pointer);
  }

}
