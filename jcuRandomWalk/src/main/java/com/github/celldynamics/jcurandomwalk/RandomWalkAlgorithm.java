package com.github.celldynamics.jcurandomwalk;

import java.nio.file.Path;
import java.util.Arrays;

import org.apache.commons.lang3.time.StopWatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.ISparseMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.IStoredOnGpu;
import com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixDevice;
import com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixHost;
import com.github.celldynamics.jcudarandomwalk.matrices.SparseMatrixType;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.StackConverter;
import ij.process.StackProcessor;
import ij.process.StackStatistics;
import jcuda.jcusparse.JCusparse;
import jcuda.runtime.JCuda;

/**
 * @author p.baniukiewicz
 *
 */
public class RandomWalkAlgorithm {

  static final Logger LOGGER = LoggerFactory.getLogger(RandomWalkAlgorithm.class.getName());

  ImageStack stack;
  IncidenceMatrixGenerator img;
  RandomWalkOptions options;

  RandomWalkAlgorithm() {

  }

  /**
   * Default constructor,
   * 
   * @param stack stack to be segmented
   * @param options options
   */
  public RandomWalkAlgorithm(ImageStack stack, RandomWalkOptions options) {
    this.options = options;
    this.stack = stack;
  }

  /**
   * Compute incidence matrix for stack.
   * 
   * @param always if true compute always if false try to load first. If file has not been found
   *        compute new incidence matrix and save it.
   * @throws Exception
   */
  public void computeIncidence(boolean always) throws Exception {
    if (always) {
      LOGGER.info("Computing new incidence matrix");
      img = new IncidenceMatrixGenerator(stack);
    } else {
      String filename = options.configBaseName + "_" + stack.toString() + options.configBaseExt;
      Path fullToFilename = options.configFolder.resolve(filename);
      if (fullToFilename.toFile().exists()) { // try to load if exists
        try {
          img = IncidenceMatrixGenerator.restoreObject(fullToFilename.toString(), stack); // load
        } catch (Exception e) {
          LOGGER.error(
                  "Incidence file could not be restored (" + filename + "): " + e.getMessage());
          LOGGER.debug(e.getMessage(), e);
          throw e;
        }
      } else { // if does not exist generate new one and save
        LOGGER.info("Computing new incidence matrix");
        img = new IncidenceMatrixGenerator(stack);
        img.saveObject(fullToFilename.toString());
      }
    }
  }

  /**
   * Compute Laplacian A'XA.
   * 
   * @return Laplacian on GPU
   */
  public ISparseMatrix computeLaplacian() {
    StopWatch timer = new StopWatch();
    timer.start();
    ISparseMatrix incidenceGpu = ((SparseMatrixHost) img.getIncidence()).toGpu().convert2csr();
    ISparseMatrix incidenceGpuT = incidenceGpu.transpose();

    ISparseMatrix wGpu = ((SparseMatrixHost) img.getWeights()).toGpu().convert2csr();
    // A'*W*A
    // ISparseMatrix ATW = incidenceGpuT.multiply(wGpu);
    // ISparseMatrix ATWA = ATW.multiply(incidenceGpu);
    ISparseMatrix lap = incidenceGpuT.multiply(wGpu).multiply(incidenceGpu);
    timer.stop();
    LOGGER.info("Laplacian computed in " + timer.toString());
    return lap;
  }

  /**
   * This method remove rows and columns from source and sink arrays from Laplacian lap.
   * 
   * <p>Laplacian must be square. Indexes from source and sink are merged first to deal with
   * duplicates. There is no special checking implemented so largest index from sink and source
   * should be < length(lap)
   * 
   * @param lap square matrix where columns are rows are removed from
   * @param source list of indices to remove
   * @param sink second list of indices to remove, merged with the first one
   * @return Reduced matrix lap.
   */
  public ISparseMatrix computeReducedLaplacian(ISparseMatrix lap, int[] source, int[] sink) {
    if (lap.getColNumber() != lap.getRowNumber()) {
      throw new IllegalArgumentException("Matrix should be square");
    }
    StopWatch timer = new StopWatch();
    timer.start();
    ISparseMatrix lapCoo = lap.convert2coo();
    ISparseMatrix cpuL;
    if (lap instanceof IStoredOnGpu) {
      cpuL = ((IStoredOnGpu) lapCoo).toCpu();
    } else {
      cpuL = lapCoo;
    }

    // any 1 at index i in this array stand for index i to remove from Lap
    // merge two arrays in one because they can point the same row/column
    int[] toRem = new int[lap.getColNumber()];
    for (int s = 0; s < source.length; s++) {
      toRem[source[s]] = 1;
    }
    for (int s = 0; s < sink.length; s++) {
      toRem[sink[s]] = 1;
    }

    // iterate over indices lists and mark those to remove by -1
    int[] colInd = cpuL.getColInd();
    for (int i = 0; i < colInd.length; i++) {
      if (toRem[colInd[i]] > 0) {
        colInd[i] = -1; // to remove
      }
    }
    int[] rowInd = cpuL.getRowInd();
    for (int i = 0; i < rowInd.length; i++) {
      if (toRem[rowInd[i]] > 0) {
        rowInd[i] = -1; // to remove
      }
    }
    // compute number of nonzero elements that remains
    // find how many col is >=0 - valid cols ...
    int remainingCol = 0;
    for (int i = 0; i < colInd.length; i++) {
      if (colInd[i] >= 0) {
        remainingCol++;
      }
    }
    // ... and find how many rows is >=0 - valid rows
    int remainingRow = 0;
    for (int i = 0; i < rowInd.length; i++) {
      if (rowInd[i] >= 0) {
        remainingRow++;
      }
    }
    // take smaller of them because valid value is that that has both indices positive
    int remaining = Math.min(remainingCol, remainingRow);

    // copy those that are >0 to new array
    int[] newColInd = new int[remaining];
    int[] newRowInd = new int[remaining];
    double[] newVal = new double[remaining];

    int l = 0;
    for (int i = 0; i < cpuL.getElementNumber(); i++) {
      if (colInd[i] < 0 || rowInd[i] < 0) {
        continue;
      }
      newColInd[l] = colInd[i];
      newRowInd[l] = rowInd[i];
      newVal[l] = cpuL.getVal()[i];
      l++;
    }
    // compress
    // newColInd contains only valid nonzero elements (without those from deleted rows and
    // cols) but indexes contain gaps, e.g. if 2nd column was removed newColInd will keep next
    // column after as third whereas it should be moved to left and become the second
    // because we assumed square matrix we will go through toRem array and check which indexes were
    // removed (marked by 1 at index i - removed) and then decrease all indexes larger than those
    // removed in newColInd/newRowInd by one to shift them
    // These arrays need to be copied first otherwise next comparison would be wrong
    int[] newColIndcp = Arrays.copyOf(newColInd, newColInd.length);
    int[] newRowIndcp = Arrays.copyOf(newRowInd, newRowInd.length);

    for (int i = 0; i < toRem.length; i++) {
      if (toRem[i] > 0) { // compress all indices larger than i
        for (int k = 0; k < remaining; k++) { // go through sparse indexes
          if (newColInd[k] > i) { // if any is larger than that removed (i)
            newColIndcp[k]--; // reduce it by one, note that we reduce copy
          }
          if (newRowInd[k] > i) { // the same for rows
            newRowIndcp[k]--;
          }
        }

      }
    }
    // use reduced indexes to build new Sparse array
    ISparseMatrix reducedL = new SparseMatrixDevice(newRowIndcp, newColIndcp, newVal,
            SparseMatrixType.MATRIX_FORMAT_COO).convert2csr();
    LOGGER.info("Laplacian reduced in " + timer.toString());
    return reducedL;
  }

  /**
   * Apply default processing to stack. Assumes 8-bit imput.
   * 
   * <p>Apply 3x3 median filer 2D in each slice and normalisation.
   * 
   */
  public void processStack() {
    StopWatch timer = new StopWatch();
    timer.start();
    ImageStack filterOut =
            ImageStack.create(stack.getWidth(), stack.getHeight(), stack.getSize(), 8);
    new StackProcessor(this.stack).filter3D(filterOut, 1, 1, 1, 0, stack.getSize(),
            StackProcessor.FILTER_MEDIAN);
    ImagePlus ip = new ImagePlus("", filterOut);

    StackStatistics stats = new StackStatistics(ip);
    double min = stats.min;
    double max = stats.max; // test if computed for the whole stack
    // convert stack to Float
    StackConverter stc = new StackConverter(ip);
    stc.convertToGray32();

    for (int z = 1; z <= ip.getImageStackSize(); z++) {
      ip.getStack().getProcessor(z).subtract(min);
      ip.getStack().getProcessor(z).sqrt();
    }
    stats = new StackStatistics(ip);
    max = 1 / stats.max;
    for (int z = 1; z <= ip.getImageStackSize(); z++) {
      ip.getStack().getProcessor(z).multiply(max);
    }
    timer.stop();
    this.stack = ip.getStack();
    LOGGER.info("Stack normalised in " + timer.toString());
  }

  /**
   * Obtain indices of seed pixels from stack. Indices are computed in column-ordered manner.
   * 
   * <p>Ordering:
   * 0 3
   * 1 4
   * 2 5
   * 
   * @param seedStack Pixels with intensity >0 are considered as seed. Expect 8-bit binary stacks.
   * @return array of indices.
   */
  public int[] getSourceIndices(ImageStack seedStack) {
    if (!seedStack.getProcessor(1).isBinary()) {
      throw new IllegalArgumentException("Seed stack is not binary");
    }
    StopWatch timer = new StopWatch();
    timer.start();
    int l = 0;
    StackStatistics stat = new StackStatistics(new ImagePlus("", seedStack));
    int[] ret = new int[stat.histogram[stat.histogram.length - 1]];
    int[] ind = new int[3];
    for (int z = 0; z < seedStack.getSize(); z++) {
      for (int x = 0; x < seedStack.getWidth(); x++) {
        for (int y = 0; y < seedStack.getHeight(); y++) {
          if (seedStack.getVoxel(x, y, z) > 0) {
            ind[0] = y; // row
            ind[1] = x;
            ind[2] = z;
            ret[l++] = IncidenceMatrixGenerator.ind20lin(ind, seedStack.getHeight(),
                    seedStack.getWidth(), seedStack.getSize());
          }
        }
      }
    }
    timer.stop();
    LOGGER.info("Seeds processed in " + timer.toString());
    return ret;
  }

  /**
   * Must be called on very beginning.
   */
  public static void initilizeGpu() {
    JCusparse.setExceptionsEnabled(true);
    JCuda.setExceptionsEnabled(true);
    JCusparse.cusparseCreate(SparseMatrixDevice.handle);
  }

  /**
   * Must be called at very end.
   */
  public static void finish() {
    JCusparse.setExceptionsEnabled(false);
    JCuda.setExceptionsEnabled(false);
    JCusparse.cusparseDestroy(SparseMatrixDevice.handle);
  }
}
