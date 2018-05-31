package com.github.celldynamics.jcurandomwalk;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;

import org.apache.commons.lang3.time.StopWatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.celldynamics.jcudarandomwalk.matrices.sparse.ISparseMatrix;
import com.github.celldynamics.jcudarandomwalk.matrices.sparse.SparseMatrixHost;
import com.github.celldynamics.jcurandomwalk.RandomWalkOptions.AlgOptions;

import ij.ImageStack;

/**
 * Incidence matrix generator.
 * 
 * <p>Generate incidence matrix and weight matrix for specified size of the stack (3D coordinate
 * space). Third dimension is optional.
 * 
 * @author baniu
 *
 */
public class IncidenceMatrixGenerator implements Serializable {
  /**
   * UUID.
   */
  private static final long serialVersionUID = 5111593778800000238L;

  static final Logger LOGGER = LoggerFactory.getLogger(IncidenceMatrixGenerator.class.getName());

  /*
   * Note that underlying array retrieved by IP.getFloatArray() is column ordered (x,y), therefore
   * ret[0][] is first column (x coordinate).
   * Internally IP keeps image in 1D array, see
   * com.github.celldynamics.quimp.utils.QuimPArrayUtils.castToNumber(ImageProcessor)
   */

  private transient ImageStack stack; // original image for segmentation
  private int nrows; // number of rows of stack
  private int ncols; // number of columns of stack
  private int nz; // number of slices of the stack
  private SparseMatrixHost incidence; // incidence matrix coords
  private transient SparseMatrixHost weights; // weights coords, depends on stack, not serialised
  private Integer[] sink; // indexes of pixel on bounding box

  // this array stores coordinates of pixels used for computing weights. The order is:
  // [ro1 col1 z1 row2 col2 z2 .....] Weight should be computed between P1 and P2, P3 and P4 etc.
  // Array is serialised and allows for restoring the whole structure and compute new weights
  // for any stack with the same dimensions like that the object was serialised.
  private int[] coords;
  private transient AlgOptions rwOptions = null;

  /**
   * For mocked tests. Do not use.
   */
  IncidenceMatrixGenerator() {

  }

  /**
   * Computes incidence matrix for given stack.
   * 
   * @param stack stack of images to be segmented
   */
  public IncidenceMatrixGenerator(ImageStack stack, AlgOptions options) {
    this.stack = stack;
    this.rwOptions = options;
    if (stack.getSize() < 3) {
      LOGGER.warn("Stack should have more than 3 slices due to how the sink box is computed.");
    }
    LOGGER.debug("Got stack: " + stack.toString());
    StopWatch timer = new StopWatch();
    nrows = stack.getHeight();
    ncols = stack.getWidth();
    nz = stack.getSize();
    timer.start();
    computeIncidence();
    computeSinkBox();
    timer.stop();
    LOGGER.info("Incidence and BBox matrices computed in " + timer.toString());
  }

  /**
   * Assign new stack to incidence matrix. Should be called when new weights need to be computed.
   * 
   * @param stack Stack to assign. Should have the same size like that used to construct the object.
   */
  public void assignStack(ImageStack stack) {
    if (this.stack.getWidth() != stack.getWidth() || this.stack.getHeight() != stack.getHeight()
            || this.stack.getSize() != stack.getSize()) {
      throw new IllegalArgumentException("Geometry of this stack is different than that used "
              + "for incidence matrix generation.");
    }
    this.stack = stack;
    recomputeWeights();
  }

  /**
   * Compute indexes of points on bounding box. Return sorted array
   */
  void computeSinkBox() {
    // TODO consider use one loop and parallelism
    int verticesInLayer = nrows * ncols;
    // taken from Till's code
    int numBoxEl = (nrows * 2 + (ncols - 2) * 2) * (nz - 2) + 2 * nrows * ncols;
    sink = new Integer[numBoxEl];
    int l = 0;
    for (int k = 1; k < nz - 1; k++) {
      for (int i = 0; i < nrows; i++) {
        sink[l++] = i + k * verticesInLayer;
        sink[l++] = (ncols - 1) * nrows + i + k * verticesInLayer;
      }
      for (int i = 1; i < ncols - 1; i++) {
        sink[l++] = i * nrows + k * verticesInLayer;
        sink[l++] = i * nrows + nrows - 1 + k * verticesInLayer;
      }
    }

    for (int k = 0; k < nz; k += nz - 1) {
      for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
          sink[l++] = j * nrows + i + k * verticesInLayer;
        }
      }
    }
    Arrays.sort(sink);
  }

  /**
   * Compute incidence matrix and fills this object.
   * 
   * <p>Pixels in stack are numbered along columns:
   * 0 3
   * 1 4
   * 2 5...
   * 
   * <p>Order of edges in incidence matrix and weight matrix is: Right, Lower, Bottom, where:
   * <ol>
   * <li>Right is pixel on right to current one: (x0+1,y0,z0)
   * <li>Lower is pixel below current one on the same slice: (x0, y0+1,z0)
   * <li>Bottom is pixel on next layer: (x0,y0,z0+1)
   * </ol>
   * 
   * <p>Note that vector of values in incidence matrix is order along rows, it matters only in
   * tests.
   * 
   * @see #getIncidence()
   * @see #getWeights()
   * @see SparseMatrixHost
   */
  void computeIncidence() {
    // right - pixel on right
    // lower - pixel below of current but on the same slice
    // bottom - pixel below but on lower slice
    final boolean bc = false; // false = do not use BC
    int edges = getEdgesNumber(nrows, ncols, nz); // number of edges
    int verts = getNodesNumber(nrows, ncols, nz);
    LOGGER.info("Number of edges: " + edges + ", number of verts: " + verts);
    weights = new SparseMatrixHost(edges);
    coords = new int[edges * 6]; // 2 pixels * 3 coords for each edge
    incidence = new SparseMatrixHost(edges * 2); // each edge has at least 2 points
    // counter for aggregating opposite pairs of neighbouring pixel in one row of
    // incidence matrix. It is incidence matrix row index (edges)
    int in = 0;
    int cl = 0;
    int[] rcz = new int[3]; // [row col z] (y,x,z) - current pixel
    int[] rczR = new int[3]; // pixel next to current (right - use separate variables for cleaner c)
    int[] rczL = new int[3]; // pixel next to current - lower
    int[] rczB = new int[3]; // pixel next to current bottom
    for (int i = 0; i < verts; i++) {
      lin20ind(i, nrows, ncols, nz, rcz);
      int col = rcz[1]; // just helpers
      int row = rcz[0];
      int z = rcz[2];
      // find 3 edges from current node (assume travelling right|lower|bottom)
      int right = col + 1;
      int lower = row + 1;
      int bottom = z + 1;
      if (bc) { // apply periodic BC
        if (right >= ncols) {
          right = 0;
        }
        if (lower >= nrows) {
          lower = 0;
        }
        if (bottom >= nz) {
          bottom = 0;
        }
      }
      rczR[0] = row; // FIXME Optimise and avoid extra variables
      rczR[1] = right;
      rczR[2] = z;
      int rightLin = ind20lin(rczR, nrows, ncols, nz);
      rczL[0] = lower;
      rczL[1] = col;
      rczL[2] = z;
      int lowerLin = ind20lin(rczL, nrows, ncols, nz);
      rczB[0] = row;
      rczB[1] = col;
      rczB[2] = bottom;
      int bottomLin = ind20lin(rczB, nrows, ncols, nz);
      // 1). from each current pixel travel only to right|lower|bottom
      // directions to avoid duplicating edges
      // 2). current pixel is positive (1), right|lower|bottom negative (-1)
      // 3). Order of edges in incidence matrix is vertex related:
      // E1 - edge on right of 1st vertex
      // E2 - edge to bottom from 1st vertexL(:,seeds)=[];
      // E3 - edge to bottom layer
      // E4 - edge to right of 2nd vertex
      // % ....
      //
      // edge from current pixel to right
      if (right < ncols) { // if no BC this can be larger or equal than ncols and then it is skipped
        incidence.add(in, i, 1.0f);
        incidence.add(in, rightLin, -1.0f);
        // weights.add(in, in, computeWeight(stack, rczR, rcz, sigmaGrad, sigmaMean, meanSource));
        coords[cl++] = rczR[0];
        coords[cl++] = rczR[1];
        coords[cl++] = rczR[2];
        coords[cl++] = rcz[0];
        coords[cl++] = rcz[1];
        coords[cl++] = rcz[2];

        in++; // go to next edge (next row in incidence matrix)
      }
      // edge from current pixel to lower
      if (lower < nrows) {
        incidence.add(in, i, 1.0f);
        incidence.add(in, lowerLin, -1.0f);
        // weights.add(in, in, computeWeight(stack, rczL, rcz, sigmaGrad, sigmaMean, meanSource));
        coords[cl++] = rczL[0];
        coords[cl++] = rczL[1];
        coords[cl++] = rczL[2];
        coords[cl++] = rcz[0];
        coords[cl++] = rcz[1];
        coords[cl++] = rcz[2];
        in++; // go to next edge (next row in incidence
      }
      // edge from current pixel to bottom
      if (nz > 1 && bottom < nz) {
        incidence.add(in, i, 1.0f);
        incidence.add(in, bottomLin, -1.0f);
        // weights.add(in, in, computeWeight(stack, rczB, rcz, sigmaGrad, sigmaMean, meanSource));
        coords[cl++] = rczB[0];
        coords[cl++] = rczB[1];
        coords[cl++] = rczB[2];
        coords[cl++] = rcz[0];
        coords[cl++] = rcz[1];
        coords[cl++] = rcz[2];
        in++; // go to next edge (next row in incidence
      }
    }
    recomputeWeights(); // fill weights
    // update dimension after using add
    incidence.updateDimension();
  }

  /**
   * Recompute weights for stack. Uses coordinates stored by {@link #computeIncidence()}.
   */
  private void recomputeWeights() {
    int edges = getEdgesNumber(nrows, ncols, nz); // number of edges
    weights = new SparseMatrixHost(edges);
    int l = 0; // diagonal counter
    int[] rcz = new int[3];
    int[] rc = new int[3];
    for (int i = 0; i < coords.length; i += 6) {
      rcz[0] = coords[i];
      rcz[1] = coords[i + 1];
      rcz[2] = coords[i + 2];
      rc[0] = coords[i + 3];
      rc[1] = coords[i + 4];
      rc[2] = coords[i + 5];
      double w = computeWeight(stack, rcz, rc, rwOptions.sigmaGrad, rwOptions.sigmaMean,
              rwOptions.meanSource);
      weights.add(l, l, (float) w);
      l++;
    }
    weights.updateDimension();
  }

  /**
   * Compute weight between two pixels.
   * 
   * @param stack 3D stack (one slice for 2d)
   * @param p1 first point [row col z] (y,x,z)
   * @param p2 second point [row col z] (y,x,z)
   * @param sigmaGrad sigma gradient parameter
   * @param sigmaMean mean
   * @param meanSource mean of source
   * @return weight between p1 and p2
   */
  double computeWeight(ImageStack stack, int[] p1, int[] p2, double sigmaGrad, double sigmaMean,
          double meanSource) {
    //!>
    double sqDiff1 = Math.pow(
            stack.getVoxel(p1[1], p1[0], p1[2])
            - stack.getVoxel(p2[1], p2[0], p2[2]),
            2);
    double sqDiff2 = Math.pow(
            stack.getVoxel(p1[1], p1[0], p1[2])
            - meanSource,
            2);
    double sigmaGrad2 = sigmaGrad * sigmaGrad;
    double sigmaMean2 = sigmaMean * sigmaMean;
    double ret = Math.exp(
            -0.5 * sqDiff1 / sigmaGrad2
            - 0.5 * sqDiff2 / sigmaMean2);
    return ret;
    //!<
  }

  /**
   * Compute number of nodes in graph for specified size of image.
   * 
   * @param nrows number of rows of the image
   * @param ncols number of columns of the image
   * @param nz number of slices of the image
   * @return number of nodes
   */
  public static int getNodesNumber(int nrows, int ncols, int nz) {
    return nrows * ncols * nz;
  }

  /**
   * Compute number of edges in graph for specified size of image. Assumes 6 point lattice.
   * 
   * @param nrows number of rows of the image
   * @param ncols number of columns of the image
   * @param nz number of slices of the image
   * @return number of edges
   */
  public static int getEdgesNumber(int nrows, int ncols, int nz) {
    int verticesInLayer = nrows * ncols;
    int edgesInLayer = nrows * (ncols - 1) + ncols * (nrows - 1);
    return nz * edgesInLayer + (nz - 1) * verticesInLayer;

  }

  /**
   * Retrieve incidence coordinates from the object.
   * 
   * <p>Incidence matrix has dimensions (after evolving to full):
   * nrows: <tt>IncidenceMatrixGenerator.getEdgesNumber(height, width, nz)</tt>
   * ncols: <tt>IncidenceMatrixGenerator.getNodesNumber(height, width, nz)</tt>
   * 
   * <p>It should be addressed as:
   * <tt>double[][] f = obj.getIncidence().full();</tt>
   * <tt>f[c][r]</tt>
   * 
   * @return the incidence
   */
  public ISparseMatrix getIncidence() {
    return incidence;
  }

  /**
   * Retrieve weights coordinates from the object.
   * 
   * <p>Weight matrix is square of size <tt>IncidenceMatrixGenerator.getEdgesNumber(height, width,
   * nz)</tt>
   * 
   * @return the weights
   */
  public ISparseMatrix getWeights() {
    return weights;
  }

  /**
   * Convert linear index to x,y coordinates, 0-based and column ordered.
   * 
   * @param lin linear index to convert.
   * @param nrows number of rows in stack
   * @param ncols number of columns in stack
   * @param nz number of slices
   * @param ind return array (for speed). ind[0] is row, ind[1] column, ind[2] is slice. Must be
   *        preallocated.
   *        Array must have size of 3
   */
  public static void lin20ind(int lin, int nrows, int ncols, int nz, int[] ind) {
    int k2 = nrows * ncols;

    int vi = lin % k2;
    int vj = (lin - vi) / k2;
    int ndx = vi;

    vi = ndx % nrows;
    ind[1] = (ndx - vi) / nrows;
    ind[0] = vi;
    ind[2] = vj;

  }

  /**
   * Convert x,y,z coordinates to linear index. 0-based and columns ordered.
   * 
   * @param ind coordinates to convert. ind[0] - row, ind[1] - column, ind [2] - z.
   * @param nrows number of rows
   * @param ncols number of columns
   * @param nz number of slices
   * @return linear index
   */
  public static int ind20lin(int[] ind, int nrows, int ncols, int nz) {
    return nrows * ind[1] + ind[0] + ind[2] * (nrows * ncols);
  }

  /**
   * Serialize this object.
   * 
   * @param filename name of the file
   * @throws IOException on problem with saving
   */
  public void saveObject(String filename) throws IOException {
    FileOutputStream file = new FileOutputStream(filename);
    ObjectOutputStream out = new ObjectOutputStream(file);
    out.writeObject(this);
    out.close();
    file.close();
    LOGGER.info("Incidence matrix saved under " + filename);
  }

  /**
   * Load serialized object.
   * 
   * <p>stack is not serialised.
   * 
   * @param filename name of the file
   * @param stack stack of the same size as that used for constructing the object.
   * @param options Options used for generating weights
   * @return Instance of loaded object
   * @throws IOException when file can not be read or deserialised
   * @throws ClassNotFoundException when file can not be read or
   *         deserialised
   */
  public static IncidenceMatrixGenerator restoreObject(String filename, ImageStack stack,
          AlgOptions options) throws IOException, ClassNotFoundException {
    FileInputStream file = new FileInputStream(filename);
    ObjectInputStream in = new ObjectInputStream(file);
    try {
      StopWatch timer = new StopWatch();
      timer.start();
      IncidenceMatrixGenerator ret = (IncidenceMatrixGenerator) in.readObject();
      ret.rwOptions = options; // assign here
      ret.stack = stack; // assign stack will require this
      ret.assignStack(stack); // stack is not serialised, assign here
      timer.stop();
      LOGGER.info("Incidence matrix restored from " + filename + " in " + timer.toString());
      return ret;
    } catch (IOException | ClassNotFoundException e) {
      throw e;
    } finally {
      in.close();
      file.close();
    }
  }

  /**
   * Get sink bounding box - pixels on edges. Array is sorted.
   * 
   * @return indexes of pixels on edges.
   */
  public Integer[] getSinkBox() {
    return sink;
  }
}
