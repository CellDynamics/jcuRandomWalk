package com.github.celldynamics.jcudarandomwalk.matrices;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
public class IncidenceMatrixGenerator {
  static final Logger LOGGER = LoggerFactory.getLogger(IncidenceMatrixGenerator.class.getName());

  /*
   * Note that underlying array retrieved by IP.getFloatArray() is column ordered (x,y), therefore
   * ret[0][] is first column (x coordinate).
   * Internally IP keeps image in 1D array, see
   * com.github.celldynamics.quimp.utils.QuimPArrayUtils.castToNumber(ImageProcessor)
   */
  private ImageStack stack; // original image for segmentation
  private int nrows; // number of rows of stack
  private int ncols; // number of columns of stack
  private int nz; // number of slices of the stack
  private SparseMatrix incidence; // incidence matrix coords
  private SparseMatrix weights; // weights coords

  /**
   * Computes incidence matrix for given stack.
   * 
   * @param stack stack of images to be segmented
   */
  public IncidenceMatrixGenerator(ImageStack stack) {
    this.stack = stack;
    LOGGER.debug("Got stack: " + stack.toString());
    nrows = stack.getHeight();
    ncols = stack.getWidth();
    nz = stack.getSize();
    computeIncidence();
  }

  /**
   * Compute incidence matrix and fills this object.
   * 
   * @see #getIncidence()
   * @see #getWeights()
   * @see SparseMatrix
   */
  void computeIncidence() {
    final double sigmaGrad = 0.1; // TODO expose
    final double sigmaMean = 1e6;
    final double meanSource = 0.6;
    // right - pixel on right
    // lower - pixel below of current but on the same slice
    // bottom - pixel below but on lower slice
    final boolean bc = false; // false = do not use BC
    int edges = getEdgesNumber(nrows, ncols, nz); // number of edges
    int verts = getNodesNumber(nrows, ncols, nz);
    LOGGER.debug("Number of edges: " + edges + " number of verts: " + verts);
    weights = new SparseMatrix(edges);
    incidence = new SparseMatrix(edges * 2); // each edge has at leas 2 points
    // counter for aggregating opposite pairs of neighbouring pixel in one row of
    // incidence matrix. It is incidence matrix row index (edges)
    int in = 0;
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
        incidence.add(in, rightLin, -1.0);
        incidence.add(in, i, 1.0);
        weights.add(in, in, computeWeight(stack, rczR, rcz, sigmaGrad, sigmaMean, meanSource));
        in++; // go to next edge (next row in incidence matrix)
      }
      // edge from current pixel to lower
      if (lower < nrows) {
        incidence.add(in, lowerLin, -1.0);
        incidence.add(in, i, 1.0);
        weights.add(in, in, computeWeight(stack, rczL, rcz, sigmaGrad, sigmaMean, meanSource));
        in++; // go to next edge (next row in incidence
      }
      // edge from current pixel to bottom
      if (nz > 1 && bottom < nz) {
        incidence.add(in, bottomLin, -1.0);
        incidence.add(in, i, 1.0);
        weights.add(in, in, computeWeight(stack, rczB, rcz, sigmaGrad, sigmaMean, meanSource));
        in++; // go to next edge (next row in incidence
      }
    }
  }

  /**
   * Compute weight between two pixels.
   * 
   * @param stack 3D stack (one slice for 2d)
   * @param p1 first point [row col z] (y,x,z)
   * @param p2 second point [row col z] (y,x,z)
   * @param sigmaGrad
   * @param sigmaMean
   * @param meanSource
   * @return weight between p1 and p2
   */
  double computeWeight(ImageStack stack, int[] p1, int[] p2, double sigmaGrad, double sigmaMean,
          double meanSource) {
    //!>
    double sqDiff1 = Math.pow(
            stack.getVoxel(p1[1], p1[0], p1[2]) -
            stack.getVoxel(p2[1], p2[0], p2[2]),
            2);
    double sqDiff2 = Math.pow(
            stack.getVoxel(p1[1], p1[0], p1[2]) -
            meanSource,
            2);
    double sigmaGrad2 = sigmaGrad*sigmaGrad;
    double sigmaMean2 = sigmaMean*sigmaMean;
    double ret = Math.exp(
            -0.5 * sqDiff1 / sigmaGrad2 - 
            0.5 * sqDiff2 / sigmaMean2);
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
  public SparseMatrix getIncidence() {
    return incidence;
  }

  /**
   * Retrieve weights coordinates from the object.
   * 
   * <p>Weight matrix is square of size IncidenceMatrixGenerator.getEdgesNumber(height, width, nz)</tt>
   * 
   * @return the weights
   */
  public SparseMatrix getWeights() {
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
   * Convert x,y,z coordinates to linear index. )-based and columns ordered.
   * 
   * @param ind coordinates to convert. ind[0] - x, ind[1] - y, ind [2] - z.
   * @param nrows number of rows
   * @param ncols number of columns
   * @param nz number of slices
   * @return linear index
   */
  public static int ind20lin(int[] ind, int nrows, int ncols, int nz) {
    return nrows * ind[1] + ind[0] + ind[2] * (nrows * ncols);
  }
}
