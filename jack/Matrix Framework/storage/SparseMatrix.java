package storage;

import utils.Constants.matrixFormat;
import utils.Conversions;

public class SparseMatrix {
	//TODO
	private int[] rowInd;
	private int[] colInd;
	private double[] vals;
	private int m;
	private int n;
	private int nnz;
	private matrixFormat fmt;
	
	public SparseMatrix(int[] rowInd,int[] colInd, double[] vals,int m,int n, int nnz, matrixFormat fmt) {
		this.rowInd = rowInd;
		this.colInd = colInd;
		this.vals = vals;
		this.m = m;
		this.n = n;
		this.nnz = nnz;
		this.fmt = fmt;
	}
	
	public int[] GetRowInd() {
		return this.rowInd;
	}
	
	public int[] GetColInd() {
		return this.colInd;
	}
	
	public double[] GetVals() {
		return this.vals;
	}
	
	public int GetM() {
		return this.m;
	}
	
	public int GetN() {
		return this.n;
	}
	
	public int GetNnz() {
		return this.nnz;
	}
	
	public matrixFormat GetFormat() {
		return this.fmt;
	}
	
	public String toString() {
		return Conversions.SMtoDM(this).toString();
	}
	
	
}
