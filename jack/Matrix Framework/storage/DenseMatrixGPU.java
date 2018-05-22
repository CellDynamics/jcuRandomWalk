package storage;

import jcuda.*;
import utils.Conversions;

import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

public class DenseMatrixGPU {
	//TODO
	private Pointer data = new Pointer();
	private int m;
	private int n;
	private int ld;
	private int size;
	
	public DenseMatrixGPU (double[][] data) {
		this.m = data.length;
		this.n = data.length;
		this.size = this.m*this.n;
		this.ld = m;
		double[] temp = new double[this.size];
		{
			int ix = 0;
			for (int j = 0; j < this.n; j++)
				for (int i = 0; i < this.m; i++)
					temp[ix++] = data[i][j];
				
		}
		cudaMalloc(this.data, this.size*Sizeof.DOUBLE);
		cudaMemcpy(this.data, Pointer.to(temp), this.size*Sizeof.DOUBLE, cudaMemcpyHostToDevice);
		
	}
	
	public int GetM() {
		return this.m;
	}
	
	public int GetN() {
		return this.n;
	}
	
	public int GetLd() {
		return this.ld;
	}
	
	public int GetSize() {
		return this.size;
	}
	
	public Pointer GetPtr() {
		return this.data;
	}
	
	public void Free() {
		cudaFree(data);
		m = 0;
		n = 0;
		ld = 0;
	}
	
	public String toString() {
		return Conversions.DMGPUtoDM(this).toString();
	}
	
}
