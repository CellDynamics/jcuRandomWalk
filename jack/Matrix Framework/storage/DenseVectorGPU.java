package storage;

import jcuda.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;


public class DenseVectorGPU {
	
	private Pointer gpuPtr = new Pointer(); //pointer for vector
	private double[] host; //host array for data
	private int size; //size of vector
	
	public DenseVectorGPU(double[] in) {
		
		this.size = in.length; //get size of vector
		this.host = in.clone(); //clone data so it can be modified elsewhere
		
		cudaMalloc(this.gpuPtr, this.size*Sizeof.DOUBLE); //allocate pointer memory
		cudaMemcpy(this.gpuPtr, Pointer.to(host), size*Sizeof.DOUBLE, cudaMemcpyHostToDevice); //copy data over
		
	}
	
	public Pointer GetPtr() { //returns pointer
		return this.gpuPtr;
	}
	
	public int GetSize() { //returns size
		return this.size;
	}
	
	public void Free() { //frees up this instances
		cudaFree(this.gpuPtr);
		this.size = 0;
		this.host = null;
	}
	
}
