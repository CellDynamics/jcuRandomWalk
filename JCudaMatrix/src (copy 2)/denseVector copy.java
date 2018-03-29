import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import jcuda.*;


public class denseVector {

	private Pointer v_gpuPtr = new Pointer();
	private int size;

	public denseVector(float[] v){
		System.out.println("in denseVector constructor ");

		size = v.length;

		cudaMalloc(v_gpuPtr, size*Sizeof.FLOAT);
		cudaMemcpy(v_gpuPtr, Pointer.to(v), size*Sizeof.FLOAT, cudaMemcpyHostToDevice);

	}

	public float[] getVector(){
		float v_host[] = new float[size];

		cudaMemcpy(Pointer.to(v_host), v_gpuPtr, size*Sizeof.FLOAT, cudaMemcpyDeviceToHost);

		return v_host;
	}
	
	public Pointer getPtr(){
		return v_gpuPtr;
	}
	
	public void free(){
	    cudaFree(v_gpuPtr);
	}

}


