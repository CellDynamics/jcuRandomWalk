import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import jcuda.*;


public class denseVector {

	private Pointer v_gpuPtr = new Pointer();
	private int size;

	public denseVector(float[] v){
		// creates dense Vector from float array
		size = v.length;

		cudaMalloc(v_gpuPtr, size*Sizeof.FLOAT);
		cudaMemcpy(v_gpuPtr, Pointer.to(v), size*Sizeof.FLOAT, cudaMemcpyHostToDevice);
	}

	public denseVector(int n){
		// allocates space for dense Vector

		cudaMalloc(v_gpuPtr, n*Sizeof.FLOAT);
	}

	public float[] getVector(){
		// returns float array on host
		float v_host[] = new float[size];

		cudaMemcpy(Pointer.to(v_host), v_gpuPtr, size*Sizeof.FLOAT, cudaMemcpyDeviceToHost);

		return v_host;
	}

	public Pointer getPtr(){
		// returns gpu pointer
		return v_gpuPtr;
	}

	public void free(){
		// frees gpu memory
		cudaFree(v_gpuPtr);
	}

}


