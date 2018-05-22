package storage;

public class DenseVector {
	
	private double[] data; //vector data
	private int size; //size of vector
	
	public DenseVector(double[] in) { //create from input array
		this.data = in; //copy data
		this.size = in.length; //size of vector
	}
	
	public double[] GetData() { //return data
		return this.data;
	}
	
	public int GetSize() { //return size
		return this.size;
	}
}
