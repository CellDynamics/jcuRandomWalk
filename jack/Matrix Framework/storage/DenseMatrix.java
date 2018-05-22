package storage;


public class DenseMatrix {
	//TODO
	private int m;
	private int n;
	private double[][] data;
	
	public DenseMatrix(double[][] data) {
		this.m = data.length;
		this.n = data[0].length;
		this.data = data;
	}
	
	public int GetM() {
		return this.m;
	}
	
	public int GetN() {
		return this.n;
	}
	
	public double[][] GetData() {
		return this.data;
	}
	
	public String toString() {
		String out = "";
		for (int i = 0; i < this.m; i++) { //print out this representation
			for (int j = 0; j < this.n; j++) {
				out += this.data[i][j];
				out += " ";
			}
			out += '\n';
		}
		return out;
	}
	
}
