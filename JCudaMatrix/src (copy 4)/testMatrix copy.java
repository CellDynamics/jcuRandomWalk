import jcuda.jcusparse.cusparseHandle;
import static jcuda.jcusparse.JCusparse.*;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.ParseException;
import java.util.ArrayList;

public class testMatrix {


	public static void main(String[] args) throws IOException, ParseException{




////create lower triangular test matrix
//float[][] a={{1.f,0.f,0.f,0.f},{2.f,3.f,0.f,0.f},{4.f,5.f,6.f,0.f},{7.f,8.f,9.f,10.f}};
////create rhs vector so that a*[5,2,6,3]=v
//float[] v={5.f, 16.f, 66.f, 135.f};
//
////print inputs
//System.out.println("matrix a");
//printMatrix(a,4,4);
//System.out.println("vector v");
//printVector(v);



String directory = "./data/"; 
String path=Paths.get(".").toAbsolutePath().normalize().toString();
System.out.println("path "+path);
int [] q = new int[1]; //size of matrix

float[][] a=readSquareMatrixFromFile(directory+"Lap.txt", q );

float[] v = readVectorFromFile(directory+"B.txt");

//printVector(v);





//set up handle
cusparseHandle handle = new cusparseHandle();
cusolverSpHandle handleSp = new cusolverSpHandle();
cusolverSpCreate();

csrSparseMatrix  cu_A = new  csrSparseMatrix(handle, a, q[0], q[0]);
System.out.print("sparse matrix cu_A created\n");

denseVector  cu_V = new  denseVector(v);
System.out.print("dense vector cu_V created\n");

float[] result = cu_A.mldivide(cu_V); //method should be renamed according to matrix type

System.out.println("A\\v =");
//printVector(result);

float[][] Y = vector2matrix(result, 20,100);
writeMatrixToFile(directory+ "Result.txt", Y);

cu_A.free();
cu_V.free();
cusparseDestroy(handle);

	}
	
	static private void printMatrix(float[][] A, int m, int n){

		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++)
				System.out.print(A[i][j]+" ");
			
			System.out.println();
		}		
	}
	
	
	static private void printVector(float[] V){

		for(int i=0;i<V.length;i++)
				System.out.print(V[i]+" ");
		
			System.out.println();
		
				
	}
	
//use generics to change type, allow for not only square matrices
	static private float[][] vector2matrix(float [] V, int m, int n){
		
		//System.out.println("size of V "+V.length);

		float [][] M = new float[m][n];
		for(int i=0, c=0;i<m;i++)
			for(int j=0;j<n;j++){
				System.out.println("i "+i+" j "+j);
				M[i][j] = V[c++]; //use smarter ArrayCopy M[i]= ...
			}
		return M;
	}
	
	
	static private void writeMatrixToFile(String filename, float [][] M){
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(filename))) {
			//System.out.println("M.length "+M.length+" M[0].length "+M[0].length);
			
			for(int i=0, c=0;i<M.length;i++){
				for(int j=0;j<M[i].length;j++){
					bw.write(Float.toString(M[i][j])); //use smarter ArrayCopy M[i]= ...
					bw.write("\t");
				}
				bw.write("\n");	
			}
			
			
			// no need to close it.
			bw.close();

			System.out.println("Done writing file");

		} catch (IOException e) {

			e.printStackTrace();

		}	
		
	
	}

	static private float[][] readSquareMatrixFromFile(String filename, int[] m) throws IOException, ParseException {

		float [][] A = null;
		
		File file = new File(filename);
		BufferedReader reader = null;
		
		
		System.out.println("file to path"+file.toPath());

		try {
		  reader = Files.newBufferedReader(file.toPath());
		  String line = null;

		  int l=0;
		  while ((line = reader.readLine()) != null) {

		    String delims = "[\t]";
		    String[] tokens = line.split(delims);
		    
		    if(A==null) {A = new float[tokens.length][tokens.length];
		    m[0]=tokens.length;
		    }
		    
		    //A[l]=new float[tokens.length];
		    
		    for (int i=0;i<tokens.length;i++)
		    A[l][i] = Float.parseFloat(tokens[i]);
		    
		    ++l;
		  }
		} catch (IOException x) {
		  System.err.format("IOException: %s%n", x);
		} finally {
		  if (reader != null)
		    reader.close();
		}
		return A;

		
				
	}
	
	

	static private float[] readVectorFromFile(String filename) throws IOException, ParseException {

		 
		
		ArrayList<Float> arraylist = new ArrayList<>();
		
		File file = new File(filename);
		BufferedReader reader = null;

		float[] V = null;
		try {
		  reader = Files.newBufferedReader(file.toPath());
		  String line = null;


		  while ((line = reader.readLine()) != null) {

		    String delims = "[\t]";
		    String[] tokens = line.split(delims);
		    
		    arraylist.add(Float.parseFloat(tokens[0]));

		  }
		} catch (IOException x) {
		  System.err.format("IOException: %s%n", x);
		} finally {
		  if (reader != null){
		    reader.close();
		    
		    System.out.println("arraylist size "+arraylist.size());
		    V  = new float[arraylist.size()];
		    {
		    int i=0;
		    for (Float f: arraylist)
		    	V[i++]=f;
		    
		    
		    
		    }
		    
		    System.out.println("V[0] "+V[0]);
		    
		    }
		    
		    
		  }
		
		return V;

		
				
	}
	

}
