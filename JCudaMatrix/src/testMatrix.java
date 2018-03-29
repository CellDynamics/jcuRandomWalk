import jcuda.jcusparse.cusparseHandle;
import static jcuda.jcusparse.JCusparse.*;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.stream.Stream;


public class testMatrix {

//test git
	public static void main(String[] args) throws IOException, ParseException{


		// What follows are three examples:
		// a) solving a triangular system A*x=b,
		// b) solving a symmetric system using LU decomposition, and
		// c) solving a bigger asymmetric system using incomplete LU factorisation
		// which provides starting values for a Bi-Conjugate
		// Gradient Stabilized (BiCGStab) iterative solver.
		// I M P O R T A N T
		// Note 1: We only consider square matrices A with size m
		// Note 2: So far we consider floats only

		int [] m = new int[1]; //size of matrix


		///////////////////////////////////////////////////

		// a)
		// create lower triangular test matrix
		// this will work fine with the mldivide() 

		//float[][] A={{1.f,0.f,0.f,0.f},{2.f,3.f,0.f,0.f},{4.f,5.f,6.f,0.f},{7.f,8.f,9.f,10.f}};
		// create rhs vector so that A*[5,2,6,3]=b
		//float[] b={5.f, 16.f, 66.f, 135.f};
		//m[0]=4;

		// example for printing inputs
		//System.out.println("matrix A");
		//printMatrix(A);
		//System.out.println("rhs vector b");
		//printVector(b);

		///////////////////////////////////////////////////

		// b)
		// symmetric works fine with LUsolve
		//float[][] A={{1.f,7.f,3.f},{7.f,4.f,-5.f},{3.f,-5.f,6.f}};
		// create rhs vector so that A*[1,2,3]=b
		//float[] b={24.f, 0.f, 11.f};
		//m[0]=3;

		// solution for simple unsymmetric case looks alright with just LUsolve, however bigger systems don't work
		//float[][] A={{1.f,7.f,5.f},{7.f,4.f,-5.f},{3.f,-5.f,6.f}};
		//create rhs vector so that A*[1,2,3]=b
		//float[] b={30.f, 0.f, 11.f};
		//m[0]=3;

		///////////////////////////////////////////////////

		// c) read in Laplacian and rhs vector for a larger system
		String directory = "./data/"; 
		String path=Paths.get(".").toAbsolutePath().normalize().toString();
		System.out.println("path to data directory"+path);
		//		int rows=20; int columns=100; int height=1; //dimension of original image/3D stack
		//
		//		float[][] A = readSquareMatrixFromFile(directory+"Lap.txt", m );
		//		float[] b = readVectorFromFile(directory+"B.txt");


		//+ example of a reduced Laplacian as input
		int rows=30; int columns=100; int height=1; //dimension of original image/3D stack
		float[][] A = readSquareMatrixFromFile(directory+"L_reduced.txt", m );
		float[] b = readVectorFromFile(directory+"B_reduced.txt");
		float[][] Seeds = readSeedsFromFile(directory+"Seeds_reduced.txt");		
		//printMatrix(Seeds);



		//set up cusparse handle, TODO: learn how to address one specific GPU, can use multiple GPUs using different handles
		cusparseHandle handle = new cusparseHandle();
		// Initialize JCusparse library
		cusparseCreate(handle);
		// Initialise JCublas library (does not seem to be needed)
		//JCublas.cublasInit();


		csrSparseMatrix  cu_A = new  csrSparseMatrix(handle, A, m[0], m[0]);
		System.out.print("sparse matrix cu_A created\n");

		denseVector  cu_b = new  denseVector(b);
		System.out.print("dense rhs vector cu_b created\n");

		//float[] x = cu_A.mldivide(cu_V); //use for examples a)
		//float[] x = cu_A.LuSolve(cu_b); //use for example b)
		float[] x = cu_A.LuSolve(cu_b, true); //use for example c), second argument switches iLuBiCGStabSolve mode on

		// if using a reduced Laplacian
		x = incorporateSeeds(x,Seeds );



		// for a), b)
		//System.out.println("A\\b =");
		//printVector(x);

		// for c)
		float[][] x2D = vector2matrix(x, rows,columns); //format x as 2D matrix (solution image)
		writeMatrixToFile(directory+ "Result.txt", x2D);
		// output file can be plotted with
		// gnuplot:  plot "Result.txt" matrix with image

		// clear up
		cu_A.free();
		cu_b.free();
		cusparseDestroy(handle);

	}

	static private void printMatrix(float[][] A){

		for(float[] row: A){
			for(float f: row)
				System.out.print(f+" ");

			System.out.println();
		}

	}


	static private void  printVector(float[] V){

		for(float f: V)
			System.out.print(f+" ");

		System.out.println();

	}


	static private float[][] vector2matrix(float [] V, int m, int n){

		float [][] M = new float[m][n];

		for(int i=0, c=0;i<m;i++)
			for(int j=0;j<n;j++)
				M[i][j] = V[c++];

		return M;
	}


	static private void writeMatrixToFile(String filename, float [][] A){

		try (BufferedWriter bw = new BufferedWriter(new FileWriter(filename))) {

			for(float[] row: A){
				for(float f: row){
					if(f==-0) //not required, but negative zeros might cause problems when importing
						f=0;
					bw.write(Float.toString(f));
					bw.write("\t");
				}
				bw.write("\n");	
			}		


			bw.close();

			System.out.println("Done writing results file");

		} catch (IOException e) {

			e.printStackTrace();

		}	


	}


	static private float[][] readSquareMatrixFromFile(String filename, int[] m) throws IOException, ParseException {

		float [][] A = null;

		File file = new File(filename);
		BufferedReader reader = null;

		System.out.println("reading in Array file: "+file.toPath());

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


	static private float[][] readSeedsFromFile( String filename)throws IOException, ParseException {

		float [][] A = null;
		int numOfSeeds;

		File file = new File(filename);
		BufferedReader reader = null;

		System.out.println("reading in Seeds file: "+file.toPath());

		try (Stream<String> lines = Files.lines(file.toPath(), Charset.defaultCharset())) {
			numOfSeeds = (int) lines.count();
		}

		A = new float[numOfSeeds][2];
		//System.out.println("number of Seeds "+numOfSeeds);

		try {
			reader = Files.newBufferedReader(file.toPath());
			String line = null;

			int l=0;
			while ((line = reader.readLine()) != null) {

				String delims = "[\t]";
				String[] tokens = line.split(delims);

				A[l][0] = Float.parseFloat(tokens[0]) - 1;//correct for Matlab indexing
				A[l][1] = Float.parseFloat(tokens[1]);

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


	static private float[] incorporateSeeds(float[] x,float[][] Seeds ){

		// method seems to do the job, but might require some more testinga

		float[] x_withSeeds = new float[x.length+Seeds.length];

		// need to sort lines of Seeds array first, according to entries in first column
		java.util.Arrays.sort(Seeds, new Comparator<float[]>() {
			@Override
			public int compare(float[] o1, float[] o2) {
				return Float.compare(o1[0], o2[0]);
			}
		});


		//printMatrix(Seeds);

		int countwith=0,count=0;
		int S=0,nonseeds;
		for (int s=0;s<Seeds.length;s++){
			nonseeds=(int) ((Seeds[s][0])-S-1); 
			for (int i=0;i<nonseeds;i++) 
				x_withSeeds[countwith++]=x[count++];


			x_withSeeds[countwith++]=Seeds[s][1];

			S=(int) Seeds[s][0];
		}

		return x_withSeeds;
	}


	static private float[] readVectorFromFile(String filename) throws IOException, ParseException {

		ArrayList<Float> arraylist = new ArrayList<>();

		File file = new File(filename);
		BufferedReader reader = null;

		System.out.println("reading in Vector file: "+file.toPath());

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

				//System.out.println("Vector size "+arraylist.size());
				V  = new float[arraylist.size()];
				{
					int i=0;
					for (Float f: arraylist)
						V[i++]=f;
				}

			}


		}

		return V;

	}


}
