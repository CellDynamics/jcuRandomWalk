import ij.IJ;
import ij.ImagePlus;
import ij.io.FileSaver;
import ij.process.ImageProcessor;
import ij.ImageStack;
import ij.gui.StackWindow;

public class IJtiff {
	private ImagePlus imgPlus;
	private ImageProcessor imgProcessor;
	private ImageStack imgStack;
	int rows;
	int columns;
	int height;
	int bitDepth;
	byte[][][] stack;
	
//private IJ ij=new IJ(); 

public IJtiff(String filename) {
		
//	imgPlus = new ImagePlus(filename);
//	imgStack = imgPlus.getImageStack();
//	StackWindow imgWin = new StackWindow(imgPlus) ;

	
	
	imgPlus = IJ.openImage(filename); //second argument is n'th frame of tiff stack
	imgStack = imgPlus.getImageStack();
	rows=imgStack.getHeight();
 	columns=imgStack.getWidth(); 
	height=imgStack.getSize();
	System.out.println("filename = "+filename);
	System.out.println("rows = "+rows+", columns: "+columns+", height= "+height);
	bitDepth=imgStack.getBitDepth();
	System.out.println("bitDepth = "+bitDepth);
		//Object [] imgArray =  imgStack.getImageArray();
	//byte [] slice = (byte[]) imgArray[0];
	
	
	stack = new byte[rows][columns][height];
	
	if (imgPlus.getType() == ImagePlus.GRAY8) {  
		
		 for (int n=0; n<height; n++) {   
		imgProcessor = imgStack.getProcessor( n+1); //check that argument is frame number

	
	    byte[] pixels = (byte[])imgProcessor.getPixels();  
	    
	    
	    for (int x=0; x<rows; x++)   
	        for (int y=0; y<columns; y++) 
	            stack[x][y][n] = pixels[x * columns + y];  
	         
	}
	    
	    
	}  
	
	
	imgPlus.show();
	//StackWindow imgWin = new StackWindow(imgPlus) ;
	
	
	}





}
