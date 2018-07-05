/**
 * Scale opened image in steps up and down
 */

step = 0.1;
numSteps = 7; // on each direction
im = getImageID(); // to process
name = getInfo("image.filename");
dir = "/home/baniuk/external/rw/folder/scaled"; // output
IJ.log(dir + " " + name)
// -
for(i=1; i<=numSteps; i++) {
	selectImage(im);
	sc = 1 - i*step;
	run("Scale...", "x=&sc y=&sc z=1.0 interpolation=Bicubic fill average process create");
	saveName = name + "_" + d2s(sc,1) + ".tif";
	IJ.log(saveName);
	saveAs("Tiff", dir+"/"+saveName);
	close();
}

// -
for(i=1; i<=numSteps+3; i++) {
	selectImage(im);
	sc = 1 + i*step;
	run("Scale...", "x=&sc y=&sc z=1.0 interpolation=Bicubic fill average process create");
	saveName = name + "_" + d2s(sc,1) + ".tif";
	IJ.log(saveName);
	saveAs("Tiff", dir+"/"+saveName);
	close();
}
saveName = name + "_" + d2s(1.0,1) + ".tif";
saveAs("Tiff", dir+"/"+saveName);