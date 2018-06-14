# Usage

There are the following combinations of options supported:

1. Options specified: `-i`, `-o`, `-s` - program expects that input image `-i` is 32-bit and scaled to range 0-1, seeds are given as white pixels of value 255 (`-s`) and result of the segmentation will be saved under name specified with `-o`.
2. No `-o` or `-s` specified - the same requirements as for all three options given but program assumes that seeds file exists in the same folder as input file (`-i`) and under name _inputimage_seeds.tif. If there is no `-o` specifed output will be saved in the folder of _inputimage_ under the name _inputimage_segm.tif
3. Option `--defaultprocessing` provided - input image can be any, it will be filtered by 3D median filter with mask size of 3 and converted to 32bit with scaling to the range 0-1
4. Option `-t` specified - input image will be thresholded with given threshold to obtain seeds. Threshold is a raw value above which pixels are considered as seeds.
5. If path specified in `-i` points to folder, all `.tif` images will be processed. In this case option `-o` is ignored and `-s` should not be used, otherwise each processed input file will use the same seeds and will be saved under the same name overriding previous segmentation.  
