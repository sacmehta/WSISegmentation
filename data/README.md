## Dataset
Please keep your dataset inside data folder. 

---

In our experiments, we followed the following structure

-- data

---- mel

---- train.txt

---- val.txt

------ trainrgb

------ trainannot

------ testrgb

------ testannot

The text files (train.txt and val.txt) contains the names of training and validation files. Each of these files stores the image file location in the following format
```
/trainrgb/<image_file1>.png,/trainrgb/<image_file1>.png
/trainrgb/<image_file2>.png,/trainrgb/<image_file2>.png
```

RGB images are contained in trainrgb and testrgb folders while their corresponing label images are contained in trainannot and testannot folders.

**NOTE:** Please note that torch doesn't except a label value of 0. If your dataset has a label with a value of 0 (usually indicating the background area), then shift the label values by 1. This can be done by setting the value of variable **labelAddVal** to **1** in file **loadMel.lua**. 
