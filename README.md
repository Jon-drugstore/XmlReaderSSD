# XmlReaderSSD
Train your SSD directly from xml files without converting to lmdb 

Software was tested on Ubuntu 16.04 with g++ 5.4.0

First of all download caffe-SSD from https://github.com/weiliu89/caffe/tree/ssd, then MobileSSD https://github.com/chuanqi305/MobileNet-SSD. 

# Build

After folowing steps above, replace  PATHtoCAFFE/caffe/src/caffe/util/io.cpp with src/io.cpp, PATHtoCAFFE/caffe/include/util/io.hpp with include/io.hpp and PATHtoCAFFE/caffe/src/caffe/proto/caffe.proto with src/caffe.proto. Then move src/my_annotated_data_layer.cpp to PATHtoCAFFE/caffe/src/caffe/layers and include/my_annotated_data_layer.hpp to PATHtoCAFFE/caffe/include/caffe/layers.
Next, folow weiliu89 README or if caffe-ssd already have built run: make -j8 in caffe dir.

# Examples

Notice that labelmap.prototxt without background label. Make two train folders and two test folders with images and xml files respectively. In order to train network replace path in train and test files in annotated_data_param xmlDir(path to xml directory) and imageDir(path to image folder) with train and test folders. Also, replace layer type from "AnnotatedData" to "MyAnnotatedData" on fourth line in test and train prototxt. 
