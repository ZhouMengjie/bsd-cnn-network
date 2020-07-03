# bsd-cnn-network
This work is an implementation of paper "Automated Map Reading: Image Based Localisation in 2-D Maps Using Binary Semantic Descriptors". 
Please cite 


fine-tuneing CNN models to get Binary Semantic Descriptors(BSDs) of Google Street View (GSV) images.
> csv : csv files including the PanoID, yaw and BSD lables from OpenStreetMap (OSM).
trainstreetlearn.csv is used to generate training set.
hudsonriver5k.csv is used to generate validation set.
unionsquare5k.csv and wallstreet5k.csv is used to generate testing sets.

> crop_images : prepare images for training, validation and testing (need to utilize csv files in data/ ).

> network: train, evaluate and visualize BSD-networks 
data: GSV images (Junctions and Gaps)
train_codes: including all scripts to train networks and extract BSDs from images.
evaluate_codes: including all scripts to evaluate models (accuracy, precision, recall, F1, loss) and plot PR/ROC curves.
visualize_codes: including all scripts to generate feature maps, grad-cams, t-sne, and occlusion maps. (used to know what networks have learned).
alexnet/resnet18/vgg/resnet50/densenet161/googlenet: extracted BSD features (.mat files).
runs: training log.
curves: PR and ROC curves.
feature_maps/Grad_CAM/Occulusion: visualization results.
