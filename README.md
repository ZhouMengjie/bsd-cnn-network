# bsd-cnn-network: fine-tuneing CNN models to get Binary Semantic Descriptors(BSDs) of Google Street View (GSV) images

This work is an implementation of paper "Automated Map Reading: Image Based Localisation in 2-D Maps Using Binary Semantic Descriptors". 
The codes and features included here are employed as the baseline for conducting a comparative analysis in the paper titled "You Are Here: Geolocation by Embedding Maps and Images."
Please cite in this form when you use these codes.


> @InProceedings{Panphattarasap2018,
  Title = {Automated map reading: Image based localisation in 2-d maps using binary semantic descriptors},
  Author = {Pilailuck Panphattarasap and Andrew Calway},
  Booktitle = {Proc. {IEEE/RSJ} Int Conf on Intelligent Robots and Systems},
  Year = {2018}}

> @inproceedings{samano2020you,
  title={You are here: Geolocation by embedding maps and images},
  author={Samano, Noe and Zhou, Mengjie and Calway, Andrew},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part XXIII 16},
  pages={502--518},
  year={2020},
  organization={Springer}
}

Here is a brief introduction about the codes. 

> csv : csv files including the PanoID, yaw and BSD lables from OpenStreetMap (OSM).

        >> trainstreetlearn.csv is used to generate training set.

        >> hudsonriver5k.csv is used to generate validation set.

        >> unionsquare5k.csv and wallstreet5k.csv is used to generate testing sets.

> crop_images : prepare images for training, validation and testing (need to utilize csv files in data/ ).

> network: train, evaluate and visualize BSD-networks.

        >> data: GSV images (Junctions and Gaps)

        >> train_codes: including all scripts to train networks and extract BSDs from images.

        >> evaluate_codes: including all scripts to evaluate models (accuracy, precision, recall, F1, loss) and plot PR/ROC curves.

        >> visualize_codes: including all scripts to generate feature maps, grad-cams, t-sne, and occlusion maps. (used to know what networks have learned).

        >> alexnet/resnet18/vgg/resnet50/densenet161/googlenet: extracted BSD features (.mat files).

        >> runs: training log.

        >> curves: PR and ROC curves.

        >> feature_maps/Grad_CAM/Occulusion: visualization results.

Please contact me (mengjie.zhou@bristol.ac.uk) if you have any questions.
