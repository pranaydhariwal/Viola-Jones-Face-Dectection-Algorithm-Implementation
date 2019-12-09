# Viola-Jones-Face-Dectection-Algorithm-Implementation

This code implements the Viola Jones algorithm mentioned in the paper: "Rapid Object Detection using a Boosted Cascade of Simple Features"

There are 4 main parts:

I. Extracting Haar features for a given image. The Viola-Jones algorithm uses Haar-like features, that is, a scalar product between the image and some Haar-like templates.

II. Converting the image to an integral image for efficient computations.

III. Train adaboost detector on different rounds

IV. Train a cascade attention model mentioned in the paper.

Packages used: <br>
numpy <br>
scikit-learn <br>
PIL <br>
matplotlib <br>
pickle <br>

The training and testing data is converted into pkl file for saving some space. <br>

How to run:

python3 face_detection.py

