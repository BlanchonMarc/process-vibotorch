# process-vibotorch

Generic pre and post processing on images. Here used for vibotorch input and output processing.

## Alignment
This Module contains methods (C++) to process alignment on multi-modal images.
#### Homographyv0
First version of Homography Based Alignment.
#### Homography
Final version of Homography Based Alignment.
#### Dense Alignement
First version of Homography + Dense Alignment (Vision + Visual Servoing).

## Conversion
This Module contains methods (Python) to process polarimetry data in order to extract from each images 3 images, AoP (Angle of Polarization), DoP (Degree of Polarization) and Intensity.
Another possible transformation is the combination of the three previous extraction into an HSL(2*AoP , DoP, Intensity).

### Estimation
This Module contains methods (Matlab) to process data after a process.
#### calError
Predictor of neessary components for F1 Score.
#### estimated
Estimation and Visualization of Accuracy per Classes for segmentation task.
#### graphcsv
Plot graph from csv.
#### grayandcrop
Convert to GrayScale and crop.
#### hitRates
Calculate Precision per pixel.
#### normalizationparam
Calculate normalization parameters (mean per channel, standard deviation per channel), in order to ensure a map corresponding to 0 -> 1 = 0 -> 255.
#### readNPY - readNPYheader
Library from [here](https://github.com/kwikteam/npy-matlab).
#### seefreq
Draw graph of frequencies of cameres.
#### visualize
Visualize the output of a Deep Learning Network.


## Cameras Parameters
This Folder contrain cameras parameters, intrinsics and extrinsics.
