# Installing Packages

All the packages required for the script can be installed by running the following command:
`pip install -r requirements.txt`

# Running the Script

`distortion.py` reads in an image and corrects the radial distortion caused by camera lens and close distance as well as the tangential distortion caused by angled perspective. 

In order to correct radio distortion, camera calibration must be done first. Images of the checkerboard should be placed within a folder named 'calibration' and should be saved as .png files. Make sure that images of the checkerboard are taken under the same focal distance as when the to-be-corrected image is taken.

Run `python distortion.py image.png a b` to correct distortion. `a` should be `1` if correcting radio distortion is desired and `0` otherwise. `b` should be `1` if correcting tangential distortion is desired and `0` otherwise.

For example, if I have an image of an object and I only wnat to correct tangential distortion, I would run `python distortion.py image.png 0 1`. If I want to correct both, then run `python distortion.py image.png 1 1`.

Notice that, for the purpose of testing, images are resized in a hardcoding manner. If this script is used to process images that are not taken by the Arducam Motorized Focus Camera, please remove the `cv2.resize()` function calls in the main function.

The original image, as well as a fixed image, will be displayed after the script finishes runing.