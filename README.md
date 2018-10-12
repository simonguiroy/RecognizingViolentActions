# RecognizingViolentActions

## Requirements:
- Python 3.6.4
- Numpy
- PyTorch 0.3
- Torchvision
- cv2
- scipy
- PIL.Image
- multiprocess

## Repository structure:

- *preprocessing.py:* The functions `preprocess_rgb` and `preprocess_flow` read a video file and respectively output a stack of RGB frames or a stack of Optical Flow frames.

- *video_dataset.py:* Creates a dataset object and by iterating throught it, returns a class label as well as the preprocessed video (output from either `preprocess_rgb` or `preprocess_flow`)

- *denseflow.py:* Used to compute optical flow from a video sequence, using the TV-L1 algorithm. Still being debugged. Will soon replace `preprocess_flow`. 

- *eval.py:* For a specified model and dataset, computes the prediction and outputs them as numpy array files in `/out/$model_name/`

- *train.py:* training script. Currently empty as it is being refactored.

- *analyse_results.ipynb:* Jupyter notebook used to produce statistics on a model's predictions, for a given dataset. Produces top1, top5 accuracy results, confusion matrix, both numerically and as pdf outputs.

- *models/:* The root of `models/` contains a python script defining each action recognition model (only i3d.py for now). There is a subdirectory for each model, which in turn contains the label map (text file) for the class logits of the model, as well as a subdirectory `checkpoints/` containing the pre-trained weights for that model (one for the RGB and Flow networks, in the case of i3d)

- *datasets/:* Where each dataset is tored and defined. The root of `datasets/` also contains useful bash scripts for resampling videos, counting videos, etc. Each dataset has a train, valid and test .csv file, as well as a `data` folder. That folder is divided into train, valid and test, and each of these subfolders are in turn divided into the various action classes, which in turn contain the video files.

- *utils/:* Various scripts useful for debugging.

- *out/:* Outputs of `eval.py` stored as Numpy array files.


