# YOLO_v2_Objective_Detection
The implementation of YOLO_v2 algorithm for a 80-classes objective-detection system. Before going to the details inside, there are two important issues need to be mentioned here.  

The first thing is regarding to the **YOLO_v2** model data. Since the size of the pre-trained model data is already 194 MB, which is too large to be uploaded into Github. Therefore, I upload this pre-trained model data into a Google drive, whose link is [https://drive.google.com/drive/folders/1RWcXJdjwNJ57wL-XfWZiC-ermIlDc1_8?usp=sharing]. The name of the file is 'yolo.h5'. And according to the setting, this file should be put under the folder 'model_data'.

Then the second thing is about the operating environment. These scripts here were executing on the tensorflow-gpu-1.8.0. After installing the tensorflow-gpu-1.8.0, you have to activate tensorflow environment by pressing the 'activate tensorflow' into the Anaconda prompt at first, and then open the Spyder from the Anaconda prompt. Then so far the environment problem shall be fixed.  

The main file is the ___YOLO.py___, and the entire script shall be executable under the environment I mentioned before. The folder 'images' is where you can put your image inside, and the result of detection will be saved into the folder 'out'. Since the model was training under a resolution (608*608), so this is the reason why in line 129, that there is 'preprocess-image' function to help transfer the resolution of the image. Besiedes, the 'max_boxes', 'score_threshold' and 'iou_threshold' are the three hyperparameters that you can tune in the model to adjust its performance. Finally, since the entire model is already well-organised, so the only thing you need to do when you take a new image, is that changing the file name in line 166, and then you can just execute the scripts to see the result of detection.  

Enjoy !
