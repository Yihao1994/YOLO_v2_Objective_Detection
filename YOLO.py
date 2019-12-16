# In[YOLO Applying]
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import cv2
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
import time


#Score filter
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold):
    
    #(19x19, 5, 1) * (19x19, 5, 80)
    box_scores = box_confidence*box_class_probs
    
    #Find the maximum from the last axis (the classes one)
    box_classes = K.argmax(box_scores, axis = -1)     # Return the index
    box_class_scores = K.max(box_scores, axis = -1)   # Return the max value
    
    #Judge if the 5 anchor boxess in each cell over the threshold? 
    filtering_mask = box_class_scores >= threshold
    
    #Kill the boxes with low score
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes



#Non-max suppression filter
'''
#Calculate the IoU at first.
def IoU(box1, box2):
    
    #Load the upper left and down right points of the boxes
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    
    #Find the top left and down right coordinate for shadow
    xi1 = max(box1_x1, box2_x1)   # top left
    yi1 = max(box1_y1, box2_y1)   # top left
    xi2 = min(box1_x2, box2_x2)   # down right
    yi2 = min(box1_y2, box2_y2)   # down right
    
    inter_width  = xi2 - xi1
    inter_height = yi2 - yi1
    
    #Calcuate the intersaction
    if inter_width<0 or inter_height<0:
        inter_area = 0
    else:
        inter_area = inter_height*inter_width
        
    box1_area = (box1_x2 - box1_x1)*(box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1)*(box2_y2 - box2_y1)
    union = box1_area + box2_area - inter_area
    
    IoU = inter_area/union
    
    return IoU
'''


def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 20, iou_threshold = 0.5):
    max_boxes_tensor = K.variable(max_boxes, dtype = 'int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    
    #Filter the index
    indices_save = tf.image.non_max_suppression(boxes, scores, max_output_size = \
                                                max_boxes, iou_threshold = 0.5)
    
    
    #Saving the 'correct' boxes vbasing on the indices_save
    scores  = K.gather(scores, indices_save)
    boxes   = K.gather(boxes, indices_save)
    classes = K.gather(classes, indices_save)
    
    return scores, boxes, classes
    
    

def yolo_eval(yolo_outputs, image_shape, max_boxes = 20, score_threshold = 0.58, iou_threshold = 0.5):
    
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    
    boxes = scale_boxes(boxes, image_shape)
    
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = 20, iou_threshold = 0.5)
    
    return scores, boxes, classes



def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the predictions.
    
    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess your image
    #image, image_data = preprocess_image("images\road1.jpg", model_image_size = (608, 608))
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    out_scores, out_boxes, out_classes = sess.run(fetches = [scores, boxes, classes], \
                                                  feed_dict = {yolo_model.input:image_data, K.learning_phase():0})

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    
    # Display the results in the notebook
    output_image_raw = cv2.imread(os.path.join("out", image_file))
    output_image = output_image_raw[...,::-1]
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes



sess = K.get_session()



#Model loading
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
yolo_model = load_model("model_data/yolo.h5")
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))


# In[Pic loading & Testing]
#Choose image
tic = time.clock()
image_file = "test21.jpg"
image_original = cv2.imread("images/" + image_file)
height = image_original.shape[0]
width = image_original.shape[1]
image_shape = (float(height), float(width))

#Apply to the YOLO filter and non-max suppression
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

#Detection
print('')
print('#######################################')
print('Objective Detection for the ' + image_file)
out_scores, out_boxes, out_classes = predict(sess, image_file)
toc = time.clock()
print('Spending time:', (toc - tic), 'sec')


