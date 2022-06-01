import argparse
import tensorflow as tf
from object_detection.utils import label_map_util, visualization_utils, config_util
from object_detection.builders import model_builder
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description="Video inference moduel this will take one video and run ifference on that video with the selected and trained model (it will alos display in real time the ifference.)")
parser.add_argument("--file_path",help="path of the video file for the detection", type=str)
parser.add_argument("--model_ckpt",help="use this to add the ckpt you want to use for ifference", type=str)
parser.add_argument("--model_config",help="the model config file of the model you loaded from ckt", type=str)
parser.add_argument("--label_map",help="the label map used for training the model.",type=str)
parser.add_argument("--video_out",help="the output file",default="infference_video.mp4",type=str)
arg = parser.parse_args()

config = config_util.get_configs_from_pipeline_file(arg.model_config)
detection_model = model_builder.build(model_config=config['model'], is_training=False)

ckpt = tf.train.Checkpoint(model=detection_model)
ckpt.restore(arg.model_ckpt).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    preditction_dict = detection_model.predict(image,shapes)
    detections = detection_model.postprocess(preditction_dict,shapes)
    return detections


category_index = label_map_util.create_category_index_from_labelmap(arg.label_map)

cap = cv.VideoCapture(arg.file_path)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fourtcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter(arg.video_out,fourtcc,30.0,(width,height))

while cap.isOpened(): 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    visualization_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=10,
                min_score_thresh=.3,
                agnostic_mode=False)
    videoout_frame = cv.resize(image_np_with_detections,(width,height))
    out.write(videoout_frame)
    cv.imshow('object detection', image_np_with_detections)

    if cv.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
out.release()