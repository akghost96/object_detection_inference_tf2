import argparse
import tensorflow as tf
from object_detection.utils import label_map_util, visualization_utils, config_util
from object_detection.builders import model_builder
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='Detection module')
parser.add_argument("--image_path",help='path of the image for runing inference',default='/home/akghost96/Desktop/Work/DPL/master_disertatie/grapes_images/train/Grape___healthy/0ac4ff49-7fbf-4644-98a4-4dc596e2fa87___Mt.N.V_HL 9004_180deg.JPG', type=str)
parser.add_argument("--model_ckpt", help="add ckpt of the network you would like to do inference",default='/home/akghost96/Desktop/Work/DPL/object_detection/centernet_hg104_512x512_coco17_tpu-8/ckpt-11',type=str)
parser.add_argument("--model_config", help="add config of the network you would like to do inference",default='/home/akghost96/Desktop/Work/DPL/object_detection/centernet_hg104_512x512_coco17_tpu-8/pipeline.config',type=str)
parser.add_argument("--image_imageout",help='path of the output image default mode is jpg',default='output.jpg', type=str)
parser.add_argument("--label_map", help="use the pbtxt label map form your dataset",default='/home/akghost96/Desktop/Work/DPL/object_detection/black_rootnew/train/Disease_label_map.pbtxt',type=str)

arg = parser.parse_args()

def image_prediction(path_config=arg.model_config,path_ckpt=arg.model_ckpt,path_label=arg.label_map,path_image=arg.image_path, path_out_image=arg.image_imageout):

    config = config_util.get_configs_from_pipeline_file(path_config)
    detection_model = model_builder.build(model_config=config['model'], is_training=False)

    ckpt = tf.train.Checkpoint(model=detection_model)
    ckpt.restore(path_ckpt).expect_partial()

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        preditction_dict = detection_model.predict(image,shapes)
        detections = detection_model.postprocess(preditction_dict,shapes)
        return detections

    category_index = label_map_util.create_category_index_from_labelmap(path_label)
    Image_path = path_image

    img = cv.imread(Image_path)
    image_np = np.array(img)

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
                min_score_thresh=.5,
                agnostic_mode=False)
    plt.axis('off')
    plt.imshow(cv.cvtColor(image_np_with_detections, cv.COLOR_BGR2RGB))
    #plt.show()
    plt.savefig(path_out_image,dpi=300)

if __name__ == "__main__":
    image_prediction()