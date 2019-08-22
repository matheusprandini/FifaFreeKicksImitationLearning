import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time
from utils import label_map_util
from utils import visualization_utils as vis_util
from FifaEnv import FifaEnv
from Keys import Keys
from PIL import Image

class CNN_V2_With_Barrier(object):
    """
    This class acts as the intermediate "API" to the actual game. Double quotes API because we are not touching the
    game's actual code. It interacts with the game simply using screen-grab (input) and keypress simulation (output)
    using some clever python libraries.
    """
    # What model to download.
    MODEL_NAME = 'fifa_graph_chintan_v2_with_barrier'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = MODEL_NAME + '/object-detection.pbtxt'

    NUM_CLASSES = 4

    detection_graph = tf.Graph()

    def __init__(self):
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        print(label_map)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    ## Run the model for the given image
    def run_inference_for_single_image(self, image):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    ## Saves a labeled image (generating the bounding boxes) from the given image
    def generate_labeled_image(self, image, number):

        def load_image_into_numpy_array(image):
            (im_width, im_height) = image.size
            return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

        # Size, in inches, of the output images.
        IMAGE_SIZE = (30, 22)

        image_path = 'image' + str(number) + '.png'

        ## Saving and loading image
        cv2.imwrite(image_path, image)
        image = Image.open(image_path)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = self.run_inference_for_single_image(image_np)

        print(output_dict)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        plt.savefig("image_labeled" + str(number))

    ## Extracts a 128 vector corresponding to the object desired to detection from the given image
    def get_image_feature_map(self, image):
        start = time.time()
        image = image[40:850,0:1400]
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                feature_vector = self.detection_graph.get_tensor_by_name(
                    "FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_5_3x3_s2_128/Relu6:0")

                image_np = cv2.resize(image, (900, 400))
                image_np_expanded = np.expand_dims(image_np, axis=0)
                rep = sess.run([feature_vector], feed_dict={image_tensor: image_np_expanded})
                return np.array(rep).reshape(-1, 128), image

    ## Validates the object detection (check if the images labeled are correct)
    def test_object_detection(self):

        game_env = FifaEnv()
        keys = Keys()
        paused = True
        cont=0

        while True:
            
            if not paused:
                
                # get the current state
                x_t = game_env.observe_state()[40:850,0:1400]

                self.generate_labeled_image(x_t, cont)

                cont+=1
                paused = True
                print('Pausing!')
            
            keys_pressed = keys.KeyCheck()
            
            if 'Q' in keys_pressed:
                if paused:
                    paused = False
                    print('unpaused!')
                    time.sleep(1)

if __name__ == '__main__':
    print('aa')
    od_model = CNN_V2_With_Barrier()
    od_model.test_object_detection()
