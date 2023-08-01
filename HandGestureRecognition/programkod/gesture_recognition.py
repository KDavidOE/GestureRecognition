from os.path import exists
import cv2
import segmentation
import classification
import numpy as np


class GestureRecognition:
    def __init__(self):
        self.frame_to_pass = None
        self.command = None
        self.outputs_changed = None
        self.segmentation_obj = segmentation.Segmentation()
        self.classification_obj = classification.Classification()
        self.gesture_seq = 0
        self.last_command = None
        self.video_capture = cv2.VideoCapture(0)
        self.predict_dim = 64
        self.model_present = False
        self.initialized = False

    def init_models(self, prediction_model_name, generator_model_name):
        predictor_exists = exists(prediction_model_name)
        generator_exists = exists(generator_model_name)
        if not (predictor_exists and generator_exists):
            self.model_present = False
            return False

        if not self.initialized:
            self.classification_obj.init_prediction_modul(prediction_model_name, generator_model_name)
            self.initialized = True
            return True

    def close_processes(self):
        self.video_capture.release()

    def data_changed_event(self):
        self.outputs_changed()

    def subscribe_for_output_change(self, method_reference):
        self.outputs_changed = method_reference

    def unsubscribe_from_output_change(self):
        self.outputs_changed = None

    @staticmethod
    def command_converter(command):
        objects = ('Start', 'Stop', 'Back', 'Forward', 'VolumeUp', 'VolumeDown')
        return objects[command]

    def evaluate_gesture(self, mirrored, de_shadow):
        ret, previous_frame = self.video_capture.read()
        prev_pic = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        prev_pic = cv2.resize(prev_pic, (640, 480))
        ret, frame = self.video_capture.read()
        frame_resized = cv2.resize(frame, (640, 480))
        current_pic = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        optical_flow_mask = self.segmentation_obj.get_optical_flow(prev_pic, current_pic)
        skin_segmented_mask = self.segmentation_obj.get_roi_by_skin_color(frame)
        segmented_mask = cv2.bitwise_and(optical_flow_mask, skin_segmented_mask)
        self.segmentation_obj.get_contours(segmented_mask, 10)
        found_hand, possible_hand, bbox = self.segmentation_obj.detect_hand(frame_resized)

        if found_hand:
            image_to_convert = None
            if not de_shadow:
                image_to_convert = self.segmentation_obj.resize_to_prediction_shape(possible_hand,
                                                                          self.predict_dim, False)
            else:
                image_to_convert = self.segmentation_obj.resize_to_prediction_shape(possible_hand,
                                                                          self.predict_dim, True)
                normalized_hand = (image_to_convert.astype(np.float32) - 127.5) / 127.5
                p = self.classification_obj.de_shadow_image(np.expand_dims(normalized_hand, axis=0))
                gen_image = p[0]
                gen_image = (gen_image * 127.5) + 127.5
                image_to_convert = self.segmentation_obj.resize_to_prediction_shape(gen_image,
                                                                          self.predict_dim, False)

            if mirrored:
                image_to_convert = cv2.flip(image_to_convert, 1)

            array = self.classification_obj.convert_image_to_model_array(image_to_convert)
            result = self.classification_obj.predict_from_array(array)
            result = list(result[0])
            img_index = result.index(max(result))
            command = self.command_converter(img_index)
            self.command = command
            if self.last_command is None:
                self.last_command = command
            else:
                if self.last_command == command:
                    self.gesture_seq += 1
                else:
                    self.last_command = command
                    self.gesture_seq = 0

            if self.gesture_seq == 6:
                self.gesture_seq = 0

            found_hand = False
            image_to_display = cv2.resize(image_to_convert, (150, 150))
            image_to_display = cv2.imencode(".png", image_to_display)[1].tobytes()

            self.frame_to_pass = image_to_display
            self.data_changed_event()

        prev_pic = current_pic
