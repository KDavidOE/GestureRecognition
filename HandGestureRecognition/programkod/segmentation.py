import numpy as np
import cv2
import mediapipe as mp


class Segmentation:
    contours = None
    p_hand = None
    intersect_ratio = 0.5
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)

    def get_hand_from_mp(self, source_frame):
        img_converted = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(img_converted)
        detected_landmarks = results.multi_hand_landmarks
        bounding_boxes = []
        h, w, c = source_frame.shape
        if detected_landmarks:
            for hand_landmarks in detected_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y

                bbox = x_min, y_min, x_max - x_min, y_max - y_min
                bounding_boxes.append(bbox)
        return bounding_boxes

    @staticmethod
    def get_optical_flow(prev_pic, current_pic):
        hsv = np.zeros_like(cv2.cvtColor(prev_pic, cv2.COLOR_GRAY2BGR))
        hsv = np.zeros_like(hsv)
        hsv[..., 1] = 255
        prev_blurred = cv2.GaussianBlur(prev_pic, (7, 7), 0)
        next_blurred = cv2.GaussianBlur(current_pic, (7, 7), 0)
        prev_blurred = cv2.resize(prev_blurred, (640, 480))
        next_blurred = cv2.resize(next_blurred, (640, 480))
        flow = cv2.calcOpticalFlowFarneback(prev_blurred, next_blurred, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        lower_threshold = np.array([0, 0, 55], np.uint8)
        higher_threshold = np.array([255, 255, 255], np.uint8)
        hsv_img = hsv
        frame_threshed = cv2.inRange(hsv_img, lower_threshold, higher_threshold)
        optical_mask = frame_threshed
        return optical_mask

    @staticmethod
    def get_roi_by_skin_color(source_image):
        lower_threshold = np.array([0, 135, 85], np.uint8)
        higher_threshold = np.array([255, 180, 135], np.uint8)
        hsv_lower_threshold = np.array([0, 15, 0], np.uint8)
        hsv_higher_threshold = np.array([17, 170, 255], np.uint8)
        image_y_cr_cb = cv2.cvtColor(source_image, cv2.COLOR_BGR2YCR_CB)
        image_hsv = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)
        skin_segmented_y_cr_cb = cv2.inRange(image_y_cr_cb, lower_threshold, higher_threshold)
        skin_segmented_hsv = cv2.inRange(image_hsv, hsv_lower_threshold, hsv_higher_threshold)
        final_mask = cv2.bitwise_and(skin_segmented_y_cr_cb, skin_segmented_hsv)
        return final_mask

    @staticmethod
    def get_box_area(box):
        return box[2] * box[3]

    @staticmethod
    def intersect_area(a, b):
        dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
        dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        else:
            return 0

    @staticmethod
    def add_padding_to_image(img, padding_size, image_bbox):
        dimensions = img.shape
        x2, y2, w2, h2 = image_bbox
        if x2 < 0:
            x2 = 0
        if y2 < 0:
            y2 = 0
        padding_left = 0
        padding_right = 0
        padding_top = 0
        padding_bottom = 0
        if x2 - padding_size >= 0:
            padding_left = padding_size
        if x2 + padding_size < dimensions[1]:
            padding_right = padding_size
        if y2 - padding_size >= 0:
            padding_top = padding_size
        if y2 + padding_size < dimensions[0]:
            padding_bottom = padding_size
        img = img[y2 - padding_top: y2 + h2 + padding_bottom, x2 - padding_left: x2 + w2 + padding_right]
        return img

    def get_contours(self, segmented_mask, k):
        dist = cv2.distanceTransform(segmented_mask, cv2.DIST_L1, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        dist = cv2.dilate(dist, kernel)
        self.contours, _ = cv2.findContours(dist.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self.contours

    def detect_hand(self, original_pic):
        dimensions = original_pic.shape
        possible_hand = original_pic[0: dimensions[0], 0: dimensions[1]]

        bounding_boxes = self.get_hand_from_mp(possible_hand)
        roi_bbox = None
        index = -1
        for i, bbox in enumerate(bounding_boxes):
            for cnt in self.contours:
                intersect_area_ratio = self.intersect_area(
                    bbox, cv2.boundingRect(cnt)) / self.get_box_area(bbox)
                if intersect_area_ratio >= self.intersect_ratio and \
                        (index == -1 or self.get_box_area(bounding_boxes[index]) <= self.get_box_area(bbox)):
                    index = i
                    detected_hand = bounding_boxes[index]
                    roi_bbox = detected_hand

        if len(bounding_boxes) == 0 or index == -1:
            return False, possible_hand, roi_bbox

        possible_hand = self.add_padding_to_image(possible_hand, 20, roi_bbox)
        found_hand = True
        self.p_hand = possible_hand
        return found_hand, possible_hand, roi_bbox

    @staticmethod
    def resize_to_prediction_shape(image, prediction_model_size, regenerate):
        corrected_image = np.ones((prediction_model_size, prediction_model_size, 3), np.uint8) * 255
        ratio = image.shape[0] / image.shape[1]
        if ratio > 1:
            k = prediction_model_size / image.shape[0]
            calculated_width = round(k * image.shape[1])
            resized_image = cv2.resize(image, (calculated_width, prediction_model_size))
            width_difference = round((prediction_model_size - calculated_width) / 2)
            corrected_image[:, width_difference:calculated_width + width_difference] = resized_image
        else:
            k = prediction_model_size / image.shape[1]
            calculated_height = round(k * image.shape[0])
            resized_image = cv2.resize(image, (prediction_model_size, calculated_height))
            height_difference = round((prediction_model_size - calculated_height) / 2)
            corrected_image[height_difference:calculated_height + height_difference, :] = resized_image
        if not regenerate:
            possible_hand = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        else:
            possible_hand = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
        return possible_hand

