import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False, smooth_segmentation=True, detection_confidence=0.5, tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def find_pose(self, image, draw=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(image_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return image

    def get_landmarks(self, image):
        landmarks = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((id, cx, cy))
        return landmarks

    def draw_bounding_box(self, image, landmarks, draw=True):
        if landmarks:
            x_min = min([lm[1] for lm in landmarks])
            x_max = max([lm[1] for lm in landmarks])
            y_min = min([lm[2] for lm in landmarks])
            y_max = max([lm[2] for lm in landmarks])

            if draw:
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return image
