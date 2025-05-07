import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import sys
import os
from geometry_msgs.msg import Pose2D
sys.path.insert(0, '/root/tracking_ws/devel/lib/python3/dist-packages')
from cv_tracker.msg import BoundingBox, Target

# Добавляем путь к py_utils и coco_utils (если потребуется)
sys.path.append('/root/vision_ros1/src/eagle_eye_vision/scripts')
from eagle_eye_vision.coco_utils import COCO_test_helper


model_path = '/root/tracking_ws/src/cv_tracker/models/yolo11n.rknn'
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)

CLASSES = (
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
)

def postprocess_yolo_onnx(output, obj_thresh=OBJ_THRESH, nms_thresh=NMS_THRESH):
    # output: (1, 84, 8400)
    output = np.squeeze(output)  # (84, 8400)
    boxes = output[:4, :]        # (4, 8400)
    scores = output[4:, :]       # (80, 8400)

    # Преобразуем боксы из xywh в xyxy
    x, y, w, h = boxes
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)  # (8400, 4)

    # Для каждого бокса — максимальный класс и его score
    class_ids = np.argmax(scores, axis=0)
    class_scores = np.max(scores, axis=0)

    # Фильтруем по порогу
    mask = class_scores > obj_thresh
    boxes_xyxy = boxes_xyxy[mask]
    class_ids = class_ids[mask]
    class_scores = class_scores[mask]

    # NMS (только для людей)
    person_mask = class_ids == 0
    boxes_xyxy = boxes_xyxy[person_mask]
    class_scores = class_scores[person_mask]
    class_ids = class_ids[person_mask]

    if len(boxes_xyxy) == 0:
        return None, None, None

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xyxy.tolist(),
        scores=class_scores.tolist(),
        score_threshold=obj_thresh,
        nms_threshold=nms_thresh
    )
    if len(indices) == 0:
        return None, None, None
    indices = indices.flatten()
    return boxes_xyxy[indices], class_ids[indices], class_scores[indices]

class YOLO11_ROS:
    def __init__(self):
        rospy.init_node('yolo11_ros', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.image_callback)
        self.pub = rospy.Publisher('/yolo/detections', Target, queue_size=1)
        from cv_tracker.rknn_executor import RKNN_model_container
        self.model = RKNN_model_container(model_path, target="rk3588")
        self.co_helper = COCO_test_helper(enable_letter_box=True)

    def image_callback(self, msg):
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Ошибка декодирования изображения: {e}")
            return
        img_height, img_width = img.shape[:2]
        img_pre = self.co_helper.letter_box(img.copy(), IMG_SIZE, pad_color=(0,0,0))
        img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_pre, axis=0)
        outputs = self.model.run([input_data])
        boxes, classes, scores = postprocess_yolo_onnx(outputs[0])
        msg_out = Target()
        msg_out.image_height = float(img_height)
        msg_out.image_width = float(img_width)
        if boxes is not None:
            real_boxes = self.co_helper.get_real_box(boxes)
            for box, score, cl in zip(real_boxes, scores, classes):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                size_x = abs(x2 - x1)
                size_y = abs(y2 - y1)
                area = size_x * size_y
                bbox = BoundingBox()
                bbox.center = Pose2D(x=center_x, y=center_y, theta=0.0)
                bbox.area = float(area)
                bbox.size_x = float(size_x)
                bbox.size_y = float(size_y)
                bbox.name = CLASSES[0]  # Только люди
                msg_out.boxes.append(bbox)
                print(f"box: {bbox.center.x}, {bbox.center.y}, {bbox.size_x}, {bbox.size_y}, {bbox.name}")
        self.pub.publish(msg_out)

if __name__ == '__main__':
    try:
        node = YOLO11_ROS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 