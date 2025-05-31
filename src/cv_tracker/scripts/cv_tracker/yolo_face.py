#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import os
import sys

# Добавляем путь к сгенерированным пакетам
sys.path.insert(0, '/root/tracking_ws/devel/lib/python3/dist-packages')

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from cv_tracker.msg import BoundingBox, Target
from geometry_msgs.msg import Pose2D


# Путь к модели и параметры
MODEL_PATH = '/root/tracking_ws/src/cv_tracker/models/yolov11n-face.rknn'
IMG_SIZE = (640, 640)
OBJ_THRESH = 0.25
NMS_THRESH = 0.45

def postprocess_yolo_face(output, obj_thresh=OBJ_THRESH, nms_thresh=NMS_THRESH):
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

    # Для каждого бокса берем максимальный score
    class_scores = np.max(scores, axis=0)

    # Фильтруем по порогу
    mask = class_scores > obj_thresh
    boxes_xyxy = boxes_xyxy[mask]
    class_scores = class_scores[mask]

    if len(boxes_xyxy) == 0:
        return None, None

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xyxy.tolist(),
        scores=class_scores.tolist(),
        score_threshold=obj_thresh,
        nms_threshold=nms_thresh
    )
    if len(indices) == 0:
        return None, None
    indices = indices.flatten()
    return boxes_xyxy[indices], class_scores[indices]

class YOLOFaceDetector:
    def __init__(self):
        # Инициализация ROS ноды
        rospy.init_node('yolo_face_detector', anonymous=True)
        rospy.loginfo("Инициализация YOLOFaceDetector...")
        
        # Параметры для подписки на изображение
        image_topic = rospy.get_param('~image_topic', '/camera/image_raw/compressed')
        self.image_sub = rospy.Subscriber(image_topic, CompressedImage, self.image_callback)
        
        # Publisher для обработанного изображения и результатов
        self.processed_image_pub = rospy.Publisher('/face_detection/image/compressed', CompressedImage, queue_size=10)
        self.target_pub = rospy.Publisher('/face_detection/targets', Target, queue_size=30)
        
        # Создание CvBridge
        self.bridge = CvBridge()

        # Импорт и инициализация RKNN модели
        from cv_tracker.rknn_executor import RKNN_model_container
        self.model = RKNN_model_container(MODEL_PATH, target="rk3588")
        
        
    def preprocess_image(self, image):
        """Предобработка изображения для YOLO"""
        img = cv2.resize(image, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.expand_dims(img, axis=0)
        
    def image_callback(self, msg):
        """Обработчик входящих изображений"""
        try:
            # Преобразование compressed сообщения в изображение OpenCV
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Получаем размеры исходного изображения
            height, width = cv_image.shape[:2]
            
            # Предобработка изображения
            input_data = self.preprocess_image(cv_image)
            
            # Запуск модели
            outputs = self.model.run([input_data])
            
            # Постобработка результатов
            boxes, scores = postprocess_yolo_face(outputs[0])
            
            # Создаем сообщение Target
            target_msg = Target()
            target_msg.image_height = height
            target_msg.image_width = width
            
            if boxes is not None:
                # Масштабируем координаты обратно к размеру исходного изображения
                scale_x = width / IMG_SIZE[0]
                scale_y = height / IMG_SIZE[1]
                
                for i, (box, score) in enumerate(zip(boxes, scores)):
                    x1, y1, x2, y2 = box
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    # Создаем BoundingBox
                    bbox = BoundingBox()
                    center = Pose2D()
                    center.x = (x1 + x2) / 2
                    center.y = (y1 + y2) / 2
                    center.theta = 0
                    
                    bbox.center = center
                    bbox.size_x = x2 - x1
                    bbox.size_y = y2 - y1
                    bbox.area = bbox.size_x * bbox.size_y
                    bbox.name = f"face{i}"
                    
                    target_msg.boxes.append(bbox)
                    
                    # Рисуем прямоугольник на изображении
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(cv_image, f"face: {score:.2f}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Публикация изображения с отрисованными bbox
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = rospy.Time.now()
            compressed_msg.format = "jpeg"
            compressed_msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
            self.processed_image_pub.publish(compressed_msg)
            
            # Публикуем сообщение Target
            self.target_pub.publish(target_msg)
            
            if boxes is not None:
                rospy.loginfo(f"Обнаружено {len(boxes)} лиц")
            else:
                rospy.loginfo("Лица не обнаружены")
            
        except Exception as e:
            rospy.logerr(f"Ошибка при обработке изображения: {e}")

if __name__ == '__main__':
    try:
        detector = YOLOFaceDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
