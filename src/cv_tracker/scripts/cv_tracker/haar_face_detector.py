#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
import os
import sys

# Добавляем путь к сгенерированным пакетам
sys.path.insert(0, '/root/tracking_ws/devel/lib/python3/dist-packages')

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


from cv_tracker.msg import BoundingBox, Target
from geometry_msgs.msg import Pose2D

class HaarFaceDetector:
    def __init__(self):
        # Инициализация ROS параметров
        rospy.init_node('haar_face_detector', anonymous=True, log_level=rospy.INFO)
        
        # Параметры для подписки на изображение
        image_topic = rospy.get_param('~image_topic', '/camera/image_raw/compressed')
        self.image_sub = rospy.Subscriber(image_topic, CompressedImage, self.image_callback)
        
        # Publisher для обработанного изображения
        self.processed_image_pub = rospy.Publisher('/face_detection/image/compressed', CompressedImage, queue_size=10)
        
        # Publisher для сообщений Target
        self.target_pub = rospy.Publisher('/face_detection/targets', Target, queue_size=10)
        
        # Создание CvBridge
        self.bridge = CvBridge()
        
        # Загрузка каскада Хаара для обнаружения лиц
        haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(haar_cascade_path):
            rospy.logerr(f"Файл каскада не найден: {haar_cascade_path}")
            exit(1)
            
        self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        if self.face_cascade.empty():
            rospy.logerr("Ошибка загрузки каскада Хаара")
            exit(1)
            
        # Параметры для настройки детектора
        self.min_neighbors = rospy.get_param('~min_neighbors', 5)
        self.scale_factor = rospy.get_param('~scale_factor', 1.1)
        self.min_size = (rospy.get_param('~min_face_width', 30), 
                         rospy.get_param('~min_face_height', 30))
        
        rospy.loginfo("Инициализация HaarFaceDetector завершена")
        
    def image_callback(self, msg):
        """Обработчик входящих изображений"""
        try:
            # Преобразование compressed сообщения в изображение OpenCV
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Получаем размеры изображения
            height, width = cv_image.shape[:2]
            
            # Преобразование изображения в оттенки серого для детектора Хаара
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Обнаружение лиц
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            
            # Создаем сообщение Target
            target_msg = Target()
            target_msg.image_height = height
            target_msg.image_width = width
            
            # Отрисовка bbox для каждого обнаруженного лица
            for i, (x, y, w, h) in enumerate(faces):
                # Добавляем BoundingBox в Target
                bbox = BoundingBox()
                
                # Создаем Pose2D для центра bbox
                center = Pose2D()
                center.x = x + w/2
                center.y = y + h/2
                center.theta = 0  # Поворот не используется для лиц
                
                bbox.center = center
                bbox.area = w * h
                bbox.size_x = w
                bbox.size_y = h
                bbox.name = f"face{i}"
                
                # Добавляем bbox в массив
                target_msg.boxes.append(bbox)
                
                # Рисуем прямоугольник на изображении
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            # Публикация изображения с отрисованными bbox
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = rospy.Time.now()
            compressed_msg.format = "jpeg"
            compressed_msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
            self.processed_image_pub.publish(compressed_msg)
            
            # Публикуем сообщение Target
            self.target_pub.publish(target_msg)
            
            if len(faces) > 0:
                rospy.loginfo(f"Обнаружено {len(faces)} лиц. Координаты: {faces}")
            else:
                rospy.loginfo("Лица не обнаружены")
            
        except Exception as e:
            rospy.logerr(f"Ошибка при обработке изображения: {e}")
            
if __name__ == '__main__':
    try:
        detector = HaarFaceDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 