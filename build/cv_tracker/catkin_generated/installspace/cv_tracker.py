#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

class ObjectTracker:
    def __init__(self):
        self.tracker = cv2.TrackerCSRT_create()
        self.bbox = None
        self.tracking = False
        self.bridge = CvBridge()
        
        # Инициализация ROS ноды
        rospy.init_node('cv_tracker', anonymous=True)
        
        # Создание подписчика на топик с изображением
        self.image_sub = rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.image_callback)
        
        # Создание издателя для публикации результатов трекинга
        self.tracking_pub = rospy.Publisher('/tracking/bbox', Point, queue_size=10)
        
        # Флаг для инициализации трекера
        self.initialized = False

    def select_roi(self, frame):
        """Позволяет пользователю выбрать область для трекинга"""
        self.bbox = cv2.selectROI("Выберите объект для трекинга", frame, False)
        cv2.destroyWindow("Выберите объект для трекинга")
        return self.bbox

    def init_tracker(self, frame, bbox):
        """Инициализация трекера с выбранной областью"""
        self.tracker = cv2.TrackerCSRT_create()
        self.tracking = self.tracker.init(frame, bbox)
        return self.tracking

    def update(self, frame):
        """Обновление позиции трекера на новом кадре"""
        if not self.tracking:
            return None, frame

        success, bbox = self.tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Публикация центра объекта
            center_x = x + w/2
            center_y = y + h/2
            point_msg = Point()
            point_msg.x = center_x
            point_msg.y = center_y
            point_msg.z = 0
            self.tracking_pub.publish(point_msg)
            
            return bbox, frame
        return None, frame

    def image_callback(self, msg):
        """Обработчик входящих изображений"""
        try:
            # Конвертация ROS сообщения в OpenCV формат
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            
            if not self.initialized:
                # Инициализация трекера при первом кадре
                bbox = self.select_roi(cv_image)
                self.init_tracker(cv_image, bbox)
                self.initialized = True
            else:
                # Обновление трекера
                bbox, cv_image = self.update(cv_image)
            
            # Отображение результата
            cv2.imshow("Трекинг объекта", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Ошибка при обработке изображения: {str(e)}")

def main():
    try:
        tracker = ObjectTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()