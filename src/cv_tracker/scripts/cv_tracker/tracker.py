import rospy
import cv2
import numpy as np
import sys

sys.path.insert(0, '/root/tracking_ws/devel/lib/python3/dist-packages')

from cv_tracker.msg import BoundingBox, Target
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge


class ObjectTracker:
    def __init__(self):
        # Инициализация ROS ноды
        rospy.init_node('object_tracker', anonymous=True)
        rospy.loginfo("Инициализация ObjectTracker...")
        
        self.tracker = cv2.TrackerCSRT_create()
        self.image_sub = rospy.Subscriber('camera/image_raw/compressed', CompressedImage, self.image_callback)
        self.target_sub = rospy.Subscriber('/face_recognizer/target_face', Target, self.target_callback)

        self.image_pub = rospy.Publisher('object_tracker/image', Image, queue_size=1)
        self.target_pub = rospy.Publisher('object_tracker/target', Target, queue_size=1)

        self.bridge = CvBridge()
        self.target = None
        self.tracking = False
        self.initialized = False
    
    def target_callback(self, msg):
        """Обработчик сообщений с информацией о целях"""
        if msg.boxes:  # Если есть обнаруженные лица
            # Берем первое обнаруженное лицо
            bbox = msg.boxes[0]
            self.target = bbox
            rospy.loginfo(f"Получен новый target: центр=({bbox.center.x}, {bbox.center.y}), размер=({bbox.size_x}, {bbox.size_y})")

    def draw_bbox(self, image, bbox, color=(0, 255, 0), thickness=2):
        """Отрисовка bbox с дополнительной информацией"""
        x, y, w, h = [int(v) for v in bbox]
        
        # Рисуем прямоугольник
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        # Добавляем текст со статусом и координатами
        text = f"Tracking: ({x+w//2}, {y+h//2})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        
        # Получаем размеры текста для правильного размещения фона
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Рисуем фон для текста
        cv2.rectangle(image, (x, y - text_height - 10), (x + text_width, y), color, -1)
        
        # Рисуем текст
        cv2.putText(image, text, (x, y - 5), font, font_scale, (0, 0, 0), thickness)
        
        # Рисуем центральную точку
        center_x, center_y = x + w//2, y + h//2
        cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)
        
        return image

    def image_callback(self, msg):
        """Обработчик входящих изображений"""
        try:
            # Преобразование compressed изображения в cv2
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            
            if self.target and not self.initialized:
                # Создаем bbox в формате (x, y, width, height)
                x = int(self.target.center.x - self.target.size_x / 2)
                y = int(self.target.center.y - self.target.size_y / 2)
                w = int(self.target.size_x)
                h = int(self.target.size_y)
                bbox = (x, y, w, h)
                
                # Отрисовываем целевой bbox
                cv_image = self.draw_bbox(cv_image, bbox, color=(255, 0, 0))
                
                # Инициализируем трекер
                if self.init_tracker(cv_image, bbox):
                    self.initialized = True
                    rospy.loginfo("Трекер успешно инициализирован")
            
            # elif self.initialized and self.tracking:
                # Обновляем позицию трекера
                success, bbox = self.tracker.update(cv_image)
                if success:
                    # Отрисовываем bbox трекера
                    cv_image = self.draw_bbox(cv_image, bbox)
                    
                    # Создаем и публикуем сообщение Target
                    x, y, w, h = [int(v) for v in bbox]
                    target_msg = Target()
                    bbox_msg = BoundingBox()
                    bbox_msg.center.x = x + w/2
                    bbox_msg.center.y = y + h/2
                    bbox_msg.size_x = w
                    bbox_msg.size_y = h
                    bbox_msg.area = w * h
                    target_msg.boxes = [bbox_msg]
                    target_msg.image_width = 640
                    target_msg.image_height = 640
                    self.target_pub.publish(target_msg)
                else:
                    rospy.logwarn("Потеряна цель трекинга")
                    self.initialized = False
                    self.tracking = False
            
            # Добавляем статус трекера в верхний левый угол
            status_text = "Status: "
            if self.tracking:
                status_text += "Tracking"
            elif self.target is not None:
                status_text += "Target Detected"
            else:
                status_text += "Waiting for Target"
                
            cv2.putText(cv_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)
            
            # Публикуем обработанное изображение
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.image_pub.publish(img_msg)
            
        except Exception as e:
            rospy.logerr(f"Ошибка при обработке изображения: {str(e)}")
    
    def init_tracker(self, frame, bbox):
        try:
            rospy.loginfo(f"Инициализация трекера с bbox: {bbox}")
            self.tracker = cv2.TrackerCSRT_create()
            self.tracking = self.tracker.init(frame, bbox)
            if self.tracking:
                rospy.loginfo("Трекер успешно инициализирован")
            return self.tracking
        except Exception as e:
            rospy.logerr(f"Ошибка при инициализации трекера: {str(e)}")
            return False

def main():
    try:
        tracker = ObjectTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Критическая ошибка: {str(e)}")

if __name__ == "__main__":
    main()
            