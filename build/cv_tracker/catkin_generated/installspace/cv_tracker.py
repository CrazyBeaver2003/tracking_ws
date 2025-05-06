#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
import face_recognition
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

path_to_known_image = "/root/diplom_ws/data/photo.jpg"

class ObjectTracker:
    def __init__(self):
        # Инициализация ROS ноды
        rospy.init_node('cv_tracker', anonymous=True)
        
        rospy.loginfo("Инициализация трекера...")
        self.tracker = cv2.TrackerKCF_create()
        self.bbox = None
        self.tracking = False
        self.bridge = CvBridge()
        
        # Параметры для оптимизации
        self.face_check_interval = 30  # Проверка лица каждые N кадров
        self.frame_counter = 0
        self.tracking_quality_threshold = 0.7  # Порог качества трекинга
        self.last_face_check_time = rospy.Time.now()
        self.face_check_period = rospy.Duration(1.0)  # Проверять лицо каждую секунду
        
        try:
            rospy.loginfo(f"Загрузка изображения из {path_to_known_image}")
            self.known_image = face_recognition.load_image_file(path_to_known_image)
            self.known_face_encoding = face_recognition.face_encodings(self.known_image)
            if not self.known_face_encoding:
                rospy.logerr("Не удалось найти лицо на эталонном изображении!")
                raise Exception("Лицо не найдено на эталонном изображении")
            self.known_face_encoding = self.known_face_encoding[0]
            rospy.loginfo("Эталонное лицо успешно закодировано")
        except Exception as e:
            rospy.logerr(f"Ошибка при загрузке эталонного изображения: {str(e)}")
            raise
        
        # Создание подписчика на топик с изображением
        self.image_sub = rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.image_callback)
        
        # Создание издателя для публикации результатов трекинга
        self.tracking_pub = rospy.Publisher('/tracking/bbox', Point, queue_size=10)
        
        # Флаг для инициализации трекера
        self.initialized = False
        rospy.loginfo("Трекер инициализирован")

    def check_tracking_quality(self, frame, bbox):
        """Проверяет качество трекинга"""
        try:
            x, y, w, h = [int(v) for v in bbox]
            
            # Проверяем, что bbox находится в пределах кадра
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                rospy.logwarn("Трекер вышел за пределы кадра")
                return False
            
            # Проверяем размер bbox
            if w < 20 or h < 20:
                rospy.logwarn("Трекер слишком маленький")
                return False
            if w > frame.shape[1] * 0.8 or h > frame.shape[0] * 0.8:
                rospy.logwarn("Трекер слишком большой")
                return False
            
            # Проверяем соотношение сторон (примерно как у лица)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 1.5:
                rospy.logwarn("Неправильное соотношение сторон трекера")
                return False
            
            return True
        except Exception as e:
            rospy.logerr(f"Ошибка при проверке качества трекинга: {str(e)}")
            return False

    def find_target_face(self, frame):
        """Находит лицо целевого человека на кадре"""
        try:
            # Конвертируем в RGB перед уменьшением размера
            rgb_frame = frame[:, :, ::-1]  # BGR to RGB
            
            # Уменьшаем размер кадра для ускорения обработки
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            
            # Находим все лица на кадре
            rospy.logdebug("Поиск лиц на кадре...")
            face_locations = face_recognition.face_locations(small_frame)
            rospy.logdebug(f"Найдено {len(face_locations)} лиц")
            
            if not face_locations:
                return None
                
            # Получаем кодировки для всех найденных лиц
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            
            best_match = None
            best_distance = float('inf')
            
            for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                # Используем расстояние вместо простого сравнения
                distance = face_recognition.face_distance([self.known_face_encoding], face_encoding)[0]
                rospy.logdebug(f"Расстояние до лица #{i+1}: {distance}")
                if distance < best_distance:
                    best_distance = distance
                    best_match = face_location
            
            if best_match and best_distance < 0.6:  # Порог расстояния для уверенного совпадения
                rospy.loginfo(f"Найдено лучшее совпадение с расстоянием: {best_distance}")
                # Преобразуем координаты обратно к исходному размеру
                top, right, bottom, left = best_match
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                return (left, top, right-left, bottom-top)
            return None
        except Exception as e:
            rospy.logerr(f"Ошибка в find_target_face: {str(e)}")
            return None

    def init_tracker(self, frame, bbox):
        """Инициализация трекера с выбранной областью"""
        try:
            rospy.loginfo(f"Инициализация трекера с bbox: {bbox}")
            self.tracker = cv2.TrackerKCF_create()
            self.tracking = self.tracker.init(frame, bbox)
            if self.tracking:
                rospy.loginfo("Трекер успешно инициализирован")
            # else:
            #     rospy.logerr("Не удалось инициализировать трекер")
            return self.tracking
        except Exception as e:
            rospy.logerr(f"Ошибка при инициализации трекера: {str(e)}")
            return False

    def update(self, frame):
        """Обновление позиции трекера на новом кадре"""
        if not self.tracking:
            return None, frame

        try:
            success, bbox = self.tracker.update(frame)
            if success and self.check_tracking_quality(frame, bbox):
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
            else:
                rospy.logwarn("Трекер потерял объект или низкое качество трекинга")
                self.tracking = False
                self.initialized = False
            return None, frame
        except Exception as e:
            rospy.logerr(f"Ошибка при обновлении трекера: {str(e)}")
            return None, frame

    def image_callback(self, msg):
        """Обработчик входящих изображений"""
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            current_time = rospy.Time.now()
            
            if not self.initialized:
                rospy.loginfo("Поиск целевого лица...")
                bbox = self.find_target_face(cv_image)
                if bbox is not None:
                    rospy.loginfo(f"Найдено лицо с bbox: {bbox}")
                    if self.init_tracker(cv_image, bbox):
                        self.initialized = True
                        self.last_face_check_time = current_time
                else:
                    rospy.logdebug("Целевое лицо не найдено")
            else:
                # Обновление трекера
                bbox, cv_image = self.update(cv_image)
                
                # Периодическая проверка лица
                if (current_time - self.last_face_check_time) > self.face_check_period:
                    rospy.loginfo("Периодическая проверка лица...")
                    new_bbox = self.find_target_face(cv_image)
                    if new_bbox is not None:
                        # Если нашли лицо, обновляем трекер
                        self.init_tracker(cv_image, new_bbox)
                    self.last_face_check_time = current_time
            
            # Отображение результата
            # cv2.imshow("Трекинг лица", cv_image)
            # cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"Ошибка при обработке изображения: {str(e)}")

def main():
    try:
        rospy.loginfo("Запуск трекера...")
        tracker = ObjectTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Прерывание ROS")
    except Exception as e:
        rospy.logerr(f"Критическая ошибка: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        rospy.loginfo("Трекер остановлен")

if __name__ == "__main__":
    main()