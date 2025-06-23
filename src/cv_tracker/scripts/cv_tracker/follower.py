#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import sys
# Добавляем путь к сгенерированным пакетам
sys.path.insert(0, '/root/tracking_ws/devel/lib/python3/dist-packages')
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import serial
import time
from cv_tracker.msg import Target
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker

class FaceFollower:
    def __init__(self):
        # Инициализация ROS ноды
        rospy.init_node('face_follower', anonymous=True)
        rospy.loginfo("Инициализация FaceFollower...")
        
        # Параметры сервопривода
        self.center_x = 320  # Центр изображения по X (предполагаемый)
        self.center_y = 320  # Центр изображения по Y (предполагаемый)
        self.current_angle_x = 90  # Начальный угол по X
        self.current_angle_y = 70  # Начальный угол по Y
        self.max_angle = 180
        self.min_angle = 0
        
        # Коэффициенты PID-регулятора (можно настроить)
        self.kp = 0.2  # Пропорциональный коэффициент
        
        # Подписка на топик с информацией о лицах
        # self.target_sub = rospy.Subscriber('/face_detection/targets', Target, self.target_callback)
        self.target_sub = rospy.Subscriber('/yolo/target_person', Target, self.target_callback)
        
        # Инициализация последовательного порта
        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            time.sleep(2)  # Ожидание инициализации порта
            rospy.loginfo("Последовательный порт открыт успешно")
        except Exception as e:
            rospy.logerr(f"Ошибка открытия последовательного порта: {e}")
            return
            
        # Установка начального положения
        self.send_angles(self.current_angle_x, self.current_angle_y)

        self.distance = None 

        self.marker_pub = rospy.Publisher('object_position/marker', Marker, queue_size=1)

        rospy.loginfo("Инициализация FaceFollower завершена")
    def read_distance(self):
        if self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8').strip()
            print(f"[DISTANCE] Расстояние до цели: {line}")
            try:
                distance = float(line)
                self.distance = distance
                print(f"Расстояние до цели: {distance} см")
                return distance
            except ValueError:
                # rospy.logerr("Не удалось преобразовать строку в число")
                # return None
                return None

    def send_angles(self, angle_x, angle_y):
        """Отправка углов на сервоприводы"""
        try:
            # Ограничение углов
            angle_x = max(min(angle_x, self.max_angle), self.min_angle)
            angle_y = max(min(angle_y, self.max_angle), self.min_angle)
            
            # Формирование и отправка команды
            command = f"{int(angle_x)},{int(angle_y)}\n"
            self.ser.write(command.encode())
            print(f"[SERVO] Отправлено: {command.strip()}")
            
            # Обновление текущих углов
            self.current_angle_x = angle_x
            self.current_angle_y = angle_y

            
        except Exception as e:
            rospy.logerr(f"Ошибка отправки команды: {e}")

    def send_markers(self):
        if self.distance is not None:
            marker = Marker()
            marker.header.frame_id = "camera_link"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "object_position"
            marker.id = 0
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = self.distance * np.cos(np.deg2rad(self.current_angle_x))/100
            marker.pose.position.y = self.distance * np.sin(np.deg2rad(self.current_angle_x))/100
            marker.pose.position.z = self.distance * np.sin(np.deg2rad(self.current_angle_y))/100
            
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5

            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            rospy.loginfo("Отправлен маркер!!!!!!!")
            self.marker_pub.publish(marker)


    def calculate_angles(self, target_x, target_y, image_width, image_height):
        """Расчет углов поворота на основе положения цели"""
        # Нормализация координат относительно центра
        error_x = (target_x - image_width/2) / (image_width/2)  # [-1, 1]
        error_y = (target_y - image_height/2) / (image_height/2)  # [-1, 1]
        if error_x < 0.02 and error_y < 0.02:
            distance = self.read_distance()
            self.distance = distance
            self.send_markers()
        # Расчет новых углов с использованием P-регулятора
        delta_angle_x = -self.kp * error_x * 30  # Максимальное изменение ±30 градусов
        delta_angle_y = -self.kp * error_y * 30  # Инвертируем для Y
        
        new_angle_x = self.current_angle_x + delta_angle_x
        new_angle_y = self.current_angle_y + delta_angle_y
        
        return new_angle_x, new_angle_y
        
    def target_callback(self, msg):
        """Обработчик сообщений с информацией о лицах"""
        try:
            if msg.boxes:  # Если обнаружены лица
                # Берем первое лицо
                face = msg.boxes[0]
                
                # Получаем координаты центра лица
                target_x = face.center.x
                target_y = face.center.y
                
                print(f"[TRACK] Лицо: x={target_x:.1f}, y={target_y:.1f}")
                
                # Расчет новых углов сервоприводов
                new_angle_x, new_angle_y = self.calculate_angles(
                    target_x, target_y, 
                    msg.image_width, msg.image_height
                )
                
                # Отправка команды на сервоприводы
                self.send_angles(new_angle_x, new_angle_y)
                
        except Exception as e:
            rospy.logerr(f"Ошибка в обработке цели: {e}")
            
    def __del__(self):
        """Деструктор для закрытия последовательного порта"""
        if hasattr(self, 'ser'):
            self.ser.close()

if __name__ == '__main__':
    try:
        follower = FaceFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

