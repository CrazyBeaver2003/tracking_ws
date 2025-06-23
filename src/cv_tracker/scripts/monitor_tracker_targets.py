#!/usr/bin/env python3

import rospy
import sys
import os
sys.path.insert(0, '/root/tracking_ws/devel/lib/python3/dist-packages')

from cv_tracker.msg import BoundingBox, Target
from geometry_msgs.msg import Point
from std_msgs.msg import Header
import time

class TrackerMonitor:
    def __init__(self):
        rospy.init_node('tracker_monitor', anonymous=True)
        
        # Счетчики сообщений
        self.tracker_target_count = 0
        self.tracker_center_count = 0
        self.face_target_count = 0
        
        # Время последних сообщений
        self.last_tracker_target_time = 0
        self.last_tracker_center_time = 0
        self.last_face_target_time = 0
        
        # Подписываемся на все топики трекера
        rospy.Subscriber('/face_tracker/target', Target, self.tracker_target_callback)
        rospy.Subscriber('/face_tracker/center', Point, self.tracker_center_callback)
        rospy.Subscriber('/face_recognizer/target_face', Target, self.face_target_callback)
        
        print("🔍 Монитор трекера запущен")
        print("📡 Отслеживаемые топики:")
        print("   - /face_tracker/target (Target)")
        print("   - /face_tracker/center (Point)")
        print("   - /face_recognizer/target_face (Target)")
        print("=" * 60)
        
        # Таймер для периодического отчета
        rospy.Timer(rospy.Duration(5.0), self.print_status)
        
    def tracker_target_callback(self, msg):
        """Обработка Target сообщений от трекера"""
        self.tracker_target_count += 1
        self.last_tracker_target_time = time.time()
        
        if msg.boxes:
            box = msg.boxes[0]
            # Определяем тип рамки по имени
            if box.name == "tracked_face":
                icon = "🔵"  # Синяя рамка трекера
            elif box.name == "unknown":
                icon = "🔴"  # Красная рамка нераспознанного лица
            else:
                icon = "🟢"  # Зеленая рамка распознанного лица
                
            print(f"{icon} TRACKER TARGET #{self.tracker_target_count}:")
            print(f"   📍 Центр: ({box.center.x:.1f}, {box.center.y:.1f})")
            print(f"   📏 Размер: {box.size_x:.1f} x {box.size_y:.1f}")
            print(f"   📐 Площадь: {box.area:.0f}")
            print(f"   🏷️  Имя: {box.name}")
            print(f"   🖼️  Изображение: {msg.image_width:.0f}x{msg.image_height:.0f}")
        else:
            print(f"🛑 TRACKER STOPPED #{self.tracker_target_count}: Пустое Target сообщение")
        print()
        
    def tracker_center_callback(self, msg):
        """Обработка Point сообщений от трекера"""
        self.tracker_center_count += 1
        self.last_tracker_center_time = time.time()
        
        # Логируем только каждое 30-е сообщение (чтобы не спамить)
        if self.tracker_center_count % 30 == 0:
            print(f"📍 TRACKER CENTER #{self.tracker_center_count}: ({msg.x:.1f}, {msg.y:.1f})")
        
    def face_target_callback(self, msg):
        """Обработка Target сообщений от распознавателя лиц"""
        self.face_target_count += 1
        self.last_face_target_time = time.time()
        
        if msg.boxes:
            box = msg.boxes[0]
            print(f"👤 FACE TARGET #{self.face_target_count}:")
            print(f"   📍 Центр: ({box.center.x:.1f}, {box.center.y:.1f})")
            print(f"   📏 Размер: {box.size_x:.1f} x {box.size_y:.1f}")
            print(f"   🏷️  Имя: {box.name}")
            print()
        
    def print_status(self, event):
        """Периодический отчет о статусе"""
        current_time = time.time()
        
        print("📊 СТАТУС ТРЕКЕРА (последние 5 секунд):")
        print(f"   🎯 Target сообщения трекера: {self.tracker_target_count} всего")
        print(f"   📍 Center сообщения трекера: {self.tracker_center_count} всего")
        print(f"   👤 Target сообщения лиц: {self.face_target_count} всего")
        
        # Проверяем активность
        if current_time - self.last_tracker_target_time < 5.0:
            print("   ✅ Трекер АКТИВЕН (Target сообщения поступают)")
        else:
            print("   ❌ Трекер НЕАКТИВЕН (нет Target сообщений)")
            
        if current_time - self.last_tracker_center_time < 5.0:
            print("   ✅ Центр трекера обновляется")
        else:
            print("   ❌ Центр трекера не обновляется")
            
        print("=" * 60)

def main():
    try:
        monitor = TrackerMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("🔚 Монитор трекера завершен")
    except KeyboardInterrupt:
        print("🔚 Монитор трекера прерван пользователем")

if __name__ == '__main__':
    main() 