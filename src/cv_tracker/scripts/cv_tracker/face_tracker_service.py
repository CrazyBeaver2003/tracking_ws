#!/usr/bin/env python3

import rospy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Point

class FaceTrackerService:
    def __init__(self):
        rospy.init_node('face_tracker_service', anonymous=True)
        
        # Создаем сервисы для управления трекером
        self.stop_service = rospy.Service('/face_tracker/stop', Empty, self.stop_tracking_callback)
        
        # Publisher для команд остановки трекера
        self.stop_cmd_pub = rospy.Publisher('/face_tracker/stop_command', Bool, queue_size=1)
        
        # Подписываемся на статус трекера
        self.tracking_status_sub = rospy.Subscriber('/face_tracker/status', Bool, self.status_callback)
        self.tracker_center_sub = rospy.Subscriber('/face_tracker/center', Point, self.center_callback)
        
        self.tracking_active = False
        self.last_center = None
        
        print("Face Tracker Service инициализирован")
        print("Доступные сервисы:")
        print("  - /face_tracker/stop - остановить трекинг")
        
    def stop_tracking_callback(self, req):
        """Сервис для остановки трекинга"""
        print("Получена команда остановки трекинга")
        self.stop_cmd_pub.publish(Bool(True))
        return EmptyResponse()
    
    def status_callback(self, msg):
        """Обработка статуса трекера"""
        if self.tracking_active != msg.data:
            self.tracking_active = msg.data
            if msg.data:
                print("✓ Трекинг активен")
            else:
                print("✗ Трекинг неактивен")
    
    def center_callback(self, msg):
        """Обработка центра трекера"""
        self.last_center = (msg.x, msg.y)
        if self.tracking_active:
            print(f"Центр трекера: ({msg.x:.1f}, {msg.y:.1f})")

if __name__ == '__main__':
    try:
        service = FaceTrackerService()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 