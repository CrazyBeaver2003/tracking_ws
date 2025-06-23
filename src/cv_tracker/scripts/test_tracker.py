#!/usr/bin/env python3

import rospy
from std_srvs.srv import Empty
from std_msgs.msg import Bool
from geometry_msgs.msg import Point

def test_face_tracker():
    """Простой тест трекера лиц"""
    rospy.init_node('test_face_tracker', anonymous=True)
    
    print("=== Тест трекера лиц ===")
    
    # Ждем запуска сервиса
    print("Ожидание запуска сервиса трекера...")
    rospy.wait_for_service('/face_tracker/stop', timeout=10.0)
    print("✓ Сервис трекера найден")
    
    # Подписываемся на статус и центр трекера
    tracking_active = [False]
    last_center = [None]
    
    def status_callback(msg):
        if tracking_active[0] != msg.data:
            tracking_active[0] = msg.data
            print(f"📊 Статус трекера: {'АКТИВЕН' if msg.data else 'НЕАКТИВЕН'}")
    
    def center_callback(msg):
        last_center[0] = (msg.x, msg.y)
        if tracking_active[0]:
            print(f"📍 Центр объекта: ({msg.x:.1f}, {msg.y:.1f})")
    
    status_sub = rospy.Subscriber('/face_tracker/status', Bool, status_callback)
    center_sub = rospy.Subscriber('/face_tracker/center', Point, center_callback)
    
    print("✓ Подписались на топики трекера")
    
    # Ждем активации трекера
    print("\n🎯 Ожидание активации трекера...")
    print("   (трекер автоматически запустится при обнаружении знакомого лица)")
    
    rate = rospy.Rate(1)  # 1 Hz
    
    try:
        while not rospy.is_shutdown():
            if tracking_active[0]:
                print(f"✅ Трекер активен! Центр: {last_center[0] if last_center[0] else 'не определен'}")
                
                # Тестируем остановку трекера
                response = input("\nВведите 's' для остановки трекера или Enter для продолжения: ")
                if response.lower() == 's':
                    try:
                        stop_service = rospy.ServiceProxy('/face_tracker/stop', Empty)
                        stop_service()
                        print("🛑 Команда остановки отправлена")
                    except rospy.ServiceException as e:
                        print(f"❌ Ошибка при вызове сервиса: {e}")
            else:
                print("⏳ Трекер неактивен, ожидание...")
            
            rate.sleep()
            
    except KeyboardInterrupt:
        print("\n👋 Тест завершен")

if __name__ == '__main__':
    try:
        test_face_tracker()
    except Exception as e:
        print(f"❌ Ошибка в тесте: {e}") 