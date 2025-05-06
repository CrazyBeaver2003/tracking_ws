#!/usr/bin/env python3

import os
import sys
import cv2
import json
import numpy as np
import time
import traceback
from pathlib import Path

# Добавляем путь к пакетам
sys.path.append(str(Path(__file__).parent / 'src'))

# Импортируем наш мост
from cv_tracker.scripts.ros_bridge import ROSBridge

# Импортируем модуль с распознавателем, а не класс напрямую
import cv_tracker.scripts.cv_tracker.face_recognizer_direct as face_rec

class FaceRecognitionBridge:
    def __init__(self):
        print("Инициализация распознавателя лиц с мостом ROS...")
        
        # Создаем мост ROS
        self.ros_bridge = ROSBridge("face_recognition_bridge")
        
        # Создаем распознаватель лиц
        self.recognizer = face_rec.FaceRecognizer()
        
        # Настраиваем подписки
        self.setup_subscriptions()
        
        print("Инициализация завершена, система готова к работе")
    
    def setup_subscriptions(self):
        """Настраивает подписки на топики ROS"""
        # Подписываемся на топик с целями (лицами) от детектора
        self.target_sub = self.ros_bridge.subscribe(
            "/face_detection/targets",
            self.target_callback
        )
        
        # Подписываемся на топик с изображениями с камеры
        self.image_sub = self.ros_bridge.subscribe(
            "/camera/image_raw/compressed",
            self.image_callback
        )
    
    def target_callback(self, msg):
        """Обработчик сообщений с целями от детектора лиц"""
        try:
            # Проверяем, есть ли данные
            if not msg:
                return
                
            # Проверяем, есть ли боксы (лица)
            if not msg.get('boxes'):
                print("Лица не обнаружены")
                return
            
            print(f"Получена информация о {len(msg['boxes'])} лицах")
            
            # Здесь будет обработка обнаруженных лиц
            # В полной версии мы бы извлекали изображения лиц и распознавали их
            
        except Exception as e:
            print(f"Ошибка при обработке целей: {e}")
            traceback.print_exc()
    
    def image_callback(self, msg):
        """Обработчик сообщений с изображением"""
        try:
            # Проверяем, есть ли данные
            if not msg:
                return
                
            # В полной версии мы бы обрабатывали изображение
            # Сейчас просто логируем получение
            print("Получено новое изображение")
            
        except Exception as e:
            print(f"Ошибка при обработке изображения: {e}")
            traceback.print_exc()
    
    def publish_recognition(self, embeddings, face_id, box_id):
        """Публикует результаты распознавания"""
        # Создаем сообщение с результатами распознавания
        # В полной версии здесь был бы сложный JSON с результатами
        msg_data = json.dumps({
            "embedding_size": len(embeddings) if embeddings is not None else 0,
            "face_id": face_id,
            "box_id": box_id
        })
        
        # Публикуем сообщение
        self.ros_bridge.publish(
            "/face_recognition/result",
            "std_msgs/String",
            msg_data
        )
    
    def run(self):
        """Запускает основной цикл работы"""
        print("Запуск основного цикла работы...")
        
        try:
            # Передаем управление мосту
            self.ros_bridge.spin()
        except KeyboardInterrupt:
            print("Получен сигнал прерывания, завершение работы...")
        except Exception as e:
            print(f"Необработанная ошибка: {e}")
            traceback.print_exc()
        finally:
            # Останавливаем распознаватель
            self.recognizer.stop()


if __name__ == "__main__":
    try:
        # Создаем и запускаем мост
        bridge = FaceRecognitionBridge()
        bridge.run()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        traceback.print_exc() 