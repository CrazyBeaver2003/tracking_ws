#!/usr/bin/env python3

"""
Мост для работы с ROS без прямой инициализации узла
"""

import os
import sys
import time
import signal
import subprocess
import json
import threading

class ROSBridge:
    """
    Класс для взаимодействия с ROS через командную строку
    Обходит проблемы с логированием и другими ошибками инициализации ROS
    """
    
    def __init__(self, name="ros_bridge"):
        self.name = name
        self.running = True
        
        # Проверяем, запущен ли roscore
        if not self._check_rosmaster():
            print("ROS Master не запущен. Запускаем roscore...")
            self._start_roscore()
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _check_rosmaster(self):
        """Проверяет, запущен ли ROS Master"""
        try:
            result = subprocess.run(
                "rostopic list", 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=5
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def _start_roscore(self):
        """Запускает roscore в отдельном процессе"""
        try:
            self.roscore_process = subprocess.Popen(
                "roscore", 
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Ждем запуска roscore
            time.sleep(5)
            
            if self._check_rosmaster():
                print("ROS Master успешно запущен")
            else:
                print("Ошибка запуска ROS Master")
        except Exception as e:
            print(f"Ошибка запуска roscore: {e}")
    
    def publish(self, topic, msg_type, msg_data):
        """
        Публикует сообщение в топик ROS
        
        Args:
            topic (str): Имя топика
            msg_type (str): Тип сообщения (например, 'std_msgs/String')
            msg_data (str): Данные сообщения в формате JSON
        """
        try:
            cmd = f"rostopic pub --once {topic} {msg_type} '{msg_data}'"
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Ошибка публикации в топик {topic}: {result.stderr}")
            
            return result.returncode == 0
        except Exception as e:
            print(f"Ошибка при публикации в топик {topic}: {e}")
            return False
    
    def subscribe(self, topic, callback, rate=10):
        """
        Подписывается на топик ROS и вызывает callback при получении сообщения
        
        Args:
            topic (str): Имя топика
            callback (function): Функция обратного вызова, принимающая сообщение
            rate (int): Частота проверки сообщений (Гц)
        """
        def _subscribe_thread():
            while self.running:
                try:
                    result = subprocess.run(
                        f"rostopic echo -n 1 {topic}",
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=1/rate,
                        text=True
                    )
                    
                    if result.returncode == 0 and result.stdout.strip():
                        # Преобразуем вывод в JSON и передаем в callback
                        try:
                            data = json.loads(result.stdout)
                            callback(data)
                        except json.JSONDecodeError:
                            # Если не удалось распарсить как JSON, передаем как есть
                            callback(result.stdout)
                except subprocess.TimeoutExpired:
                    # Тайм-аут - это нормально, продолжаем работу
                    pass
                except Exception as e:
                    print(f"Ошибка при получении сообщения из топика {topic}: {e}")
                    time.sleep(1)
        
        # Запускаем поток подписки
        thread = threading.Thread(target=_subscribe_thread)
        thread.daemon = True
        thread.start()
        
        return thread
    
    def _signal_handler(self, sig, frame):
        """Обработчик сигналов для корректного завершения"""
        print(f"Получен сигнал {sig}, завершение работы...")
        self.running = False
        
        if hasattr(self, 'roscore_process'):
            self.roscore_process.terminate()
        
        sys.exit(0)
    
    def spin(self):
        """Блокирует выполнение до получения сигнала завершения"""
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Получен сигнал прерывания, завершение работы...")
            self.running = False


if __name__ == "__main__":
    # Пример использования
    bridge = ROSBridge("test_bridge")
    
    def callback(msg):
        print(f"Получено сообщение: {msg}")
    
    # Подписываемся на топик
    bridge.subscribe("/rosout", callback)
    
    # Публикуем сообщение
    bridge.publish("/chatter", "std_msgs/String", '{"data": "Привет от ROSBridge!"}') 