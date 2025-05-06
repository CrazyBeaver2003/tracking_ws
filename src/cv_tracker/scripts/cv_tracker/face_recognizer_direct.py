#!/usr/bin/env python3

import cv2
import os
import sys
import glob
import numpy as np
import time
import threading
from pathlib import Path

# Добавляем путь к пакетам
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except Exception as e:
    print(f"Ошибка при импорте RKNNLite: {e}")
    RKNN_AVAILABLE = False

class RKNN_model_container:
    def __init__(self, model_path, target=None, device_id=None) -> None:
        self.rknn = RKNNLite()
        print(f"Загрузка модели: {model_path}")
        self.rknn.load_rknn(model_path)
        
        print("--> Инициализация среды выполнения")
        ret = self.rknn.init_runtime()
        if ret != 0:
            print("Ошибка инициализации среды выполнения")
            raise RuntimeError(f"Ошибка инициализации RKNN: {ret}")
        print("Готово!")

    def run(self, inputs):
        if self.rknn is None:
            print("ОШИБКА: rknn был освобожден")
            return []

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)
        return result

    def release(self):
        if self.rknn:
            self.rknn.release()
            self.rknn = None

class FaceRecognizer:
    def __init__(self):
        print("Инициализация распознавателя лиц...")
        
        # Определяем путь к моделям
        base_path = str(current_dir.parent.parent)  # путь к cv_tracker
        models_path = os.path.join(base_path, 'models')
        
        # Ищем модель facenet
        model_path = os.path.join(models_path, 'facenet_model.rknn')
        if not os.path.exists(model_path):
            print(f"Модель не найдена по пути: {model_path}, ищем другие модели...")
            models = glob.glob(os.path.join(models_path, 'facenet*.rknn'))
            if models:
                model_path = models[0]
                print(f"Используем модель: {model_path}")
            else:
                print("Модели facenet не найдены!")
                raise FileNotFoundError("Модели facenet не найдены!")
        
        if RKNN_AVAILABLE:
            try:
                self.facenet_model = RKNN_model_container(model_path)
                print("Модель RKNN успешно загружена")
            except Exception as e:
                print(f"Ошибка при загрузке модели RKNN: {e}")
                self.facenet_model = None
        else:
            print("RKNN не доступен, распознавание лиц не будет работать")
            self.facenet_model = None
            
        # Флаг для остановки работы
        self.running = True
        
        # Для тестирования на случайных данных
        self.testing_thread = threading.Thread(target=self.test_model_periodically)
        self.testing_thread.daemon = True
        self.testing_thread.start()
    
    def test_model_periodically(self):
        """Периодически тестирует модель на случайных данных"""
        while self.running:
            print("\n=== Периодический тест модели ===")
            test_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
            embeddings = self.recognize_face(test_image)
            
            if embeddings is not None:
                print(f"Тест прошел успешно. Получен вектор признаков длиной {embeddings.shape}")
                print(f"Мин: {np.min(embeddings):.2f}, Макс: {np.max(embeddings):.2f}")
            else:
                print("Тест не удался - получен пустой результат")
                
            # Ждем 10 секунд
            for i in range(10):
                if not self.running:
                    break
                time.sleep(1)
    
    def recognize_face(self, face_image):
        """
        Распознает лицо на изображении и возвращает вектор признаков
        """
        if self.facenet_model is None:
            print("Модель не инициализирована!")
            return None
        
        # Предобработка изображения для модели
        # Обычно требуется изменение размера до 160x160 и нормализация
        if face_image.shape[:2] != (160, 160):
            face_image = cv2.resize(face_image, (160, 160))
        
        # Нормализация
        face_image = face_image.astype(np.float32)
        face_image = (face_image - 127.5) / 128.0
        
        # Добавляем размерность батча (NHWC формат, где N=1)
        # Модель ожидает 4D тензор: [batch_size, height, width, channels]
        face_image_batch = np.expand_dims(face_image, axis=0)
        print(f"Форма входного тензора: {face_image_batch.shape}")
        
        try:
            # Запуск модели
            outputs = self.facenet_model.run([face_image_batch])
            
            # Проверка на None
            if outputs is None or len(outputs) == 0:
                print("Модель вернула пустой результат")
                return None
                
            # Предполагаем, что выход - это вектор признаков
            embeddings = outputs[0]
            print(f"Форма выходного тензора: {embeddings.shape}")
            
            # Убираем батч, если он есть
            if len(embeddings.shape) > 1 and embeddings.shape[0] == 1:
                embeddings = embeddings[0]
                
            return embeddings
        
        except Exception as e:
            print(f"Ошибка при вызове модели: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def stop(self):
        """Останавливает работу распознавателя"""
        self.running = False
        if hasattr(self, 'facenet_model') and self.facenet_model is not None:
            self.facenet_model.release()
            
    def __del__(self):
        self.stop()


if __name__ == "__main__":
    try:
        print("Запуск прямого тестирования распознавателя лиц...")
        
        # Создаем распознаватель
        recognizer = FaceRecognizer()
        
        print("\nРаспознаватель запущен и работает в фоновом режиме.")
        print("Нажмите Ctrl+C для завершения.")
        
        # Ожидаем завершения
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nПолучен сигнал прерывания, завершение работы...")
        if 'recognizer' in locals():
            recognizer.stop()
        
    except Exception as e:
        import traceback
        print(f"Ошибка: {e}")
        traceback.print_exc() 