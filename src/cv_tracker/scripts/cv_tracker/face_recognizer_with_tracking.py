import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
import sys
import os
import logging
import threading
import queue
import time
from sklearn import preprocessing
sys.path.insert(0, '/root/tracking_ws/devel/lib/python3/dist-packages')
from cv_tracker.msg import BoundingBox, Target

# Встраиваем функции конфигурации OpenCV
def configure_opencv():
    """Настройка OpenCV для избежания проблем с Mali GPU на ARM устройствах"""
    try:
        # АГРЕССИВНОЕ отключение всех GPU ускорений
        cv2.ocl.setUseOpenCL(False)
        
        # Отключаем все возможные GPU backends
        os.environ['OPENCV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2_0'] = '0'
        os.environ['OPENCV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2_1'] = '0'
        os.environ['OPENCV_DNN_OPENCL'] = '0'
        
        # Критически важные настройки для Mali GPU
        os.environ['OPENCV_OPENCL_DEVICE'] = 'disabled'
        os.environ['OPENCV_OCL_RUNTIME'] = ''
        os.environ['OPENCL_VENDOR_PATH'] = '/dev/null'
        
        # Отключаем TBB для избежания проблем многопоточности
        os.environ['OPENCV_FOR_THREADS_NUM'] = '1'
        
        # Принудительно используем только CPU
        os.environ['OPENCV_DNN_BACKEND'] = 'opencv'
        os.environ['OPENCV_DNN_TARGET'] = 'cpu'
        
        # Ограничиваем использование памяти
        os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = '89478485'
        
        # Настройки для многопоточности - используем только 1 поток для избежания проблем
        cv2.setNumThreads(1)
        
        # Дополнительная проверка - если OpenCL все еще доступен, принудительно отключаем
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(False)
            try:
                cv2.ocl.finish()
            except:
                pass
        
        print("OpenCV configured for ARM device compatibility")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"OpenCL support: {cv2.ocl.haveOpenCL()}")
        print(f"OpenCL enabled: {cv2.ocl.useOpenCL()}")
        print(f"Number of threads: {cv2.getNumThreads()}")
        
    except Exception as e:
        print(f"Error configuring OpenCV: {e}")

def create_safe_tracker():
    """Создание безопасного трекера для ARM устройств"""
    try:
        # Попробуем создать CSRT трекер (лучшая точность)
        tracker = cv2.legacy.TrackerCSRT_create()
        print("Created CSRT tracker")
        return tracker
    except AttributeError:
        try:
            # Альтернатива для новых версий OpenCV
            tracker = cv2.TrackerCSRT_create()
            print("Created CSRT tracker (new API)")
            return tracker
        except:
            pass
    except Exception as e:
        print(f"Failed to create CSRT tracker: {e}")
    
    try:
        # Fallback к KCF трекеру
        tracker = cv2.legacy.TrackerKCF_create()
        print("Created KCF tracker (fallback)")
        return tracker
    except AttributeError:
        try:
            tracker = cv2.TrackerKCF_create()
            print("Created KCF tracker (new API, fallback)")
            return tracker
        except:
            pass
    except Exception as e:
        print(f"Failed to create KCF tracker: {e}")
    
    try:
        # Последний fallback к MOSSE трекеру (самый быстрый)
        tracker = cv2.legacy.TrackerMOSSE_create()
        print("Created MOSSE tracker (last fallback)")
        return tracker
    except AttributeError:
        try:
            tracker = cv2.TrackerMOSSE_create()
            print("Created MOSSE tracker (new API, last fallback)")
            return tracker
        except:
            pass
    except Exception as e:
        print(f"Failed to create MOSSE tracker: {e}")
    
    print("Failed to create any tracker!")
    return None


faces_dir = "/root/tracking_ws/src/cv_tracker/faces"
model_path = "/root/tracking_ws/src/cv_tracker/models/facenet_test.rknn"

# Настройка стандартного логгера Python вместо ROS-логгера
logger = logging.getLogger('face_recognizer')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class FaceRecognizer:
    def __init__(self):
        rospy.init_node('face_recognizer', anonymous=True)
        
        # Настраиваем OpenCV для избежания проблем с Mali GPU
        configure_opencv()
        
        self.image_sub = rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.image_callback)
        self.target_sub = rospy.Subscriber('/face_detection/targets', Target, self.target_callback)
        
        # Добавляем publisher для изображения с распознанными лицами
        self.annotated_image_pub = rospy.Publisher('/annotated_image/compressed', CompressedImage, queue_size=1)
        self.target_face_pub = rospy.Publisher('/face_recognizer/target_face', Target, queue_size=1)
        
        # Добавляем publishers для трекера
        self.tracker_center_pub = rospy.Publisher('/face_tracker/center', Point, queue_size=1)
        self.tracking_status_pub = rospy.Publisher('/face_tracker/status', Bool, queue_size=1)
        
        # Подписываемся на команды управления трекером
        self.stop_cmd_sub = rospy.Subscriber('/face_tracker/stop_command', Bool, self.stop_command_callback)

        # Инициализация модели RKNN для распознавания лиц
        from cv_tracker.rknn_executor import RKNN_model_container
        self.model = RKNN_model_container(model_path, target="rk3588")
        
        # Инициализация моста для преобразования ROS изображений в формат OpenCV
        self.bridge = CvBridge()
        
        # Очереди для многопоточной обработки
        self.image_queue = queue.Queue(maxsize=2)  # Очередь для изображений
        self.target_queue = queue.Queue(maxsize=5)  # Очередь для целей
        
        # Переменные для трекера с thread-safe доступом
        self.tracker_lock = threading.RLock()
        self.tracker = None
        self.tracking_active = False
        self.tracking_box = None
        self.tracked_face_name = None
        self.last_successful_update = time.time()
        
        # Буфер последнего изображения
        self.current_image = None
        self.image_lock = threading.RLock()
        
        # Загрузка эталонных лиц для распознавания
        self.reference_faces = {}
        self.load_reference_faces()
        
        # Порог расстояния для распознавания лица
        self.recognition_threshold = 1.0
        
        # Запуск потоков обработки
        self.start_processing_threads()
        
        print("Распознаватель лиц с оптимизированным трекингом инициализирован")
    
    def start_processing_threads(self):
        """Запуск потоков для обработки изображений и трекинга"""
        # Поток для обработки изображений
        self.image_thread = threading.Thread(target=self._process_images)
        self.image_thread.daemon = True
        self.image_thread.start()
        
        # Поток для обработки целей (распознавание лиц)
        self.target_thread = threading.Thread(target=self._process_targets)
        self.target_thread.daemon = True
        self.target_thread.start()
        
        # Поток для обновления трекера
        self.tracker_thread = threading.Thread(target=self._update_tracker_loop)
        self.tracker_thread.daemon = True
        self.tracker_thread.start()
        
        print("✓ Потоки обработки запущены")
    
    def _process_images(self):
        """Фоновая обработка изображений"""
        while not rospy.is_shutdown():
            try:
                msg = self.image_queue.get(timeout=0.1)
                
                # Преобразование сжатого изображения в формат OpenCV
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                # Обновляем текущее изображение thread-safe способом
                with self.image_lock:
                    self.current_image = cv_image
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка при обработке изображения: {e}")
    
    def _process_targets(self):
        """Фоновая обработка обнаруженных лиц"""
        while not rospy.is_shutdown():
            try:
                msg_data = self.target_queue.get(timeout=0.1)
                msg, current_image = msg_data
                
                if current_image is None:
                    continue
                    
                # Создаем копию изображения для рисования
                annotated_image = current_image.copy()
                
                # Рисуем рамку трекера, если активен
                with self.tracker_lock:
                    if self.tracking_active and self.tracking_box is not None:
                        x, y, w, h = self.tracking_box
                        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Синяя рамка для трекера
                        
                        # Добавляем подпись трекера
                        if self.tracked_face_name:
                            tracker_label = f"TRACKING: {self.tracked_face_name}"
                            cv2.putText(annotated_image, tracker_label, (x, y - 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                best_face_for_tracking = None
                best_distance = float('inf')
                
                for box in msg.boxes:
                    try:
                        # Извлечение координат bbox
                        center_x = int(box.center.x)
                        center_y = int(box.center.y)
                        size_x = int(box.size_x)
                        size_y = int(box.size_y)
                        
                        # Вычисление координат прямоугольника
                        x1 = max(0, center_x - size_x // 2)
                        y1 = max(0, center_y - size_y // 2)
                        x2 = min(current_image.shape[1], center_x + size_x // 2)
                        y2 = min(current_image.shape[0], center_y + size_y // 2)
                        
                        # Вырезание области лица
                        face_img = current_image[y1:y2, x1:x2]
                        
                        # Пропуск, если область слишком маленькая
                        if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                            continue
                            
                        # Изменение размера для facenet
                        face_resized = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_CUBIC)
                        
                        # Получение эмбеддинга лица
                        face_embedding = self.get_face_embedding(face_resized)
                        
                        if face_embedding is not None:
                            # Распознавание лица
                            name, distance = self.recognize_face(face_embedding)
                            
                            if name and distance < self.recognition_threshold:
                                print(f"Распознано лицо: {name}, расстояние: {distance:.4f}")
                                
                                # Рисуем рамку и подпись на изображении (зеленая для распознанного лица)
                                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Добавляем имя над рамкой
                                label = f"{name} ({distance:.2f})"
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
                                cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                                
                                # Сохраняем лучшее лицо для инициализации трекера
                                if distance < best_distance:
                                    best_distance = distance
                                    best_face_for_tracking = {
                                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                                        'name': name,
                                        'distance': distance
                                    }
                                
                            else:
                                print(f"Лицо не распознано, минимальное расстояние: {distance:.4f}")
                                # Не рисуем рамку для нераспознанного лица
                                
                    except Exception as e:
                        print(f"Ошибка при обработке лица: {e}")
                
                # Инициализируем или переинициализируем трекер для лучшего лица
                if best_face_for_tracking is not None:
                    with self.tracker_lock:
                        if not self.tracking_active:
                            # Инициализируем трекер для нового лица
                            self.initialize_tracker_unsafe(best_face_for_tracking['bbox'], 
                                                         best_face_for_tracking['name'], 
                                                         current_image)
                        elif self.tracked_face_name != best_face_for_tracking['name']:
                            # Переинициализируем трекер для другого лица с лучшим распознаванием
                            print(f"Переключение трекинга с {self.tracked_face_name} на {best_face_for_tracking['name']}")
                            self.initialize_tracker_unsafe(best_face_for_tracking['bbox'], 
                                                         best_face_for_tracking['name'], 
                                                         current_image)
                
                # Отправляем аннотированное изображение
                msg_out = CompressedImage()
                msg_out.header.stamp = rospy.Time.now()
                msg_out.format = "jpeg"
                msg_out.data = np.array(cv2.imencode('.jpg', annotated_image)[1]).tobytes()
                self.annotated_image_pub.publish(msg_out)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка при обработке целей: {e}")
    
    def _update_tracker_loop(self):
        """Цикл обновления трекера в отдельном потоке"""
        rate = rospy.Rate(30)  # 30 Hz для плавного трекинга
        
        while not rospy.is_shutdown():
            try:
                with self.tracker_lock:
                    if self.tracking_active and self.tracker is not None:
                        with self.image_lock:
                            current_image = self.current_image
                        
                        if current_image is not None:
                            success = self.update_tracker_unsafe(current_image)
                            if success:
                                self.last_successful_update = time.time()
                            else:
                                # Если трекер не удается обновить длительное время, продолжаем попытки
                                # Убираем автоматическую остановку трекера
                                pass
                
                rate.sleep()
                
            except Exception as e:
                print(f"Ошибка в цикле трекера: {e}")
                rate.sleep()
    
    def load_reference_faces(self):
        """Загрузка эталонных лиц из директории faces_dir"""
        if not os.path.exists(faces_dir):
            print(f"Директория с лицами не найдена: {faces_dir}")
            return
            
        for filename in os.listdir(faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(faces_dir, filename)
                
                try:
                    # Загрузка и предобработка изображения
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Не удалось загрузить изображение: {image_path}")
                        continue
                        
                    # Изменение размера для facenet
                    img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_CUBIC)
                    
                    # Получение эмбеддинга лица
                    embedding = self.get_face_embedding(img)
                    if embedding is not None:
                        self.reference_faces[name] = embedding
                        print(f"Загружено эталонное лицо: {name}")
                except Exception as e:
                    print(f"Ошибка при обработке {image_path}: {e}")
        
        print(f"Загружено {len(self.reference_faces)} эталонных лиц")
        
    def get_face_embedding(self, face_img):
        """Получение эмбеддинга (вектора признаков) для лица"""
        try:
            # Расширяем размерность для batch (NHWC формат)
            input_data = np.expand_dims(face_img, axis=0)
            # Получение эмбеддинга
            result = self.model.run(input_data)
            if result is None or len(result) == 0:
                print("Модель вернула пустой результат")
                return None
            # Преобразование в numpy массив
            outputs = np.array(result[0])
            # Нормализация результата
            outputs = preprocessing.normalize(outputs, norm='l2')
            return outputs
        except Exception as e:
            print(f"Ошибка при получении эмбеддинга: {e}")
            return None
    
    def recognize_face(self, face_embedding):
        """Распознавание лица по эмбеддингу"""
        if not self.reference_faces:
            return None, float('inf')
            
        min_distance = float('inf')
        recognized_name = None
        
        for name, ref_embedding in self.reference_faces.items():
            try:
                # Вычисление евклидова расстояния между эмбеддингами
                distance = np.linalg.norm(face_embedding - ref_embedding)
                
                if distance < min_distance:
                    min_distance = distance
                    recognized_name = name
            except Exception as e:
                print(f"Ошибка при сравнении с эталоном {name}: {e}")
        
        return recognized_name, min_distance
    
    def initialize_tracker_unsafe(self, bbox, face_name, current_image):
        """Инициализация трекера без блокировки (вызывается изнутри блокировки)"""
        try:
            # Создаем новый трекер
            self.tracker = create_safe_tracker()
            if self.tracker is None:
                print("Не удалось создать трекер")
                return False
                
            # Инициализируем трекер
            x, y, w, h = bbox
            success = self.tracker.init(current_image, (x, y, w, h))
            
            if success:
                self.tracking_active = True
                self.tracking_box = bbox
                self.tracked_face_name = face_name
                self.last_successful_update = time.time()
                print(f"Трекер инициализирован для лица: {face_name}")
                self.tracking_status_pub.publish(Bool(True))
                
                # Публикуем первоначальные данные трекера в target_face топик
                center_x = x + w // 2
                center_y = y + h // 2
                
                bbox_msg = BoundingBox()
                bbox_msg.center.x = center_x
                bbox_msg.center.y = center_y
                bbox_msg.size_x = w
                bbox_msg.size_y = h
                bbox_msg.name = face_name
                
                target_msg = Target()
                target_msg.image_height = current_image.shape[0]
                target_msg.image_width = current_image.shape[1]
                target_msg.boxes.append(bbox_msg)
                
                self.target_face_pub.publish(target_msg)
                
                return True
            else:
                print("Не удалось инициализировать трекер")
                return False
                
        except Exception as e:
            print(f"Ошибка при инициализации трекера: {e}")
            return False
    
    def update_tracker_unsafe(self, current_image):
        """Обновление трекера без блокировки (вызывается изнутри блокировки)"""
        try:
            # Обновляем трекер
            success, bbox = self.tracker.update(current_image)
            
            if success:
                x, y, w, h = [int(v) for v in bbox]
                
                # Проверяем, что координаты валидны
                height, width = current_image.shape[:2]
                if x >= 0 and y >= 0 and x + w <= width and y + h <= height and w > 0 and h > 0:
                    self.tracking_box = (x, y, w, h)
                    
                    # Публикуем центр трекера
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    center_msg = Point()
                    center_msg.x = center_x
                    center_msg.y = center_y
                    center_msg.z = 0
                    self.tracker_center_pub.publish(center_msg)
                    
                    # Публикуем данные рамки трекера в target_face топик
                    if self.tracked_face_name:
                        bbox_msg = BoundingBox()
                        bbox_msg.center.x = center_x
                        bbox_msg.center.y = center_y
                        bbox_msg.size_x = w
                        bbox_msg.size_y = h
                        bbox_msg.name = self.tracked_face_name
                        
                        target_msg = Target()
                        target_msg.image_height = height
                        target_msg.image_width = width
                        target_msg.boxes.append(bbox_msg)
                        
                        self.target_face_pub.publish(target_msg)
                    
                    self.tracking_status_pub.publish(Bool(True))
                    return True
                else:
                    print("Трекер вернул некорректные координаты")
                    return False
            else:
                print("Трекинг временно потерян")
                return False
                
        except Exception as e:
            print(f"Ошибка при обновлении трекера: {e}")
            return False
    
    def stop_tracking(self):
        """Остановка трекинга"""
        with self.tracker_lock:
            self.tracking_active = False
            self.tracker = None
            self.tracking_box = None
            self.tracked_face_name = None
        self.tracking_status_pub.publish(Bool(False))
        print("Трекинг остановлен")
    
    def stop_command_callback(self, msg):
        """Обработка команды остановки трекинга"""
        if msg.data:
            with self.tracker_lock:
                if self.tracking_active:
                    print("Получена команда остановки трекинга")
                    self.stop_tracking()
        
    def image_callback(self, msg):
        """Обработка входящего изображения"""
        try:
            # Добавляем изображение в очередь для обработки
            self.image_queue.put(msg, block=False)
        except queue.Full:
            # Если очередь полная, пропускаем кадр для избежания задержек
            pass
        except Exception as e:
            print(f"Ошибка при добавлении изображения в очередь: {e}")

    def target_callback(self, msg):
        """Обработка обнаруженных лиц"""
        try:
            # Получаем текущее изображение
            with self.image_lock:
                current_image = self.current_image
            
            # Добавляем цель в очередь для обработки
            self.target_queue.put((msg, current_image), block=False)
        except queue.Full:
            # Если очередь полная, пропускаем обработку
            pass
        except Exception as e:
            print(f"Ошибка при добавлении цели в очередь: {e}")

if __name__ == '__main__':
    try:
        face_recognizer = FaceRecognizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
