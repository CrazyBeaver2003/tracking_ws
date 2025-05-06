import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import sys
import os
import logging
from sklearn import preprocessing
sys.path.insert(0, '/root/tracking_ws/devel/lib/python3/dist-packages')
from cv_tracker.msg import BoundingBox, Target


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
        self.image_sub = rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.image_callback)
        self.target_sub = rospy.Subscriber('/face_detection/targets', Target, self.target_callback)
        
        # Инициализация модели RKNN для распознавания лиц
        from cv_tracker.rknn_executor import RKNN_model_container
        self.model = RKNN_model_container(model_path, target="rk3588")
        
        # Инициализация моста для преобразования ROS изображений в формат OpenCV
        self.bridge = CvBridge()
        
        # Последнее полученное изображение
        self.current_image = None
        
        # Загрузка эталонных лиц для распознавания
        self.reference_faces = {}
        self.load_reference_faces()
        
        # Порог расстояния для распознавания лица
        self.recognition_threshold = 0.9
        
        print("Распознаватель лиц инициализирован")
    
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
        
    def image_callback(self, msg):
        """Обработка входящего изображения"""
        try:
            # Преобразование сжатого изображения в формат OpenCV
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.current_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Ошибка при обработке изображения: {e}")

    def target_callback(self, msg):
        """Обработка обнаруженных лиц"""
        if self.current_image is None:
            print("Изображение не получено")
            return
            
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
                x2 = min(self.current_image.shape[1], center_x + size_x // 2)
                y2 = min(self.current_image.shape[0], center_y + size_y // 2)
                
                # Вырезание области лица
                face_img = self.current_image[y1:y2, x1:x2]
                
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
                    else:
                        print(f"Лицо не распознано, минимальное расстояние: {distance:.4f}")
                        
            except Exception as e:
                print(f"Ошибка при обработке лица: {e}")

if __name__ == '__main__':
    try:
        face_recognizer = FaceRecognizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
