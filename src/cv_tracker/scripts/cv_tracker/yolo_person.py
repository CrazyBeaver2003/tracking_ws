import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import sys
import os
import threading
import queue
import time
from geometry_msgs.msg import Pose2D
sys.path.insert(0, '/root/tracking_ws/devel/lib/python3/dist-packages')
from cv_tracker.msg import BoundingBox, Target

# Добавляем путь к py_utils и coco_utils (если потребуется)
sys.path.append('/root/vision_ros1/src/eagle_eye_vision/scripts')
from eagle_eye_vision.coco_utils import COCO_test_helper


model_path = '/root/tracking_ws/src/cv_tracker/models/yolo11n.rknn'
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)

CLASSES = (
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
)

def postprocess_yolo_onnx(output, obj_thresh=OBJ_THRESH, nms_thresh=NMS_THRESH):
    # output: (1, 84, 8400)
    output = np.squeeze(output)  # (84, 8400)
    boxes = output[:4, :]        # (4, 8400)
    scores = output[4:, :]       # (80, 8400)

    # Преобразуем боксы из xywh в xyxy
    x, y, w, h = boxes
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)  # (8400, 4)

    # Для каждого бокса — максимальный класс и его score
    class_ids = np.argmax(scores, axis=0)
    class_scores = np.max(scores, axis=0)

    # Фильтруем по порогу
    mask = class_scores > obj_thresh
    boxes_xyxy = boxes_xyxy[mask]
    class_ids = class_ids[mask]
    class_scores = class_scores[mask]

    # NMS (только для людей)
    person_mask = class_ids == 0
    boxes_xyxy = boxes_xyxy[person_mask]
    class_scores = class_scores[person_mask]
    class_ids = class_ids[person_mask]

    if len(boxes_xyxy) == 0:
        return None, None, None

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xyxy.tolist(),
        scores=class_scores.tolist(),
        score_threshold=obj_thresh,
        nms_threshold=nms_thresh
    )
    if len(indices) == 0:
        return None, None, None
    indices = indices.flatten()
    return boxes_xyxy[indices], class_ids[indices], class_scores[indices]

def calculate_intersection_area(box1, box2):
    """Рассчет площади пересечения между двумя прямоугольниками"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Находим координаты пересечения
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Площадь пересечения
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area

def calculate_iou(box1, box2):
    """Рассчет IoU (Intersection over Union) между двумя прямоугольниками"""
    intersection_area = calculate_intersection_area(box1, box2)
    if intersection_area == 0:
        return 0.0
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Площади прямоугольников
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # IoU
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0

def face_inside_person(face_box, person_box, overlap_threshold=0.5):
    """Проверяет, находится ли больше половины лица внутри тела человека"""
    face_x1 = face_box["x1"]
    face_y1 = face_box["y1"]
    face_x2 = face_box["x2"]
    face_y2 = face_box["y2"]
    
    person_x1, person_y1, person_x2, person_y2 = person_box
    
    # Создаем боксы для расчета
    face_bbox = [face_x1, face_y1, face_x2, face_y2]
    person_bbox = [person_x1, person_y1, person_x2, person_y2]
    
    # Рассчитываем площадь пересечения
    intersection_area = calculate_intersection_area(face_bbox, person_bbox)
    
    # Рассчитываем площадь лица
    face_area = (face_x2 - face_x1) * (face_y2 - face_y1)
    
    if face_area == 0:
        return False
    
    # Проверяем, что больше половины лица внутри тела человека
    overlap_ratio = intersection_area / face_area
    
    # Дополнительная проверка: лицо должно быть в верхней половине тела
    person_height = person_y2 - person_y1
    face_center_y = (face_y1 + face_y2) / 2
    upper_half = person_y1 + person_height * 0.6  # Расширяем до 60% для гибкости
    
    is_in_upper_part = face_center_y < upper_half
    has_sufficient_overlap = overlap_ratio > overlap_threshold
    
    return has_sufficient_overlap and is_in_upper_part

class YOLO11_ROS:
    def __init__(self):
        rospy.init_node('yolo11_ros', anonymous=True)
        print("Инициализация YOLO11_ROS с многопоточностью и трекером...")
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.image_callback)
        self.target_sub = rospy.Subscriber('/face_recognizer/target_face', Target, self.target_callback)
        self.pub = rospy.Publisher('/yolo/detections', Target, queue_size=1)
        self.target_person_pub = rospy.Publisher('/yolo/target_person', Target, queue_size=1)
        self.image_pub = rospy.Publisher('/yolo/annotated_image/compressed', CompressedImage, queue_size=1)
        
        # Очередь для обработки кадров (как в yolo_face)
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Инициализация RKNN модели
        from cv_tracker.rknn_executor import RKNN_model_container
        self.model = RKNN_model_container(model_path, target="rk3588")
        self.co_helper = COCO_test_helper(enable_letter_box=True)
        
        # Переменные состояния
        self.face_target = None
        self.face_dict = {}
        self.last_image_time = None
        self.last_face_time = None
        
        # Переменные для трекера
        self.tracker = None
        self.tracker_initialized = False
        self.tracked_person_name = ""
        self.tracker_bbox = None
        self.face_lost_timeout = 10.0  # Увеличиваем время до 10 секунд
        self.tracker_confidence_threshold = 0.1  # Снижаем порог до 0.1
        self.tracker_fail_count = 0  # Счетчик неудачных обновлений
        self.max_tracker_fails = 5  # Максимум неудач перед сбросом
        self.last_successful_track = None  # Время последнего успешного трекинга
        
        # Запуск потока обработки
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("YOLO11_ROS инициализирован с многопоточностью и трекером")

    def target_callback(self, msg):
        """Callback для получения распознанного лица"""
        if len(msg.boxes) > 0:
            face_box = msg.boxes[0]
            self.face_target = msg
            self.face_dict = {
                "x1": face_box.center.x - face_box.size_x/2,
                "y1": face_box.center.y - face_box.size_y/2, 
                "x2": face_box.center.x + face_box.size_x/2,
                "y2": face_box.center.y + face_box.size_y/2,
                "name": face_box.name
            }
            self.last_face_time = time.time()
            print(f"Получено распознанное лицо: {self.face_dict['name']} в координатах ({self.face_dict['x1']:.1f}, {self.face_dict['y1']:.1f}, {self.face_dict['x2']:.1f}, {self.face_dict['y2']:.1f})")



    def draw_boxes(self, image, person_box, face_name, score, is_tracked=False):
        """Отрисовка ограничивающих рамок на изображении"""
        img_with_boxes = image.copy()
        if person_box is not None:
            x1, y1, x2, y2 = map(int, person_box)
            
            # Выбираем цвет в зависимости от источника данных
            if is_tracked:
                color = (0, 165, 255)  # Оранжевый для трекера
                text = f"{face_name} (Tracked): {score:.2f}"
            else:
                color = (255, 0, 0)  # Синий для YOLO
                text = f"{face_name}: {score:.2f}"
            
            # Рисуем рамку
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
            
            # Добавляем текст с именем лица
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Фон для текста
            cv2.rectangle(img_with_boxes, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0], y1), color, -1)
            
            # Текст
            cv2.putText(img_with_boxes, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                       
        return img_with_boxes

    def image_callback(self, msg):
        """Обработчик входящих изображений - добавляет в очередь"""
        try:
            # Пытаемся добавить кадр в очередь, если она не полная
            self.frame_queue.put(msg, block=False)
        except queue.Full:
            # Если очередь полная, пропускаем кадр
            pass

    def _process_frames(self):
        """Фоновая обработка кадров"""
        while not rospy.is_shutdown():
            try:
                # Получаем кадр из очереди с таймаутом
                msg = self.frame_queue.get(timeout=0.1)
                
                # Обработка кадра
                self._process_single_frame(msg)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка при обработке кадра: {e}")

    def _process_single_frame(self, msg):
        """Обработка одного кадра"""
        self.last_image_time = rospy.Time.now()
        
        try:
            # Преобразование compressed сообщения в изображение OpenCV
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Ошибка декодирования изображения: {e}")
            return
            
        img_height, img_width = img.shape[:2]
        
        # Проверяем таймаут данных о лице
        current_time = time.time()
        if (self.last_face_time is not None and 
            current_time - self.last_face_time > self.face_lost_timeout):
            # Очищаем данные о лице по таймауту
            print(f"Таймаут данных о лице: {current_time - self.last_face_time:.1f}с")
            self.face_dict = {}
            self.last_face_time = None
        
        # Запускаем YOLO детекцию
        img_pre = self.co_helper.letter_box(img.copy(), IMG_SIZE, pad_color=(0,0,0))
        img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_pre, axis=0)
        outputs = self.model.run([input_data])
        boxes, classes, scores = postprocess_yolo_onnx(outputs[0])
        
        print(f"DEBUG: YOLO обнаружил {len(boxes) if boxes is not None else 0} людей")
        
        # Публикуем все обнаруженные объекты
        msg_out = Target()
        msg_out.image_height = float(img_height)
        msg_out.image_width = float(img_width)

        target_person_found = False
        target_person_box = None
        target_face_name = "Unknown"
        is_from_tracker = False

        if boxes is not None:
            real_boxes = self.co_helper.get_real_box(boxes)
            
            # Сначала добавляем все обнаруженные объекты в общее сообщение
            for box, score, cl in zip(real_boxes, scores, classes):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                size_x = abs(x2 - x1)
                size_y = abs(y2 - y1)
                area = size_x * size_y
                
                bbox = BoundingBox()
                bbox.center = Pose2D(x=center_x, y=center_y, theta=0.0)
                bbox.area = float(area)
                bbox.size_x = float(size_x)
                bbox.size_y = float(size_y)
                bbox.name = CLASSES[0]  # Только люди
                msg_out.boxes.append(bbox)
            
            # Пытаемся найти соответствие между лицом и телом (если есть данные о лице)
            if len(self.face_dict) > 0:
                print(f"DEBUG: Ищем соответствие для лица {self.face_dict['name']}")
                best_overlap = 0
                best_person_box = None
                
                for i, box in enumerate(real_boxes):
                    face_inside = face_inside_person(self.face_dict, box)
                    print(f"DEBUG: Проверяем человека {i}: face_inside={face_inside}")
                    
                    if face_inside:
                        # Рассчитываем площадь пересечения для выбора лучшего совпадения
                        face_bbox = [self.face_dict["x1"], self.face_dict["y1"], 
                                   self.face_dict["x2"], self.face_dict["y2"]]
                        intersection = calculate_intersection_area(face_bbox, box)
                        face_area = (self.face_dict["x2"] - self.face_dict["x1"]) * (self.face_dict["y2"] - self.face_dict["y1"])
                        overlap_ratio = intersection / face_area if face_area > 0 else 0
                        
                        print(f"DEBUG: Человек {i} overlap_ratio={overlap_ratio:.3f}")
                        
                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            best_person_box = box
                            target_face_name = self.face_dict["name"]
                
                if best_person_box is not None:
                    target_person_found = True
                    target_person_box = best_person_box
                    is_from_tracker = False
                    print(f"DEBUG: Найден целевой человек! {target_face_name}")
                    
                    # Инициализируем или обновляем трекер только если нужно
                    if (not self.tracker_initialized or 
                        self.tracked_person_name != target_face_name or
                        self.tracker_fail_count > 2):
                        
                        print(f"Переинициализируем трекер для {target_face_name}")
                        if self.init_tracker(img, best_person_box, target_face_name):
                            print(f"Трекер успешно инициализирован")
                        else:
                            print(f"Не удалось инициализировать трекер")
                    
                    print(f"Найден человек с лицом {target_face_name}: центр ({(best_person_box[0]+best_person_box[2])/2:.1f}, {(best_person_box[1]+best_person_box[3])/2:.1f}), перекрытие: {best_overlap:.2f}")
                else:
                    print(f"DEBUG: Целевой человек НЕ найден")
        
        # Если не нашли через YOLO, но трекер активен - используем трекер
        if not target_person_found and self.tracker_initialized:
            tracked_box, confidence = self.update_tracker(img)
            
            if tracked_box is not None and confidence > self.tracker_confidence_threshold:
                target_person_found = True
                target_person_box = tracked_box
                target_face_name = self.tracked_person_name
                is_from_tracker = True
                print(f"Трекер отслеживает {target_face_name}: confidence={confidence:.2f}")
        
        # Проверяем, нужно ли сбросить трекер
        if self.tracker_initialized:
            should_reset, reason = self.should_reset_tracker()
            if should_reset:
                print(f"Сбрасываем трекер: {reason}")
                self.reset_tracker()
        
        # Публикуем результат для найденного человека
        if target_person_found:
            source = "TRACKER" if is_from_tracker else "YOLO"
            print(f"DEBUG: ПУБЛИКУЕМ target_person для {target_face_name} из {source}")
            
            target_msg = Target()
            target_msg.image_height = float(img_height)
            target_msg.image_width = float(img_width)
            
            x1, y1, x2, y2 = target_person_box
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            size_x = abs(x2 - x1)
            size_y = abs(y2 - y1)
            area = size_x * size_y
            
            bbox = BoundingBox()
            bbox.center = Pose2D(x=center_x, y=center_y, theta=0.0)
            bbox.area = float(area)
            bbox.size_x = float(size_x)
            bbox.size_y = float(size_y)
            bbox.name = target_face_name
            target_msg.boxes.append(bbox)
            
            # Публикуем человека с распознанным лицом
            self.target_person_pub.publish(target_msg)
            print(f"DEBUG: Опубликован bbox ({center_x:.1f}, {center_y:.1f}) размером {size_x:.1f}x{size_y:.1f} из {source}")
        else:
            print(f"DEBUG: Нет активного трекинга, ничего не публикуем")
        
        # Отрисовка и публикация изображения
        if target_person_found:
            annotated_image = self.draw_boxes(img, target_person_box, target_face_name, 1.0, is_from_tracker)
        else:
            annotated_image = img.copy()
            # Рисуем всех обнаруженных людей серым цветом если нет целевого
            if boxes is not None:
                for box, score in zip(real_boxes, scores):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (128, 128, 128), 2)
                    cv2.putText(annotated_image, f"Person: {score:.2f}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                
        try:
            msg_img = CompressedImage()
            msg_img.header.stamp = rospy.Time.now()
            msg_img.format = "jpeg"
            msg_img.data = np.array(cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 60])[1]).tobytes()
            self.image_pub.publish(msg_img)
        except Exception as e:
            print(f"Ошибка публикации изображения: {e}")
        
        # Публикуем все обнаружения
        self.pub.publish(msg_out)

    def init_tracker(self, image, bbox, name):
        """Инициализация трекера для отслеживания человека"""
        try:
            # Сбрасываем предыдущий трекер если есть
            if self.tracker is not None:
                self.tracker = None
            
            # Пробуем разные типы трекеров в порядке приоритета
            tracker_types = [
                lambda: cv2.TrackerKCF_create(),
                lambda: cv2.legacy.TrackerKCF_create() if hasattr(cv2, 'legacy') else None,
                lambda: cv2.TrackerCSRT_create() if hasattr(cv2, 'TrackerCSRT_create') else None
            ]
            
            # Конвертируем bbox в формат (x, y, w, h)
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            # Проверяем валидность bbox
            if w <= 0 or h <= 0 or x1 < 0 or y1 < 0:
                print(f"Невалидный bbox для инициализации трекера: {bbox}")
                return False
                
            tracker_bbox = (int(x1), int(y1), int(w), int(h))
            
            # Пробуем создать трекер
            for tracker_creator in tracker_types:
                try:
                    self.tracker = tracker_creator()
                    if self.tracker is None:
                        continue
                        
                    # Инициализируем трекер с таймаутом
                    success = self.tracker.init(image, tracker_bbox)
                    
                    if success:
                        self.tracker_initialized = True
                        self.tracked_person_name = name
                        self.tracker_bbox = tracker_bbox
                        self.tracker_fail_count = 0
                        self.last_successful_track = time.time()
                        print(f"Трекер инициализирован для {name} с bbox: {tracker_bbox}")
                        return True
                    else:
                        self.tracker = None
                        continue
                        
                except Exception as e:
                    print(f"Ошибка при создании трекера типа: {e}")
                    self.tracker = None
                    continue
            
            print("Не удалось создать ни один тип трекера")
            self.tracker_initialized = False
            return False
                
        except Exception as e:
            print(f"Критическая ошибка при создании трекера: {e}")
            self.tracker_initialized = False
            return False

    def update_tracker(self, image):
        """Обновление позиции трекера"""
        if not self.tracker_initialized or self.tracker is None:
            return None, 0.0
            
        try:
            success, bbox = self.tracker.update(image)
            
            if success and len(bbox) == 4:
                x, y, w, h = bbox
                
                # Проверяем валидность результата
                if w <= 0 or h <= 0:
                    self.tracker_fail_count += 1
                    print(f"Трекер вернул невалидный размер: w={w}, h={h}")
                    return None, 0.0
                
                # Конвертируем обратно в формат (x1, y1, x2, y2)
                tracker_result = [float(x), float(y), float(x + w), float(y + h)]
                
                # Проверяем границы изображения
                img_h, img_w = image.shape[:2]
                confidence = 1.0
                
                # Корректируем bbox если он выходит за границы
                if x < 0:
                    tracker_result[0] = 0
                    confidence *= 0.8
                if y < 0:
                    tracker_result[1] = 0
                    confidence *= 0.8
                if x + w > img_w:
                    tracker_result[2] = img_w
                    confidence *= 0.8
                if y + h > img_h:
                    tracker_result[3] = img_h
                    confidence *= 0.8
                    
                # Проверяем размер bbox
                final_w = tracker_result[2] - tracker_result[0]
                final_h = tracker_result[3] - tracker_result[1]
                
                if final_w < 20 or final_h < 20:  # Слишком маленький bbox
                    confidence *= 0.5
                    
                if final_w > img_w * 0.8 or final_h > img_h * 0.8:  # Слишком большой bbox
                    confidence *= 0.6
                
                # Успешное обновление
                self.tracker_fail_count = 0
                self.last_successful_track = time.time()
                
                return tracker_result, confidence
            else:
                self.tracker_fail_count += 1
                print(f"Трекер не смог обновиться (fail #{self.tracker_fail_count})")
                return None, 0.0
                
        except Exception as e:
            self.tracker_fail_count += 1
            print(f"Ошибка обновления трекера: {e} (fail #{self.tracker_fail_count})")
            return None, 0.0

    def reset_tracker(self):
        """Сброс трекера"""
        try:
            if self.tracker is not None:
                self.tracker = None
        except:
            pass
        
        self.tracker_initialized = False
        self.tracked_person_name = ""
        self.tracker_bbox = None
        self.tracker_fail_count = 0
        self.last_successful_track = None
        print("Трекер сброшен")
        
    def should_reset_tracker(self):
        """Проверяет, нужно ли сбросить трекер"""
        current_time = time.time()
        
        # Сброс по таймауту без данных о лице
        if (self.last_face_time is not None and 
            current_time - self.last_face_time > self.face_lost_timeout):
            return True, "Таймаут без данных о лице"
            
        # Сброс по количеству неудач
        if self.tracker_fail_count >= self.max_tracker_fails:
            return True, f"Превышен лимит неудач: {self.tracker_fail_count}"
            
        # Сброс если долго нет успешного трекинга
        if (self.last_successful_track is not None and 
            current_time - self.last_successful_track > 5.0):
            return True, "Долго нет успешного трекинга"
            
        return False, ""

if __name__ == '__main__':
    try:
        node = YOLO11_ROS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 