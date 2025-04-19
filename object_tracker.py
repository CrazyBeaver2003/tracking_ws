import cv2
import numpy as np

class ObjectTracker:
    def __init__(self):
        self.tracker = cv2.TrackerCSRT_create()
        self.bbox = None
        self.tracking = False

    def select_roi(self, frame):
        """Позволяет пользователю выбрать область для трекинга"""
        self.bbox = cv2.selectROI("Выберите объект для трекинга", frame, False)
        cv2.destroyWindow("Выберите объект для трекинга")
        return self.bbox

    def init_tracker(self, frame, bbox):
        """Инициализация трекера с выбранной областью"""
        self.tracker = cv2.TrackerCSRT_create()
        self.tracking = self.tracker.init(frame, bbox)
        return self.tracking

    def update(self, frame):
        """Обновление позиции трекера на новом кадре"""
        if not self.tracking:
            return None, frame

        success, bbox = self.tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return bbox, frame
        return None, frame

def main():
    # Инициализация захвата видео (0 - веб-камера)
    cap = cv2.VideoCapture(0)
    tracker = ObjectTracker()
    
    # Чтение первого кадра
    ret, frame = cap.read()
    if not ret:
        print("Ошибка при чтении видео")
        return

    # Выбор области для трекинга
    bbox = tracker.select_roi(frame)
    tracker.init_tracker(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обновление трекера
        bbox, frame = tracker.update(frame)

        # Отображение результата
        cv2.imshow("Трекинг объекта", frame)

        # Выход по клавише 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()