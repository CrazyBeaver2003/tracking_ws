import os
import cv2
import sys
import numpy as np

# Добавляем путь к py_utils и coco_utils (если потребуется)
sys.path.append('/root/vision_ros1/src/eagle_eye_vision/scripts')
from eagle_eye_vision.coco_utils import COCO_test_helper
from cv_tracker.rknn_executor import RKNN_model_container

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

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)

def dfl(position):
    import torch
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y = y.softmax(2)
    acc_metrix = torch.arange(mc).float().reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y.numpy()

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    return xyxy

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))
    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)
    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
    if not nclasses and not nscores:
        return None, None, None
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores

# --- Новый постпроцесс для выхода (1, 84, 8400) ---
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

def main():
    faces_dir = os.path.join(os.path.dirname(__file__), '../faces')
    file_list = sorted(os.listdir(faces_dir))
    img_list = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not img_list:
        print('Нет изображений в папке faces')
        return
    model = RKNN_model_container(model_path, target="rk3588")
    co_helper = COCO_test_helper(enable_letter_box=True)
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    for img_name in img_list:
        img_path = os.path.join(faces_dir, img_name)
        img_src = cv2.imread(img_path)
        if img_src is None:
            print(f'Не удалось загрузить {img_name}')
            continue
        img = co_helper.letter_box(img_src.copy(), IMG_SIZE, pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img, axis=0)
        outputs = model.run([input_data])
        # Для вашего выхода используем только первый аутпут
        boxes, classes, scores = postprocess_yolo_onnx(outputs[0])
        img_vis = img_src.copy()
        if boxes is not None:
            real_boxes = co_helper.get_real_box(boxes)
            found = False
            for box, score, cl in zip(real_boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                print(f"{img_name}: Найден человек bbox=({x1},{y1},{x2},{y2}), score={score:.2f}")
                # Рисуем бокс
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_vis, f"person {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                found = True
            if not found:
                print(f"{img_name}: Люди не найдены")
        else:
            print(f"{img_name}: Люди не найдены")
        # Сохраняем результат
        result_path = os.path.join(results_dir, img_name)
        cv2.imwrite(result_path, img_vis)
    model.release()

if __name__ == '__main__':
    main() 