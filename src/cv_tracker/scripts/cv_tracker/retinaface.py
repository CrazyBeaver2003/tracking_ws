from rknnlite.api import RKNNLite
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

class FaceDetector:
    def __init__(self):

        # Инициализация ROS параметров
        rospy.init_node('face_detector', anonymous=True, log_level=rospy.INFO)
        self.image_sub = rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.image_callback)
        
        # Publisher для обработанного изображения
        self.processed_image_pub = rospy.Publisher('/face_detection/image/compressed', CompressedImage, queue_size=10)

        #CV initialization
        self.bridge = CvBridge()
        self.bbox = None

        # Инициализация RKNNLite
        rknn = RKNNLite()

        self.model_path ="/home/alex/Documents/tracking_ws/src/cv_tracker/models/retinaface.rknn"
        rknn.load_rknn(self.model_path)

        ret = rknn.init_runtime()
        if ret != 0:
            rospy.logerr("Init runtime environment failed")
            exit(ret)
        rospy.loginfo("done")

        self.rknn = rknn
        rospy.loginfo("Инициализация FaceDetector завершена")

    def rknn_run(self, inputs):
        if self.rknn is None:
            rospy.logerr("ERROR: rknn has been released")
            return []

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)

        return result

    def rknn_release(self):
        self.rknn.release()
        self.rknn = None


    def image_callback(self, msg):
        """Обработчик входящих изображений"""
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
            current_time = rospy.Time.now()

            # Преобразование изображения в формат, ожидаемый моделью
            cv_image = cv2.resize(cv_image, (640, 640))
            
            outputs = self.rknn_run(cv_image)

            loc, conf, landm = outputs

            # Обработка результатов
        except Exception as e:
            rospy.logerr(f"Ошибка при обработке изображения: {e}")
            raise



if __name__ == '__main__':
    try:
        detector = FaceDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass









