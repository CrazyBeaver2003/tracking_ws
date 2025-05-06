import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import sys
sys.path.insert(0, '/root/tracking_ws/devel/lib/python3/dist-packages')
from cv_tracker.msg import BoundingBox, Target


faces_dir = "/root/tracking_ws/src/cv_tracker/faces"
model_path = "/root/tracking_ws/src/cv_tracker/models/facenet_test.rknn"

class FaceRecognizer:
    def __init__(self):
        rospy.init_node('face_recognizer', anonymous=True)
        self.image_sub = rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.image_callback)
        self.target_sub = rospy.Subscriber('/face_detection/targets', Target, self.target_callback)
        
        from cv_tracker.rknn_executor import RKNN_model_container
        
    def image_callback(self, msg):
        pass

    def target_callback(self, msg):
        for box in msg.boxes:
            print(f"Получены координаты лица {box.name}")
            print(box.center.x, box.center.y, box.size_x, box.size_y)
        

if __name__ == '__main__':
    try:
        face_recognizer = FaceRecognizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
