import cv2
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


class ImagePublisher:
    def __init__(self):
        rospy.init_node('image_publisher', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('/camera/image_raw/compressed', CompressedImage, queue_size=30)
        self.cap = cv2.VideoCapture(0)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            rospy.logerr("Не удалось открыть камеру")
            return

    def publish_image(self):
        ret, frame = self.cap.read()
        if not ret:
            rospy.logerr("Ошибка при чтении видео")
            return
        
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')
            self.image_pub.publish(compressed_msg)
        except Exception as e:
            rospy.logerr(f"Ошибка при публикации изображения: {str(e)}")
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        
if __name__ == "__main__":
    try:
        image_publisher = ImagePublisher()
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            image_publisher.publish_image()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
