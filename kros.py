#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from mrcnn.config import Config
from mrcnn import model as modellib

class CustomConfig(Config):
    NAME = "custom"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # Background + 사용자 정의 클래스 수
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3

class CocoConfig(Config):
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  # Background + COCO 데이터셋 클래스 수

class ObjectDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

        self.pose_pub_bottle = rospy.Publisher("/detected_objects_pose/bottle", PoseArray, queue_size=10)
        self.pose_pub_cup = rospy.Publisher("/detected_objects_pose/cup", PoseArray, queue_size=10)
        self.pose_pub_stop_sign = rospy.Publisher("/detected_objects_pose/stop_sign", PoseArray, queue_size=10)

        self.color_image = None
        self.depth_image = None
        self.camera_info = None

        self.custom_config = CustomConfig()
        self.custom_model = modellib.MaskRCNN(mode="inference", config=self.custom_config, model_dir='logs')
        self.custom_model.load_weights('/home/kimm/hong_ws/src/hong/scripts/mask_rcnn_object_0060.h5', by_name=True)

        self.coco_config = CocoConfig()
        self.coco_model = modellib.MaskRCNN(mode="inference", config=self.coco_config, model_dir='logs')
        self.coco_model.load_weights('/home/kimm/hong_ws/src/hong/scripts/mask_rcnn_coco.h5', by_name=True)

    def image_callback(self, data):
        self.color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def depth_callback(self, data):
        self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")

    def camera_info_callback(self, data):
        self.camera_info = data

    def process_frame(self):
        if self.color_image is not None and self.depth_image is not None and self.camera_info is not None:
            custom_results = self.custom_model.detect([self.color_image], verbose=0)
            custom_r = custom_results[0]

            coco_results = self.coco_model.detect([self.color_image], verbose=0)
            coco_r = coco_results[0]

            poses_bottle = PoseArray()
            poses_cup = PoseArray()
            poses_stop_sign = PoseArray()

            for i, class_id in enumerate(custom_r['class_ids']):
                pose = self.calculate_pose(custom_r, i, class_id)
                if pose:
                    if class_id == 1:
                        poses_bottle.poses.append(pose)
                    elif class_id == 2:
                        poses_cup.poses.append(pose)

            for i, class_id in enumerate(coco_r['class_ids']):
                pose = self.calculate_pose(coco_r, i, class_id)
                if pose and class_id == 12:
                    poses_stop_sign.poses.append(pose)

            self.pose_pub_bottle.publish(poses_bottle)
            self.pose_pub_cup.publish(poses_cup)
            self.pose_pub_stop_sign.publish(poses_stop_sign)
            self.visualize_results(self.color_image, np.concatenate((custom_r['rois'], coco_r['rois']), axis=0),
                                   np.concatenate((custom_r['masks'], coco_r['masks']), axis=2) if custom_r['masks'].size > 0 and coco_r['masks'].size > 0 else np.array([]),
                                   np.concatenate((custom_r['class_ids'], coco_r['class_ids']), axis=0),
                                   np.concatenate((custom_r['scores'], coco_r['scores']), axis=0))

        

    def calculate_pose(self, results, index, class_id):
        y1, x1, y2, x2 = results['rois'][index]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        depth = self.depth_image[int(center_y), int(center_x)]
        if depth == 0:
            return None
        depth_meters = depth / 1000.0

        real_x = (center_x - self.camera_info.K[2]) * depth_meters / self.camera_info.K[0]
        real_y = (center_y - self.camera_info.K[5]) * depth_meters / self.camera_info.K[4]
        real_z = depth_meters

        pose = Pose()
        pose.position.x = real_x
        pose.position.y = real_y
        pose.position.z = real_z
        pose.orientation.w = 1.0
        return pose

    def visualize_results(self, image, boxes, masks, class_ids, scores):
        N = boxes.shape[0]
        if self.camera_info is None:
            print("No camera info available.")
            return
    
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        for i in range(N):
            class_id = class_ids[i]
            # bottle (1), cup (2), and stop sign (12) classes
            if class_id not in [1, 2, 12]:
                continue

            y1, x1, y2, x2 = boxes[i]
            mask = masks[:, :, i] if masks.size > 0 else np.array([])
            score = scores[i] if scores.size > 0 else 0  # Default score to 0 if not available
            center_x = int((x1 + x2) / 2.0)
            center_y = int((y1 + y2) / 2.0)

            # 깊이 정보를 이용하여 실제 세계 좌표 계산
            depth = self.depth_image[center_y, center_x] / 1000.0  # Assuming depth is in mm
            real_x = (center_x - cx) * depth / fx
            real_y = (center_y - cy) * depth / fy
            real_z = depth

            if mask.size > 0:
                image = self.apply_mask(image, mask, color=np.random.rand(3))

            # 바운딩 박스 계산
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
            # 객체 정보 표시
            label_map = {1: "Bottle", 2: "Cup", 12: "Stop Sign"}
            label = f"{label_map[class_id]}: {score:.2f}, X: {real_x:.2f}, Y: {real_y:.2f}, Z: {real_z:.2f}"
            image = cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 결과 이미지 표시
        cv2.imshow("Detected Objects", image)
        cv2.waitKey(3)



    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image."""
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

if __name__ == '__main__':
    rospy.init_node('object_detector_node', anonymous=True)
    od = ObjectDetector()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        od.process_frame()
        rate.sleep()
