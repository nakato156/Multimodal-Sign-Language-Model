import numpy as np

from video_reader import PyVideoReader
from rtmlib import RTMPose, RTMDet, draw_skeleton, Custom
from PIL import Image
import pickle
import cv2
import numpy as np

from IPython.display import display

DEBUG = False

class KeypointProcessing:
    def __init__(self):
        backend = 'onnxruntime'
        device = 'cuda'
        self.openpose= True

#        self.model_pose = RTMPose(onnx_model='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip',
#                            backend=backend,
#                            device=device)
#
#        self.model_det = RTMDet(onnx_model='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
#                            backend=backend,
#                            device=device)

        self.model = Custom(
            to_openpose=self.openpose,
            det_class='RTMDet',
            det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
            det_input_size=(640, 640),
            pose_class='RTMPose',
            pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip',
            pose_input_size=(288, 384),
            backend=backend,
            device=device,
        )

    def process_keypoints(self, videoPath):
        print("Video: ", videoPath)
        
        keypoints = []
        try:
            video_reader = PyVideoReader(videoPath, device='cuda')
        except Exception as e:
            print(f"CUDA failed: {e}\nFalling back to CPU.")
            video_reader = PyVideoReader(videoPath, device='cpu')

        video = video_reader.decode()

        num_frames = len(video)
        print(num_frames)
        
        for i in range(num_frames):
#            bboxes = self.model_det(video[i])
#            keypoint_dict, scores= self.model_pose(video[i], bboxes)

            keypoint_dict, scores= self.model(video[i])

            keypoints.append(keypoint_dict)

            #img_show = np.array(video[i], dtype=np.uint8)
            #img_show = draw_skeleton(img_show, keypoint_dict, scores, kpt_thr=0.5, openpose_skeleton=self.openpose)

            #print(f"Keypoints{[i]}", keypoint_dict)
            #print(f"Score{[i]}", scores)

            #cv2.imwrite(f"./frames/deb{i}.jpg", img_show)

        #print(keypoints)
        
        if DEBUG: 
            print("Saving keypoints")
            with open("keypointsTest.pkl", "wb") as kp:
                pickle.dump(keypoints, kp)
            print("Keypoints saved")
        
        result = self.compute_person_movement(keypoints)
        signer_id = max(result.items(), key=lambda x: x[1]['total'])[0]
        print("Likely signer:", signer_id, "with total hand movement:", result[signer_id]['total'])
        
        keypoints_array = self.process_frames_to_array(result[signer_id]['frames'])

        return keypoints_array

    def process_frames_to_array(self, signer_frames):
        all_frames = []

        for frame in signer_frames:
            pose = np.array(frame['pose_keypoints_2d']).reshape(-1, 2)
            hand_l = np.array(frame['hand_left_keypoints_2d']).reshape(-1, 2)
            hand_r = np.array(frame['hand_right_keypoints_2d']).reshape(-1, 2)
            face = np.array(frame['face_keypoints_2d']).reshape(-1, 2)

            full_frame = np.concatenate([pose, face, hand_l, hand_r], axis=0)  
            all_frames.append(full_frame)

        return np.stack(all_frames)

    def extract_coords(self, keypoint, index):
        x, y = keypoint[index * 2:index*2+2]
        return np.array([x, y])

    def compute_relative_position(self, pose_keypoints, left_hand_keypoints, right_hand_keypoints):
        l_shoulder = self.extract_coords(pose_keypoints, 2) 
        r_shoulder = self.extract_coords(pose_keypoints, 5)
        
        center = (l_shoulder + r_shoulder)/2

        l_hand = self.extract_coords(left_hand_keypoints, 0)
        r_hand = self.extract_coords(right_hand_keypoints, 0)

        return l_hand - center, r_hand - center

    def compute_person_movement(self, frames):
        person_trackers = {}

        for frame_idx, frame in enumerate(frames):
            for person_idx, person in enumerate(frame):
                key = f"{person_idx}"

                pose = person[0:25,:]
                face = person[25:95,:]
                hand_l = person[95:116,:]
                hand_r = person[0:,:]

                lh_rel, rh_rel = self.compute_relative_position(pose, hand_l, hand_r)

                tracker = person_trackers.setdefault(key, {
                    'prev_lh': None,
                    'prev_rh': None,
                    'total': 0.0,
                    'frames': []
                })

                if tracker['prev_lh'] is not None:
                    tracker['total'] += np.linalg.norm(lh_rel - tracker['prev_lh'])
                if tracker['prev_rh'] is not None:
                    tracker['total'] += np.linalg.norm(rh_rel - tracker['prev_rh'])

                tracker['prev_lh'] = lh_rel
                tracker['prev_rh'] = rh_rel
                tracker['frames'].append({
                    'pose_keypoints_2d': pose,
                    'face_keypoints_2d': face,
                    'hand_left_keypoints_2d': hand_l,
                    'hand_right_keypoints_2d': hand_r
                })

        return person_trackers