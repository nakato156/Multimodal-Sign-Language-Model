import numpy as np

from video_reader import PyVideoReader
from dwpose import DwposeDetector
from PIL import Image
import pickle

DEBUG = False

class KeypointProcessing:
    def __init__(self):
        self.model = DwposeDetector.from_pretrained_default()

    def process_keypoints(self, videoPath):
        print("Video: ", videoPath)
        
        keypoints = []
        video_reader = PyVideoReader(videoPath, device='cuda')
        video = video_reader.decode()

        num_frames = len(video)

        for i in range(num_frames):
            frame_image = Image.fromarray(video[i])

            imageDeb, keypoint_dict, _ = self.model(frame_image, include_hand=True, include_face=True, include_body=True, image_and_json=True)
            keypoints.append(keypoint_dict)

            imageDeb.save("deb.jpg")

        print(keypoints)
        
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
            pose = np.array(frame['pose_keypoints_2d']).reshape(-1, 3)
            hand_l = np.array(frame['hand_left_keypoints_2d']).reshape(-1, 3)
            hand_r = np.array(frame['hand_right_keypoints_2d']).reshape(-1, 3)
            face = np.array(frame['face_keypoints_2d']).reshape(-1, 3)

            full_frame = np.concatenate([pose, face, hand_l, hand_r], axis=0)  
            all_frames.append(full_frame)

        return np.stack(all_frames)

    def extract_coords(self, keypoint, index):
        x, y, c = keypoint[index * 3:index*3+3]
        return np.array([x, y]), c

    def compute_relative_position(self, pose_keypoints, left_hand_keypoints, right_hand_keypoints):
        l_shoulder, c1 = self.extract_coords(pose_keypoints, 2) 
        r_shoulder, c2 = self.extract_coords(pose_keypoints, 5)

        if c1 < 0.5 or c2 < 0.5:
            return None, 0, None, 0
        
        center = (l_shoulder + r_shoulder)/2

        l_hand, lc = self.extract_coords(left_hand_keypoints, 0)
        r_hand, rc = self.extract_coords(right_hand_keypoints, 0)

        return l_hand - center, lc, r_hand - center, rc

    def compute_person_movement(self, frames):
        person_trackers = {}

        for frame_idx, frame in enumerate(frames):
            for person_idx, person in enumerate(frame['people']):
                key = f"{person_idx}"
                
                pose = person.get('pose_keypoints_2d', [])
                face = person.get('face_keypoints_2d', [])
                hand_l = person.get('hand_left_keypoints_2d', [])
                hand_r = person.get('hand_right_keypoints_2d', [])

                if not pose or not hand_l or not hand_r:
                    continue

                lh_rel, lc, rh_rel, rc = self.compute_relative_position(pose, hand_l, hand_r)

                if lc < 0.5 or rc < 0.5:
                    continue

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