"""
link of code:
https://github.com/spmallick/learnopencv/blob/master/PyTorch-Keypoint-RCNN/run_pose_estimation.py
"""

import cv2
import numpy as np

# import config.config_ as config

# create the list of keypoints.
keypoints = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def get_limbs_from_keypoints(keypoints):
    limbs = [
        [keypoints.index("right_eye"), keypoints.index("nose")],
        [keypoints.index("right_eye"), keypoints.index("right_ear")],
        [keypoints.index("left_eye"), keypoints.index("nose")],
        [keypoints.index("left_eye"), keypoints.index("left_ear")],
        [keypoints.index("right_shoulder"), keypoints.index("right_elbow")],
        [keypoints.index("right_elbow"), keypoints.index("right_wrist")],
        [keypoints.index("left_shoulder"), keypoints.index("left_elbow")],
        [keypoints.index("left_elbow"), keypoints.index("left_wrist")],
        [keypoints.index("right_hip"), keypoints.index("right_knee")],
        [keypoints.index("right_knee"), keypoints.index("right_ankle")],
        [keypoints.index("left_hip"), keypoints.index("left_knee")],
        [keypoints.index("left_knee"), keypoints.index("left_ankle")],
        [keypoints.index("right_shoulder"), keypoints.index("left_shoulder")],
        [keypoints.index("right_hip"), keypoints.index("left_hip")],
        [keypoints.index("right_shoulder"), keypoints.index("right_hip")],
        [keypoints.index("left_shoulder"), keypoints.index("left_hip")],
    ]
    return limbs


def filter_persons(scores):
    person_indx = np.argmax(scores)
    return person_indx


def draw_skeleton_per_person(
    img,
    persons_scores,
    all_keypoints,
    keypoints_scores,
    keypoint_threshold=2,
    conf_threshold=0.9,
):
    WHITE_COLOR = (255, 255, 255)
    GREEN_COLOR = (0, 255, 0)
    limbs = get_limbs_from_keypoints(keypoints)
    # create a copy of the image
    img_copy = img.copy()
    # thickness
    tkn1 = 2
    tkn2 = 4
    scores = persons_scores.cpu().detach().numpy()
    if len(scores) > 0:
        person_id = filter_persons(scores)
        # check if the keypoints are detected
        if len(all_keypoints) > 0:
            # iterate for every person detected
            # check the confidence score of the detected person
            if scores[person_id] > conf_threshold:
                # grab the keypoint-locations for the detected person
                keypoints = all_keypoints[person_id, ...].cpu()
                # iterate for every limb
                for limb_id in range(len(limbs)):
                    # pick the start-point of the limb
                    limb_loc1 = (
                        keypoints[limbs[limb_id][0], :2]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int32)
                    )
                    # pick the start-point of the limb
                    limb_loc2 = (
                        keypoints[limbs[limb_id][1], :2]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int32)
                    )
                    # consider limb-confidence score as the minimum keypoint score among the two keypoint scores
                    limb_score = min(
                        keypoints_scores[person_id, limbs[limb_id][0]],
                        keypoints_scores[person_id, limbs[limb_id][1]],
                    )
                    # check if limb-score is greater than threshold
                    if limb_score > keypoint_threshold:
                        # draw the line for the limb
                        cv2.line(
                            img_copy,
                            tuple(limb_loc1),
                            tuple(limb_loc2),
                            GREEN_COLOR,
                            tkn1,
                        )

                scores = keypoints_scores[person_id, ...].cpu().detach().numpy()
                # iterate for every keypoint-score
                for kp in range(len(scores)):
                    # check the confidence score of detected keypoint
                    if scores[kp] >= keypoint_threshold:
                        # convert the keypoint float-array to a python-list of intergers
                        keypoint = tuple(
                            map(int, keypoints[kp, :2].detach().numpy().tolist())
                        )
                        # draw a cirle over the keypoint location
                        cv2.circle(img_copy, keypoint, tkn2, WHITE_COLOR, -1)

    return img_copy


def draw_tracking_skeleton_per_person(img, all_keypoints):
    WHITE_COLOR = (255, 255, 255)
    GREEN_COLOR = (0, 255, 0)
    limbs = get_limbs_from_keypoints(keypoints)
    # create a copy of the image
    img_copy = img.copy()
    # thickness
    tkn1 = 2
    tkn2 = 4
    if len(all_keypoints) > 0:
        # grab the keypoint-locations for the detected person
        keypoints = all_keypoints[0].cpu()
        # iterate for every limb
        for limb_id in range(len(limbs)):
            # pick the start-point of the limb
            limb_loc1 = (
                keypoints[limbs[limb_id][0], :2].detach().cpu().numpy().astype(np.int32)
            )
            # pick the start-point of the limb
            limb_loc2 = (
                keypoints[limbs[limb_id][1], :2].detach().cpu().numpy().astype(np.int32)
            )
            #   # draw the line for the limb
            cv2.line(img_copy, tuple(limb_loc1), tuple(limb_loc2), GREEN_COLOR, tkn1)

        # iterate for every keypoint-score
        for kp in range(len(keypoints)):
            keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
            # draw a cirle over the keypoint location
            cv2.circle(img_copy, keypoint, tkn2, WHITE_COLOR, -1)

    return img_copy
