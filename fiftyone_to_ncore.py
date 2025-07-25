# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import logging

from pathlib import Path
from scipy.linalg import expm
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import tqdm
import open3d as o3d

from ncore_waymo.impl.data_converter.data_converter import DataConverter

# type: ignore

from ncore.impl.data.data3 import ContainerDataWriter, JsonLike
from ncore.impl.data.types import (
    Poses,
    OpenCVPinholeCameraModelParameters,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
    TrackLabel,
    FrameLabel3,
    BBox3,
    LabelSource,
    DynamicFlagState,
    Tracks,
)
from ncore.impl.common.common import PoseInterpolator
from ncore.impl.common.transformations import (
    transform_point_cloud,
    se3_inverse,
    transform_bbox,
    is_within_3d_bboxes,
)
import fiftyone as fo
from fiftyone import ViewField as F

CAMERA_MAP = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

LIDAR_MAP = ["LIDAR_TOP"]


# This is the “NuScenes → NCore” axis‐mapping from your notes:
#   R_nuscenes_to_ncore = np.array([
#       [ 0, -1,  0],
#       [ 0,  0, -1],
#       [ 1,  0,  0]
#   ], dtype=np.float32)
#
# In other words:  a point (x_nuscenes, y_nuscenes, z_nuscenes) in NuScenes frame
# can be converted into NCore frame by R @ [x_n,y_n,z].  We apply this to the 3×3
# rotation submatrix after we’ve built it from the NuScenes quaternion.
R_nuscenes_to_ncore_lidar = np.array(
    [
        [1, 0, 0],  # X_nu (right) → X_nc (right)
        [0, 0, -1],  # Y_nu (forward) → Z_nc (forward)
        [0, 1, 0],  # Z_nu (up) → -Y_nc (down)
    ],
    dtype=np.float32,
)

R_nuscenes_to_ncore_camera = np.array(
    [
        [0, -1, 0],  # left → -X (right)
        [0, 0, -1],  # up   → -Y (down)
        [1, 0, 0],  # forward → Z (principal)
    ],
    dtype=np.float32,
)


def is_orthonormal(R: np.ndarray, atol: float = 1e-6) -> bool:
    """
    Check that R is (nearly) orthonormal:  R @ R^T = I, det(R) ≈ +1.
    """
    should_be_I = R @ R.T
    I = np.eye(3, dtype=R.dtype)
    cond1 = np.allclose(should_be_I, I, atol=atol)
    cond2 = np.isclose(np.linalg.det(R), +1.0, atol=atol)
    return cond1 and cond2


def extract_basis(R: np.ndarray):
    """
    Given a 3×3 rotation matrix R, return three unit‐vectors:
      - forward := R[:, 0]
      - right   := R[:, 1]
      - up      := R[:, 2]
    (assuming your convention is that the first column is “forward,” second is “right,” third is “up.”)
    """
    assert R.shape == (3, 3)
    fwd = R[:, 0]
    right = R[:, 1]
    up = R[:, 2]
    return fwd, right, up


def test_given_quat_from_user_example(R_ncore):
    """
    Now re‐run the same pipeline on the translation+quaternion the user provided,
    then print out (and assert) the three basis vectors are orthonormal.
    We compare them against a “manual” reconstruction of R from pyquaternion + axis‐flip.
    """

    R_ncore = R_ncore[:3, :3]
    # Extract basis‐vectors
    fwd, right, up = extract_basis(R_ncore)

    # 1) Check that R_ncore is orthonormal:
    assert is_orthonormal(R_ncore), (
        "After applying R_nuscenes_to_ncore, the 3×3 should be orthonormal. " f"But R_ncore =\n{R_ncore}"
    )

    # 2) Check that each basis vector is unit‐length and mutually orthogonal:
    np.testing.assert_allclose(np.linalg.norm(fwd), 1.0, atol=1e-5, err_msg="‖forward‖ ≠ 1")
    np.testing.assert_allclose(np.linalg.norm(right), 1.0, atol=1e-5, err_msg="‖right‖ ≠ 1")
    np.testing.assert_allclose(np.linalg.norm(up), 1.0, atol=1e-5, err_msg="‖up‖ ≠ 1")

    np.testing.assert_allclose(np.dot(fwd, right), 0.0, atol=1e-6, err_msg="forward⋅right ≠ 0")
    np.testing.assert_allclose(np.dot(fwd, up), 0.0, atol=1e-6, err_msg="forward⋅up ≠ 0")
    np.testing.assert_allclose(np.dot(right, up), 0.0, atol=1e-6, err_msg="right⋅up ≠ 0")


def extrapolate_pose_based_on_velocity(
    T_SDC_global: np.ndarray, v_global: np.ndarray, w_global: np.ndarray, dt_sec: float
) -> np.ndarray:
    T_extrapolated = np.eye(4, dtype=np.float32)
    T_extrapolated[:3, :3] = expm(get_skew_symmetric(w_global) * dt_sec) @ T_SDC_global[:3, :3]
    T_extrapolated[:3, 3:4] = T_SDC_global[:3, 3:4] + v_global * dt_sec

    return T_extrapolated


def get_skew_symmetric(vec: np.ndarray) -> np.ndarray:
    skew_sym = np.zeros((3, 3), dtype=np.float32)

    skew_sym[0, 1] = -vec[2]
    skew_sym[0, 2] = vec[1]
    skew_sym[1, 0] = vec[2]
    skew_sym[1, 2] = -vec[0]
    skew_sym[2, 0] = -vec[1]
    skew_sym[2, 1] = vec[0]

    return skew_sym


class NuscenesConverter(DataConverter):
    """
    Dataset preprocessing class, which preprocess waymo-open dataset to a canonical data representation as used within the Nvidia NRECore-SDK project.
    Waymo-open data can be downloaded from https://waymo.com/intl/en_us/open/download/ in form of tfrecords files. Further details on the dataset are
    available in the original publication https://arxiv.org/abs/1912.04838 or the githbub repository https://github.com/waymo-research/waymo-open-dataset
    """

    FINE_LIDAR_TIMESTAMPS_MINIMUM_SENSOR_MOTION_THRESHOLD_M = (
        0.1  # threshold on the per spin linear motion of the lidar sensor to allow accurate timestamp inference
    )

    def __init__(self, config):
        super().__init__(config)

        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_sequence_paths(config) -> list[Path]:

        dataset = fo.load_dataset("nuscenes-rerun-fo")
        sequences = []
        tokens = list(set(dataset.values("scene_token")))
        for token in tokens:
            sequences.append(dataset.match(F("scene_token") == token))
        return sequences

    @staticmethod
    def from_config(config) -> DataConverter:
        return NuscenesConverter(config)

    def convert_sequence(self, sequence) -> None:
        """
        Runs dataset-specific conversion for a sequence
        """
        self.logger.info(sequence.first().scene_token)

        # Check that all frames in the dataset have the same sequence name (i.e. belong to the same sequence)
        # and deserialize into memory
        frames = []
        sequence_name = sequence.first().scene_token
        for sample in sequence:
            group_id = sample.group.id

            group = sequence.get_group(group_id)
            frames.append(group)

        # DataWriter for all outputs
        self.data_writer = ContainerDataWriter(
            self.output_dir / sequence_name,
            sequence_name,
            self.get_active_camera_ids(CAMERA_MAP),
            self.get_active_lidar_ids(LIDAR_MAP),
            self.get_active_radar_ids([]),
            "waymo-calibration",
            "waymo-egomotion",
            sequence_name,
            {},  # no generic sequence meta data
            # single shard
            0,
            1,
            False,
        )

        # Decode poses
        self.decode_poses(frames)

        # Decode lidar frames
        self.decode_lidars(frames)

        # Decode camera frames
        self.decode_cameras(frames)

        # Store per-shard meta data / final success state / close file
        self.data_writer.finalize()

    def decode_poses(self, frames) -> None:
        # Grab poses from our groups lidar sample
        T_rig_worlds_array = []
        T_rig_world_timestamps_us_array = []

        # rewrite to only have lidar timestamps
        for i, frame in enumerate(frames):  # frame has all sensors 6 cams 1 lidar

            nuscenes_T_rig_world = np.array(frame["3D"]["T_rig_world"])

            # Convert T_rig_world to point correct way
            # (a) Extract rotation + translation from NuScenes:
            R_from_quat = nuscenes_T_rig_world[:3, :3]  # 3×3 rotation in NuScenes
            t_nuscenes = nuscenes_T_rig_world[:3, 3]  # 3×1 translation in NuScenes

            # (b) Convert both R and t into NCore coordinates:
            R_ncore = R_nuscenes_to_ncore_lidar @ R_from_quat
            t_ncore = R_nuscenes_to_ncore_lidar @ t_nuscenes

            # (c) Build a new 4×4 'T_fixed' in NCore conventions:
            T_fixed = np.eye(4, dtype=np.float32)
            T_fixed[:3, :3] = R_ncore
            T_fixed[:3, 3] = t_ncore

            # (d) Append T_fixed to your array of poses (instead of copying t_nuscenes directly):
            test_given_quat_from_user_example(T_fixed)
            T_rig_worlds_array.append(T_fixed)
            T_rig_world_timestamps_us_array.append(frame["3D"]["rig_timestamp"])

        # make unique + sort + stack all poses (common canonical format convention)
        T_rig_world_timestamps_us, unique_indices = np.unique(
            np.array(T_rig_world_timestamps_us_array, dtype=np.uint64),
            return_index=True,
        )
        T_rig_worlds = np.stack(T_rig_worlds_array)[unique_indices].astype(np.float64)

        # adjust for skipping a frame
        T_rig_worlds = T_rig_worlds[:-1]

        T_rig_world_timestamps_us = T_rig_world_timestamps_us[:-1]

        T_rig_world_base = T_rig_worlds[0]

        camera_ts = [frame[c]["timestamp"] for frame in frames for c in CAMERA_MAP]
        lidar_ts = [frame["3D"]["rig_timestamp"] for frame in frames]
        min_sensor_ts = min(camera_ts + lidar_ts)
        max_sensor_ts = max(camera_ts + lidar_ts)

        first_pose_ts = T_rig_world_timestamps_us[0]
        last_pose_ts = T_rig_world_timestamps_us[-1]

        T_rig_worlds = np.linalg.inv(T_rig_world_base) @ T_rig_worlds
        T_rig_world_base = T_rig_worlds[0]

        self.poses = Poses(
            T_rig_world_base=T_rig_world_base.astype(np.float64),
            T_rig_worlds=T_rig_worlds.astype(np.float64),
            T_rig_world_timestamps_us=T_rig_world_timestamps_us,
        )

        # === Padding Start (only if needed) ===
        if first_pose_ts > min_sensor_ts:
            print(f"[Replacing] First pose timestamp with {min_sensor_ts}")
            self.poses.T_rig_world_timestamps_us[0] = min_sensor_ts

        # === Replace final timestamp with max_sensor_ts (only if it's smaller) ===
        if last_pose_ts < max_sensor_ts:
            print(f"[Replacing] Last pose ts with {max_sensor_ts}")
            self.poses.T_rig_world_timestamps_us[-1] = max_sensor_ts

        # Rebuild interpolator
        self.pose_interpolator = PoseInterpolator(self.poses.T_rig_worlds, self.poses.T_rig_world_timestamps_us)

        print(f"POSES LENGTH {len(self.poses.T_rig_world_timestamps_us)}")
        assert self.poses.T_rig_world_timestamps_us[0] <= min_sensor_ts
        assert self.poses.T_rig_world_timestamps_us[-1] >= max_sensor_ts
        assert len(self.poses.T_rig_world_timestamps_us) == len(frames) - 1

        # Save the poses
        self.data_writer.store_poses(self.poses)

        # Log base pose to share it more easily with downstream teams (it is serialized also explicitly)
        with np.printoptions(floatmode="unique", linewidth=200):  # print in highest precision
            self.logger.info(f"> processed {len(T_rig_worlds)} poses, using base pose:\n{T_rig_world_base}")

    # Unconditionally dynamic / static label types
    LABEL_STRINGS_UNCONDITIONALLY_DYNAMIC: set[str] = set(
        [
            "pedestrian",
            "cyclist",
        ]
    )
    LABEL_STRINGS_UNCONDITIONALLY_STATIC: set[str] = set(["sign"])

    # Velocity threshold to classify moving objects as dynamic
    GLOBAL_SPEED_DYNAMIC_THRESHOLD = 1.0 / 3.6

    # Dynamic flag from label bbox padding
    LIDAR_DYNAMIC_FLAG_BBOX_PADDING_METERS = 0.5

    def decode_lidars(self, frames) -> None:
        """
        Converts the raw point cloud data into 3D depth rays in space also compensating for the
        motion of the ego-car (lidar unwinding)
        """

        ## Collect calibrations

        ## Collect frame start timestamps
        raw_frame_start_timestamps_us = [frame["3D"].rig_timestamp for frame in frames]

        ## Parse frame-associated labels in rig space (will be transformed to sensor frames below)

        lidar = frames[0]["3D"]

        # Determine sensor extrinsics
        T_sensor_rig_nu = np.array(lidar["T_sensor_rig"]).reshape(4, 4)

        R_nu = T_sensor_rig_nu[:3, :3]
        t_nu = T_sensor_rig_nu[:3, 3]

        R_nc = R_nuscenes_to_ncore_lidar @ R_nu
        t_nc = R_nuscenes_to_ncore_lidar @ t_nu

        T_sensor_rig = np.eye(4, dtype=np.float32)
        T_sensor_rig[:3, :3] = R_nc
        T_sensor_rig[:3, 3] = t_nc

        # Variables associated with intrinsics
        lidar_model_parameters: Optional[RowOffsetStructuredSpinningLidarModelParameters] = None

        for i, frame in tqdm.tqdm(enumerate(frames), desc=f"Process LIDAR_TOP", total=len(frames)):
            # Get frame timestamps
            if i < len(frames) - 1:
                lidar = frame["3D"]
                frame_start_timestamp_us = raw_frame_start_timestamps_us[i]

                frame_end_timestamp_us = raw_frame_start_timestamps_us[i + 1]

                timestamps_us = np.array([frame_start_timestamp_us, frame_end_timestamp_us], dtype=np.uint64)

                scene = fo.Scene.from_fo3d(lidar.filepath)
                for obj in scene.traverse():
                    if obj.name == "LIDAR_TOP":
                        pcd = obj

                pcd_loaded = np.array(o3d.io.read_point_cloud(pcd.pcd_path).points).astype(np.float32)

                T_rig_worlds = self.pose_interpolator.interpolate_to_timestamps(timestamps_us)

                # Serialize lidar frame
                self.data_writer.store_lidar_frame(
                    lidar_id="LIDAR_TOP",
                    continuous_frame_index=i,
                    xyz_s=pcd_loaded,
                    xyz_e=pcd_loaded,
                    intensity=np.zeros(pcd_loaded.shape[0]).astype(np.float32),
                    timestamp_us=np.full((pcd_loaded.shape[0],), lidar.rig_timestamp).astype(np.uint64),
                    frame_labels=[],
                    model_element=None,
                    T_rig_worlds=T_rig_worlds,
                    timestamps_us=timestamps_us,
                    generic_data={},
                    generic_meta_data={},
                )
        print(f"LIDAR frame {i}")
        ## Store all static sensor data
        self.data_writer.store_lidar_meta(
            lidar_id="LIDAR_TOP",
            frame_timestamps_us=np.array(raw_frame_start_timestamps_us[:-1], dtype=np.uint64),
            T_sensor_rig=T_sensor_rig.astype(np.float32),
            lidar_model_parameters=lidar_model_parameters,
            generic_meta_data={},
        )

    def decode_cameras(self, frames) -> None:
        """
        Extracts the images and camera metadata for all cameras within a single frame. Camera metadata must hold
        the information used to compensate for rolling shutter effect and to convert RGB images to 3D RGB rays in space
        """

        calibrations = frames[0]

        for camera_id in CAMERA_MAP:

            T_sensor_rig = np.array(frames[0][camera_id]["T_sensor_rig"]).reshape(4, 4)
            """T_sensor_rig_nu = np.array(frames[0][camera_id]["T_sensor_rig"], dtype=np.float32).reshape(4, 4)
            R_cam_rig_nu = T_sensor_rig_nu[:3, :3]
            

            # (b) Convert to NCore camera conventions:
            R_cam_rig_nc = R_nuscenes_to_ncore_camera @ R_cam_rig_nu

            # (c) Build a new 4×4 T_sensor_rig in NCore frame:
            T_sensor_rig = np.eye(4, dtype=np.float32)
            T_sensor_rig[:3, :3] = R_cam_rig_nc
            t_cam_rig_nu = T_sensor_rig_nu[:3, 3]
            t_cam_rig_nc = R_nuscenes_to_ncore_camera @ t_cam_rig_nu

            T_sensor_rig[:3, 3] = t_cam_rig_nc"""

            frame_end_timestamps_us = []
            continuous_frame_index = 0
            for i, frame in tqdm.tqdm(enumerate(frames), desc=f"Process {camera_id}", total=len(frames)):
                # Get frame timestamps

                if i < len(frames) - 1:
                    ## Load current camera's image
                    image = frame[camera_id]

                    ## Get frame timestamps
                    frame_start_timestamp_us = int(image.timestamp)
                    frame_end_timestamp_us = int(image.timestamp + 20000)

                    ## Collect timestamps poses
                    timestamps_us = np.array([frame_start_timestamp_us, frame_end_timestamp_us], dtype=np.uint64)

                    # Extrapolate the pose to the start and end timestamp of the image frame considering the (angular) velocity at the time of the acquisition
                    T_rig_worlds = self.pose_interpolator.interpolate_to_timestamps(timestamps_us)

                    frame_end_timestamps_us.append(frame_end_timestamp_us)
                    with open(image.filepath, "rb") as f:
                        image_bin = f.read()

                    # Store the image and its metadata

                    self.data_writer.store_camera_frame(
                        camera_id, i, image_bin, "jpeg", T_rig_worlds, timestamps_us, {}, {}
                    )

            # Extract intrinsic data
            width = image.metadata.width
            height = image.metadata.height
            intrinsic = image.intrinsics
            f_u = intrinsic[0, 0]
            f_v = intrinsic[1, 1]
            c_u = intrinsic[0, 2]
            c_v = intrinsic[1, 2]
            k1, k2, p1, p2, k3 = 0, 0, 0, 0, 0

            rolling_shutter_direction = ShutterType.GLOBAL

            self.data_writer.store_camera_meta(
                camera_id,
                np.array(frame_end_timestamps_us, dtype=np.uint64),
                T_sensor_rig.astype(np.float32),
                OpenCVPinholeCameraModelParameters(
                    resolution=np.array([width, height], dtype=np.uint64),
                    shutter_type=rolling_shutter_direction,
                    external_distortion_parameters=None,
                    principal_point=np.array([c_u, c_v], dtype=np.float32),
                    focal_length=np.array([f_u, f_v], dtype=np.float32),
                    radial_coeffs=np.array([k1, k2, k3, 0, 0, 0], dtype=np.float32),
                    tangential_coeffs=np.array([p1, p2], dtype=np.float32),
                    thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32),
                ),
                None,
                {},
            )
            print(f"Cam {camera_id} frame {i}")
