from __future__ import annotations

import argparse
import math
import os
import pathlib
from typing import Any, Literal, Sequence

import fiftyone as fo
import fiftyone.utils.utils3d as fou3d
import matplotlib
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
from pyquaternion import Quaternion
import numpy as np
from fiftyone.utils.rerun import RrdFile
from nuscenes import nuscenes
from nuscenes.lidarseg.lidarseg_utils import paint_points_label
from nuscenes.scripts.export_poses import derive_latlon as derive_latlon_nu
from nuscenes.utils.color_map import get_colormap
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image, view_points
from PIL import Image

"""
# download the NuScenes mini dataset
wget https://www.nuscenes.org/data/v1.0-mini.tgz
wget https://www.nuscenes.org/data/nuScenes-lidarseg-mini-v1.0.tar.bz2

# extract the dataset to a folder named nuscenes-mini
tar -xvf v1.0-mini.tgz
tar -xvf nuScenes-lidarseg-mini-v1.0.tar.bz2

python examples/load-nuscenes.py --nuscenes-data-dir /path/to/nuscenes --rrd --fiftyone

refer to https://github.com/voxel51/fiftyone-rerun-plugin?tab=readme-ov-file for more info
"""

# used to calculate the color for lidar/radar in rerun and radar in FiftyOne
cmap = matplotlib.colormaps["turbo_r"]
norm = matplotlib.colors.Normalize(
    vmin=3.0,
    vmax=75.0,
)

# --- RERUN ---

EARTH_RADIUS_METERS = 6.378137e6
REFERENCE_COORDINATES = {
    "boston-seaport": [42.336849169438615, -71.05785369873047],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
}


def transform_matrix(
    translation: np.ndarray, quaternion_xyzw: np.ndarray, inverse: bool = False
) -> np.ndarray:
    """
    Build a 4×4 homogeneous transform from a NuScenes‐style translation + (x,y,z,w) quaternion.
    If inverse=True, return its inverse.
    """
    if inverse:
        T = np.eye(4, dtype=np.float32)
        # translation should be length‐3
        T[:3, 3] = -translation.astype(np.float32)
        # build 3×3 rotation from quaternion
        R3 = Quaternion(quaternion_xyzw).rotation_matrix.T.astype(np.float32)
        T[:3, :3] = R3
    else:
        T = np.eye(4, dtype=np.float32)
        # translation should be length‐3
        T[:3, 3] = translation.astype(np.float32)
        # build 3×3 rotation from quaternion
        R3 = Quaternion(quaternion_xyzw).rotation_matrix.astype(np.float32)
        T[:3, :3] = R3
    return T




def get_coordinate(
    ref_lat: float, ref_lon: float, bearing: float, dist: float
) -> tuple[float, float]:
    """
    Using a reference coordinate, extract the coordinates of another point in space given its distance and bearing
    to the reference coordinate. For reference, please see: https://www.movable-type.co.uk/scripts/latlong.html.

    Parameters
    ----------
    ref_lat : float
        Latitude of the reference coordinate in degrees, e.g., 42.3368.
    ref_lon : float
        Longitude of the reference coordinate in degrees, e.g., 71.0578.
    bearing : float
        The clockwise angle in radians between the target point, reference point, and the axis pointing north.
    dist : float
        The distance in meters from the reference point to the target point.

    Returns
    -------
    tuple[float, float]
        A tuple of latitude and longitude.

    """  # noqa: D205
    lat, lon = math.radians(ref_lat), math.radians(ref_lon)
    angular_distance = dist / EARTH_RADIUS_METERS

    target_lat = math.asin(
        math.sin(lat) * math.cos(angular_distance)
        + math.cos(lat) * math.sin(angular_distance) * math.cos(bearing)
    )
    target_lon = lon + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat),
        math.cos(angular_distance) - math.sin(lat) * math.sin(target_lat),
    )
    return math.degrees(target_lat), math.degrees(target_lon)


def derive_latlon(
    location: str, pose: dict[str, Sequence[float]]
) -> tuple[float, float]:
    """
    Extract lat/lon coordinate from pose.

    This makes the following two assumptions in order to work:
        1. The reference coordinate for each map is in the south-western corner.
        2. The origin of the global poses is also in the south-western corner (and identical to 1).

    Parameters
    ----------
    location : str
        The name of the map the poses correspond to, i.e., `boston-seaport`.
    pose : dict[str, Sequence[float]]
        nuScenes egopose.

    Returns
    -------
    tuple[float, float]
    Latitude and longitude coordinates in degrees.

    """
    assert (
        location in REFERENCE_COORDINATES.keys()
    ), f"Error: The given location: {location}, has no available reference."

    reference_lat, reference_lon = REFERENCE_COORDINATES[location]
    x, y = pose["translation"][:2]
    bearing = math.atan(x / y)
    distance = math.sqrt(x**2 + y**2)
    lat, lon = get_coordinate(reference_lat, reference_lon, bearing, distance)

    return lat, lon


def log_lidar_and_ego_pose(
    nusc: nuscenes.NuScenes,
    first_lidar_token: str,
    max_timestamp_us: float,
    stream: rr.RecordingStream,
) -> None:
    """Log lidar data and vehicle pose."""
    current_lidar_token = first_lidar_token

    while current_lidar_token != "":
        sample_data = nusc.get("sample_data", current_lidar_token)
        sensor_name = sample_data["channel"]

        if max_timestamp_us < sample_data["timestamp"]:
            break

        # timestamps are in microseconds
        stream.set_time_seconds("timestamp", sample_data["timestamp"] * 1e-6)

        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        rotation_xyzw = np.roll(ego_pose["rotation"], shift=-1)  # go from wxyz to xyzw
        stream.log(
            "world/ego_vehicle",
            rr.Transform3D(
                translation=ego_pose["translation"],
                rotation=rr.Quaternion(xyzw=rotation_xyzw),
                from_parent=False,
            ),
        )
        current_lidar_token = sample_data["next"]

        data_file_path = nusc.dataroot / sample_data["filename"]
        pointcloud = nuscenes.LidarPointCloud.from_file(str(data_file_path))
        points = pointcloud.points[:3].T  # shape after transposing: (num_points, 3)
        point_distances = np.linalg.norm(points, axis=1)
        point_colors = cmap(norm(point_distances))
        stream.log(
            f"world/ego_vehicle/{sensor_name}", rr.Points3D(points, colors=point_colors)
        )


def log_radars(
    nusc: nuscenes.NuScenes,
    first_radar_tokens: list[str],
    max_timestamp_us: float,
    stream: rr.RecordingStream,
) -> None:
    """Log radar data."""
    for first_radar_token in first_radar_tokens:
        current_camera_token = first_radar_token
        while current_camera_token != "":
            sample_data = nusc.get("sample_data", current_camera_token)
            if max_timestamp_us < sample_data["timestamp"]:
                break
            sensor_name = sample_data["channel"]
            rr.set_time_seconds("timestamp", sample_data["timestamp"] * 1e-6)
            data_file_path = nusc.dataroot / sample_data["filename"]
            pointcloud = nuscenes.RadarPointCloud.from_file(str(data_file_path))
            points = pointcloud.points[:3].T  # shape after transposing: (num_points, 3)
            point_distances = np.linalg.norm(points, axis=1)
            point_colors = cmap(norm(point_distances))
            stream.log(
                f"world/ego_vehicle/{sensor_name}",
                rr.Points3D(points, colors=point_colors),
            )
            current_camera_token = sample_data["next"]


def log_annotations(
    nusc: nuscenes.NuScenes,
    location: str,
    first_sample_token: str,
    max_timestamp_us: float,
    stream: rr.RecordingStream,
) -> None:
    rr.log
    """Log 3D bounding boxes."""
    label2id: dict[str, int] = {}
    current_sample_token = first_sample_token
    while current_sample_token != "":
        sample_data = nusc.get("sample", current_sample_token)
        if max_timestamp_us < sample_data["timestamp"]:
            break
        stream.set_time_seconds("timestamp", sample_data["timestamp"] * 1e-6)
        ann_tokens = sample_data["anns"]
        sizes = []
        centers = []
        quaternions = []
        class_ids = []
        lat_lon = []
        for ann_token in ann_tokens:
            ann = nusc.get("sample_annotation", ann_token)

            rotation_xyzw = np.roll(ann["rotation"], shift=-1)  # go from wxyz to xyzw
            width, length, height = ann["size"]
            sizes.append((length, width, height))  # x, y, z sizes
            centers.append(ann["translation"])
            quaternions.append(rr.Quaternion(xyzw=rotation_xyzw))
            if ann["category_name"] not in label2id:
                label2id[ann["category_name"]] = len(label2id)
            class_ids.append(label2id[ann["category_name"]])
            lat_lon.append(derive_latlon(location, ann))

        stream.log(
            "world/anns",
            rr.Boxes3D(
                sizes=sizes,
                centers=centers,
                quaternions=quaternions,
                class_ids=class_ids,
            ),
            rr.GeoPoints(lat_lon=lat_lon),
        )
        current_sample_token = sample_data["next"]

    annotation_context = [(i, label) for label, i in label2id.items()]
    stream.log("world/anns", rr.AnnotationContext(annotation_context), static=True)


def log_front_camera(
    nusc: nuscenes.NuScenes, sample_data: dict[str, Any], stream: rr.RecordingStream
) -> None:
    """Log front pinhole camera with its calibration."""
    calibrated_sensor_token = sample_data["calibrated_sensor_token"]
    calibrated_sensor = nusc.get("calibrated_sensor", calibrated_sensor_token)
    rotation_xyzw = np.roll(
        calibrated_sensor["rotation"], shift=-1
    )  # go from wxyz to xyzw
    stream.log(
        f"world/ego_vehicle/CAM_FRONT",
        rr.Transform3D(
            translation=calibrated_sensor["translation"],
            rotation=rr.Quaternion(xyzw=rotation_xyzw),
            from_parent=False,
        ),
        static=True,
    )
    if len(calibrated_sensor["camera_intrinsic"]) != 0:
        stream.log(
            f"world/ego_vehicle/CAM_FRONT",
            rr.Pinhole(
                image_from_camera=calibrated_sensor["camera_intrinsic"],
                width=sample_data["width"],
                height=sample_data["height"],
            ),
            static=True,
        )


def log_sensor_calibration(
    nusc: nuscenes.NuScenes, sample_data: dict[str, Any], stream: rr.RecordingStream
) -> None:
    """Log sensor calibration (pinhole camera, sensor poses, etc.)."""
    sensor_name = sample_data["channel"]
    calibrated_sensor_token = sample_data["calibrated_sensor_token"]
    calibrated_sensor = nusc.get("calibrated_sensor", calibrated_sensor_token)
    rotation_xyzw = np.roll(
        calibrated_sensor["rotation"], shift=-1
    )  # go from wxyz to xyzw
    stream.log(
        f"world/ego_vehicle/{sensor_name}",
        rr.Transform3D(
            translation=calibrated_sensor["translation"],
            rotation=rr.Quaternion(xyzw=rotation_xyzw),
            from_parent=False,
            axis_length=0.0,
        ),
        static=True,
    )


def log_nuscenes(
    nusc: nuscenes.NuScenes,
    scene_name: str,
    max_time_sec: float,
    stream: rr.RecordingStream,
) -> None:
    """Log nuScenes scene to the given stream."""

    scene = next(s for s in nusc.scene if s["name"] == scene_name)

    location = nusc.get("log", scene["log_token"])["location"]

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get("sample", scene["first_sample_token"])

    first_lidar_token = ""
    first_radar_tokens = []

    for sample_data_token in first_sample["data"].values():
        sample_data = nusc.get("sample_data", sample_data_token)
        log_sensor_calibration(nusc, sample_data, stream)

        if sample_data["sensor_modality"] == "lidar":
            first_lidar_token = sample_data_token
        elif sample_data["sensor_modality"] == "radar":
            first_radar_tokens.append(sample_data_token)
        elif sample_data["channel"] == "CAM_FRONT":
            log_front_camera(nusc, sample_data, stream)

    first_timestamp_us = nusc.get("sample_data", first_lidar_token)["timestamp"]
    max_timestamp_us = first_timestamp_us + 1e6 * max_time_sec

    log_lidar_and_ego_pose(nusc, first_lidar_token, max_timestamp_us, stream)
    log_radars(nusc, first_radar_tokens, max_timestamp_us, stream)
    log_annotations(nusc, location, first_sample_token, max_timestamp_us, stream)


def setup_rerun(nusc, output_dir):
    print("Outputting rrd files for nuscenes dataset")
    # blueprint dictates how the data is visualized by default
    blueprint = rrb.Vertical(
        rrb.Spatial3DView(
            name="3D",
            origin="world",
            # Default for `ImagePlaneDistance` so that the pinhole frustum visualizations don't take up too much space.
            defaults=[rr.components.ImagePlaneDistance(4.0)],
            # Transform arrows for the vehicle shouldn't be too long.
            overrides={"world/ego_vehicle": [rr.components.AxisLength(3.0)]},
        ),
        rrb.MapView(
            origin="world",
            name="MapView",
            zoom=rrb.archetypes.MapZoom(18.0),
            background=rrb.archetypes.MapBackground(
                rrb.components.MapProvider.OpenStreetMap
            ),
        ),
        row_shares=[2, 1],
    )

    all_scene_names = [scene["name"] for scene in nusc.scene]

    for scene_name in all_scene_names:
        this_scene_recording = rr.new_recording(
            application_id="nuscenes", recording_id=scene_name
        )

        log_nuscenes(
            nusc, scene_name, max_time_sec=float("inf"), stream=this_scene_recording
        )

        rrd_path = output_dir / f"{scene_name}.rrd"

        if rrd_path.exists():
            print(f"{rrd_path} already exists, overwriting...")
            rrd_path.unlink()

        this_scene_recording.save(rrd_path, default_blueprint=blueprint)
        print(f"{rrd_path} saved")


# --- FIFTYONE ---


def get_3d_colors(
    nuscenes_dir, nusc, points, token, modality=Literal["lidar", "radar"]
):
    if modality == "radar":
        point_distances = np.linalg.norm(points, axis=1)
        point_colors = cmap(norm(point_distances))
        return o3d.utility.Vector3dVector(point_colors[:, :3])

    # Grab and Generate Colormaps
    gt_from = "lidarseg"

    lidarseg_filename = str(nuscenes_dir / nusc.get(gt_from, token)["filename"])
    colormap = get_colormap()
    name2index = nusc.lidarseg_name2idx_mapping

    coloring = paint_points_label(
        lidarseg_filename, None, name2index, colormap=colormap
    )
    colors = coloring[:, :3]
    return o3d.utility.Vector3dVector(colors)


def write_pcd_file(
    nuscenes_dir, output_dir, nusc, token, modality=Literal["lidar", "radar"]
):
    filepath = str(nuscenes_dir / nusc.get("sample_data", token)["filename"])
    if modality == "radar":
        cloud = RadarPointCloud.from_file(filepath)
    else:
        cloud = LidarPointCloud.from_file(filepath)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points[:3, :].T)
    pcd.colors = get_3d_colors(nuscenes_dir, nusc, pcd.points, token, modality)

    # Save back Point Cloud
    pcd_file_name = os.path.basename(filepath).split(".")[0] + ".pcd"
    pcd_output_path = os.path.join(output_dir, pcd_file_name)
    o3d.io.write_point_cloud(pcd_output_path, pcd)
    return pcd_output_path


def get_threed_detections(nusc, lidar_token):
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(
        lidar_token,
        box_vis_level=BoxVisibility.NONE,
    )
    detections = []
    for box in boxes:

        x, y, z = box.orientation.yaw_pitch_roll
        w, l, h = box.wlh.tolist()

        detection = fo.Detection(
            label=box.name,
            location=box.center.tolist(),
            rotation=[z, y, x],
            dimensions=[l, w, h],
        )
        detections.append(detection)
    return detections


def get_camera_sample(nusc, group, filepath, sensor, token, scene):
    sample = fo.Sample(filepath=filepath, group=group.element(sensor))
    data = nusc.get("sample_data", token)
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(
        token,
        box_vis_level=BoxVisibility.NONE,
    )
    image = Image.open(data_path)
    width, height = image.size
    shape = (height, width)
    polylines = []
    log = nusc.get("log", scene["log_token"])
    location = log["location"]
    ego = nusc.get("ego_pose", data["ego_pose_token"])
    ego_list = [ego]

    latlon = derive_latlon_nu(location, ego_list)
    lat = latlon[0]["latitude"]
    lon = latlon[0]["longitude"]
    sample["location"] = fo.GeoLocation(point=[lon, lat])
    for box in boxes:
        if box_in_image(box, camera_intrinsic, shape, vis_level=BoxVisibility.ALL):
            corners = view_points(box.corners(), camera_intrinsic, normalize=True)[
                :2, :
            ]
            front = [
                (corners[0][0] / width, corners[1][0] / height),
                (corners[0][1] / width, corners[1][1] / height),
                (corners[0][2] / width, corners[1][2] / height),
                (corners[0][3] / width, corners[1][3] / height),
            ]
            back = [
                (corners[0][4] / width, corners[1][4] / height),
                (corners[0][5] / width, corners[1][5] / height),
                (corners[0][6] / width, corners[1][6] / height),
                (corners[0][7] / width, corners[1][7] / height),
            ]
            polylines.append(fo.Polyline.from_cuboid(front + back, label=box.name))

    T = transform_matrix(
        np.array(ego["translation"]), np.array(ego["rotation"]), inverse=True
    )

    sample["T_rig_world"] = T
    sample["ego_translation"] = np.array(ego["translation"])
    sample["ego_rotation"] = np.array(ego["rotation"])
    calib = nusc.get("calibrated_sensor", data["calibrated_sensor_token"])
    T_sensor_rig = transform_matrix(
        np.array(calib["translation"]),
        np.array(calib["rotation"]),
        inverse=True,
    )  # shape (4, 4)
    sample["T_sensor_rig"] = T_sensor_rig
    sample["cs_translation"] = np.array(calib["translation"])
    sample["cs_rotation"] = np.array(calib["rotation"])
    sample["cuboids"] = fo.Polylines(polylines=polylines)

    intrinsics = get_camera_intrinsics(nusc, token)
    sample["intrinsics"] = intrinsics
    return sample


def compute_ego_velocity(nusc, sample_data_token):
    """
    Estimate ego vehicle velocity by computing the difference in position over time.
    """
    curr_sd = nusc.get("sample_data", sample_data_token)
    if not curr_sd["prev"]:
        return np.array([0.0, 0.0, 0.0])  # First frame has no previous

    prev_sd = nusc.get("sample_data", curr_sd["prev"])

    # Get ego poses
    curr_pose = nusc.get("ego_pose", curr_sd["ego_pose_token"])
    prev_pose = nusc.get("ego_pose", prev_sd["ego_pose_token"])

    # Compute velocity as delta_position / delta_time
    pos_curr = np.array(curr_pose["translation"])  # x, y, z
    pos_prev = np.array(prev_pose["translation"])
    delta_pos = pos_curr - pos_prev

    delta_time = (
        curr_pose["timestamp"] - prev_pose["timestamp"]
    ) * 1e-6  # convert µs to s
    velocity = delta_pos / delta_time
    return velocity


def compute_ego_angular_velocity(nusc, sample_data_token):
    """
    Estimate ego vehicle angular velocity (rad/s) using rotation delta over time.
    Returns angular velocity as a 3D vector: [roll_rate, pitch_rate, yaw_rate].
    """
    curr_sd = nusc.get("sample_data", sample_data_token)
    if not curr_sd["prev"]:
        return np.array([0.0, 0.0, 0.0])  # No angular velocity for first frame

    prev_sd = nusc.get("sample_data", curr_sd["prev"])

    # Get ego poses
    curr_pose = nusc.get("ego_pose", curr_sd["ego_pose_token"])
    prev_pose = nusc.get("ego_pose", prev_sd["ego_pose_token"])

    # Time difference in seconds
    dt = (curr_pose["timestamp"] - prev_pose["timestamp"]) * 1e-6
    if dt == 0:
        return np.array([0.0, 0.0, 0.0])

    # Convert rotation to quaternion
    q_curr = Quaternion(curr_pose["rotation"])  # [w, x, y, z]
    q_prev = Quaternion(prev_pose["rotation"])

    # Relative rotation
    delta_q = q_curr * q_prev.inverse
    axis = np.array(delta_q.axis)
    angle = delta_q.angle  # in radians

    angular_velocity = (axis * angle) / dt  # rad/s
    return angular_velocity


def get_camera_intrinsics(nusc, sample_data_token):
    """
    Get the 3x3 camera intrinsic matrix from a sample_data token.

    Returns:
        intrinsics (np.ndarray): 3x3 camera intrinsic matrix
    """
    sample_data = nusc.get("sample_data", sample_data_token)
    calibrated_sensor = nusc.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )
    intrinsics = np.array(calibrated_sensor["camera_intrinsic"], dtype=np.float32)
    return intrinsics


def build_transform_matrix(translation, rotation):
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T


def setup_fiftyone(nusc, nuscenes_dir, output_dir):
    try:
        fo.delete_dataset("nuscenes-rerun-fo")
    except:
        pass

    dataset = fo.Dataset("nuscenes-rerun-fo", overwrite=True)
    dataset.persistent = True
    dataset.add_group_field("group", default="CAM_FRONT")

    sensor_names = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
        "LIDAR_TOP",
        "RADAR_FRONT",
        "RADAR_FRONT_LEFT",
        "RADAR_FRONT_RIGHT",
        "RADAR_BACK_LEFT",
        "RADAR_BACK_RIGHT",
    ]

    samples = []
    for scene in nusc.scene:
        my_scene = scene
        scene_name = scene["name"]
        token = my_scene["first_sample_token"]
        my_sample = nusc.get("sample", token)
        T_rig_worlds = []
        T_rig_world_timestamps_us = []

        while not my_sample["next"] == "":
            scene_token = my_sample["scene_token"]
            lidar_token = my_sample["data"]["LIDAR_TOP"]
            group = fo.Group()

            pcds = {}
            threed_detections = []

            log = nusc.get("log", scene["log_token"])
            location = log["location"]
            ego = nusc.get(
                "ego_pose", nusc.get("sample_data", lidar_token)["ego_pose_token"]
            )
            velocity = compute_ego_velocity(nusc, lidar_token)
            a_velocity = compute_ego_angular_velocity(nusc, lidar_token)
            ego_list = [ego]

            latlon = derive_latlon_nu(location, ego_list)
            lat = latlon[0]["latitude"]
            lon = latlon[0]["longitude"]
            timestamp = nusc.get("sample_data", my_sample["data"]["LIDAR_TOP"])[
                "timestamp"
            ]

            T = transform_matrix(
                np.array(ego["translation"]), np.array(ego["rotation"]), inverse=False
            )

            T_rig_worlds.append(T)
            T_rig_world_timestamps_us.append(timestamp)

            lidar_sensor_info = {}

            for sensor in sensor_names:
                data = nusc.get("sample_data", my_sample["data"][sensor])
                modality = data["sensor_modality"]
                filepath = nuscenes_dir / data["filename"]

                if modality == "lidar" or modality == "radar":
                    this_token = my_sample["data"][sensor]
                    filepath = write_pcd_file(
                        nuscenes_dir, output_dir, nusc, this_token, modality
                    )
                    pcds[sensor] = filepath

                    # skip radar annotations since they're repeated
                    if modality == "lidar":

                        threed_detections.extend(
                            get_threed_detections(nusc, this_token)
                        )
                        lidar_sensor_info["rig_timestamp"] = data["timestamp"]
                        lidar_sensor_info["T_rig_world"] = T
                        lidar_sensor_info["ego_translation"] = ego["translation"]
                        lidar_sensor_info["ego_rotation"] = ego["rotation"]
                        calib = nusc.get(
                            "calibrated_sensor", data["calibrated_sensor_token"]
                        )
                        T_sensor_rig = transform_matrix(
                            np.array(calib["translation"]),
                            np.array(calib["rotation"]),
                            inverse=False,
                        )  # shape (4, 4)
                        lidar_sensor_info["T_sensor_rig"] = T_sensor_rig
                        lidar_sensor_info["cs_translation"] = np.array(
                            calib["translation"]
                        )
                        lidar_sensor_info["cs_rotation"] = np.array(calib["rotation"])

                        lidar_sensor_info["sample_token"] = data["token"]
                        lidar_sensor_info["token"] = my_sample["token"]
                        lidar_sensor_info["ego_pose_token"] = data["ego_pose_token"]
                        lidar_sensor_info["calibrated_sensor_token"] = data[
                            "calibrated_sensor_token"
                        ]
                        lidar_sensor_info["timestamp"] = data["timestamp"]
                        lidar_sensor_info["is_key_frame"] = data["is_key_frame"]
                        lidar_sensor_info["prev"] = data["prev"]
                        lidar_sensor_info["next"] = data["next"]
                        lidar_sensor_info["scene_token"] = scene_token
                        lidar_sensor_info["scene_name"] = scene_name
                        lidar_sensor_info["location"] = fo.GeoLocation(point=[lon, lat])
                        lidar_sensor_info["velocity"] = velocity
                        lidar_sensor_info["angular_velocity"] = a_velocity

                        lidar_sensor_info["lidar"] = RrdFile(
                            filepath=str(output_dir / f"{scene_name}.rrd")
                        )

                    sample = lidar_sensor_info
                elif modality == "camera":
                    sample = get_camera_sample(
                        nusc, group, filepath, sensor, my_sample["data"][sensor], scene
                    )
                    sample["sample_token"] = data["token"]
                    sample["token"] = my_sample["token"]
                    sample["ego_pose_token"] = data["ego_pose_token"]
                    sample["calibrated_sensor_token"] = data["calibrated_sensor_token"]
                    sample["timestamp"] = data["timestamp"]
                    sample["is_key_frame"] = data["is_key_frame"]
                    sample["prev"] = data["prev"]
                    sample["next"] = data["next"]
                    sample["scene_token"] = scene_token
                    sample["scene_name"] = scene_name
                    sample["location"] = fo.GeoLocation(point=[lon, lat])
                    sample["velocity"] = velocity
                    sample["angular_velocity"] = a_velocity

                    sample["lidar"] = RrdFile(
                        filepath=str(output_dir / f"{scene_name}.rrd")
                    )
                else:
                    sample = fo.Sample(filepath=filepath, group=group.element(sensor))

                # we handle lidar separately
                if modality != "lidar" and modality != "radar":
                    samples.append(sample)

            fo3d_scene = fo.Scene(camera=fo.PerspectiveCamera(up="Z"))
            ftoken = my_sample["token"]
            fo3d_filepath = os.path.join(output_dir, f"{scene_name}_{ftoken}.fo3d")
            for sensor, pcd in pcds.items():
                fo3d_scene.add(
                    fo.PointCloud(
                        name=sensor,
                        pcd_path=pcd,
                        flag_for_projection=sensor == "LIDAR_TOP",
                    )
                )
            fo3d_scene.write(fo3d_filepath)

            fo3d_sample = fo.Sample(filepath=fo3d_filepath, group=group.element("3D"))
            fo3d_sample["ground_truth"] = fo.Detections(detections=threed_detections)
            for key, value in lidar_sensor_info.items():
                fo3d_sample[key] = value
            samples.append(fo3d_sample)

            token = my_sample["next"]

            my_sample = nusc.get("sample", token)

    dataset.add_samples(samples)
    view = dataset.group_by("scene_name", order_by="timestamp")
    dataset.save_view("ordered", view)

    print("Computing orthographic projects for the grid...")
    orthographic_images_output_dir = str(output_dir / "orthographic_images")
    fou3d.compute_orthographic_projection_images(
        dataset,
        (-1, 512),
        orthographic_images_output_dir,
        in_group_slice="3D",
        shading_mode="rgb",
    )


def get_nusc(nuscenes_dir: str):

    return nuscenes.NuScenes(version="v1.0-mini", dataroot=nuscenes_dir, verbose=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rrd",
        action="store_true",
        help="Output RRD files for lidar and radar",
    )
    parser.add_argument(
        "--fiftyone",
        action="store_true",
        help="Setup fiftyone dataset",
    )
    parser.add_argument(
        "--nuscenes-data-dir",
        type=pathlib.Path,
        default=os.environ.get("NUSCENES_DATA_DIR"),
        required=True,
        help="Path to the NuScenes data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Path to the output directory where the RRD files, PCD\
            files, FO3D files, and orthographic projection images will be stored.\
            If not provided, the files will be stored in the `data` directory.",
    )
    args = parser.parse_args()

    if not args.nuscenes_data_dir:
        print(
            "Please provide the NuScenes data directory via --nuscenes-data-dir or set the \
              NUSCENES_DATA_DIR environment variable."
        )
        exit(1)

    if not args.output_dir:
        output_dir = pathlib.Path(__file__).parent / "data"
        output_dir.mkdir(exist_ok=True)
        args.output_dir = output_dir

    nusc = get_nusc(args.nuscenes_data_dir)

    if args.rrd:
        setup_rerun(nusc, args.output_dir)
    else:
        print("Skipping outputting RRD files (--rrd not set)")

    if args.fiftyone:
        setup_fiftyone(nusc, args.nuscenes_data_dir, args.output_dir)
    else:
        print("Skipping setting up fiftyone dataset (--fiftyone not set)")

    print("Done!")


if __name__ == "__main__":
    main()
