# 🧪 AV Dataset Sanity Toolkit

This repository provides a modular, extensible toolkit for validating the **calibration, completeness, and visual fidelity** of large-scale autonomous driving datasets — with current support for **Waymo Open Dataset** and **nuScenes**. It is designed to help **ML engineers** working on AV/ADAS pipelines ensure that their datasets are clean, calibrated, and ready for downstream use in tools like **FiftyOne**, **Omniverse**, **Cosmos**, or **NUREC**.

---

## 🚦 Why This Matters

In real-world AV data pipelines, even minor inconsistencies — misaligned timestamps, broken sensor calibration, or missing sensor streams — can invalidate training or render simulation unusable. This toolkit helps catch those issues early by:

- Verifying **sensor presence** and **temporal consistency**
- Visualizing **LIDAR-camera alignment**
- Surfacing **metadata integrity issues**
- Supporting future conversion to **ncore** or **cosmos** formats

---

## 🔍 What It Checks

This toolkit performs the following sanity checks and transformations:

- ✅ **Metadata integrity**: Missing fields, malformed records, etc.
- 🕓 **Timestamp alignment**: Across all sensor modalities
- 📉 **Missing sensor streams**: Camera or LIDAR views missing from certain frames
- 🌀 **LIDAR to camera projection**: Verify calibration by overlaying PCD points onto camera frames
- 📂 **Sensor playback inspection**: Frame-by-frame review of each sensor stream
- 🔧 **Label alignment** *(coming soon)*: Cross-checking annotation presence and temporal sync
- 🔍 **PCD alignment**: Verifying 3D consistency across frames and sensors

All validation is visualizable with **[FiftyOne](https://voxel51.com/fiftyone/)**.

---

## 📁 Supported Formats

| Source Dataset | Supported Modalities             | Export Formats              |
|----------------|----------------------------------|-----------------------------|
| Waymo          | Images, LIDAR, Calibration, Poses | `.jpg`, `.pcd`, `.fo3d`, FiftyOne |
| nuScenes       | Images, LIDAR, Calibration, Poses | `.jpg`, `.pcd`, `.fo3d`, FiftyOne |

Export support for **ncore** and **cosmos** formats is coming soon.

---

## 🛠️ Pipeline Overview

```bash
1. Load TFRecords or nuScenes logs
2. Decode and extract:
    - Camera images
    - LIDAR point clouds
    - Calibration matrices
    - Ego poses
3. Convert to standard output formats (.jpg, .pcd, .fo3d)
4. Create grouped FiftyOne samples per frame
5. Run visual inspection + projection validation
```
| **Filename**                     | **Description**                                                                                                                                |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_sanity_waymo.ipynb`        | Validates Waymo dataset exports. Checks calibration, timestamps, sensor coverage, and LIDAR-to-camera projections using FiftyOne.              |
| `data_sanity_nuscenes.ipynb`     | Validates nuScenes data. Performs playback, sensor integrity, calibration checks, and grouping for visualization in FiftyOne.                  |
| `waymo-importer.ipynb`           | Converts Waymo TFRecords into structured `.jpg`, `.pcd`, and `.fo3d` outputs. Handles per-frame extraction and grouped sample creation.        |
| `load-nuscenes.py`               | Loads nuScenes data and prepares grouped samples for FiftyOne. Handles image + LIDAR sync, transform handling, and sensor completeness checks. |
| `fiftyone_to_ncore.py`           | Prototype script to convert a FiftyOne dataset into Nucleus Core (ncore) format. Supports Cosmos/NUREC ingestion pipeline compatibility.       |
| `nvidia_data_sanity_check.ipynb` | A unified demo notebook that applies all key sanity checks (timestamps, calibration, projections, playback) across both Waymo and nuScenes.    |


### Known issues

- fiftyone_to_ncore is still a work in progress. The template is there for going to fiftyone to ncore but this is several months old now
- Waymo open dataset pip package only works on Linux! To grab the dataset sample without going through the pain, get in on hugging face [here](https://huggingface.co/datasets/dgural/fo_waymo_sample) 
- Even if you go to linux, many waymo scripts are bugged when decoding and use `bytearray` instead of `bytes`. This will fix any potential error you see.