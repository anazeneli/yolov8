# Module: yolov8-tracker

This module is an **extension of the Viam YOLOv8 module**, adding support for **tracking objects across frames** using tracking algorithms like **BoT-SORT** and **ByteTrack**. It allows you to maintain consistent object identities in a video stream, which is essential for multi-object tracking (MOT) use cases.

---

## Model: `azeneli:yolov8:yolov8`

This model wraps YOLOv8 with tracking capabilities. It takes as input a video stream from a configured camera and uses a YOLOv8 model to detect objects, while maintaining object tracking state across frames using your chosen tracker.

---

### Configuration

The following attribute template is used to configure the model:

```json
{
  "camera_name": "<string>",
  "tracker_config_location": "<string>",
  "model_location": "<string>"
}


Attributes
Name	| Type	| Inclusion |	Description
camera_name	| string	| Required	 | Name of the camera resource to use as the input stream
tracker_config_location | string	| Required	| Path to the YAML config file for the tracker (e.g. BoT-SORT, ByteTrack)
model_location |	string	| Required	| Path to the YOLOv8 model weights (e.g. .pt file for pose or object detection)


{
  "camera_name": "camera-1",
  "tracker_config_location": "/home/viam/yolov8-tracker/src/configs/botsort.yaml",
  "model_location": "/home/viam/yolov8-tracker/src/weights/yolov8n-pose.pt"
}


