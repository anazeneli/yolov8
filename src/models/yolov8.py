import os, sys

from pathlib import Path
from typing import ClassVar, Final, List, Mapping, Optional, Sequence, Tuple, cast
from typing_extensions import Self
from urllib.request import urlretrieve

from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import (Classification, Detection,
                                       GetPropertiesResponse)
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import *
from viam.utils import ValueTypes

from viam.utils import struct_to_dict
from ultralytics.engine.results import Results
from viam.components.camera import Camera, ViamImage
from viam.media.utils.pil import viam_to_pil_image

from ultralytics import YOLO
import torch

from viam.logging import getLogger

LOGGER = getLogger(__name__)

MODEL_DIR = os.environ.get(
    "VIAM_MODULE_DATA", os.path.join(os.path.expanduser("~"), ".data", "models")
) 

class Yolov8(Vision, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(ModelFamily("azeneli", "yolov8"), "yolov8")

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Vision service.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both implicit and explicit)

        Returns:
            Self: The resource
        """
        return super().new(config, dependencies)

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any required dependencies or optional dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Tuple[Sequence[str], Sequence[str]]: A tuple where the
                first element is a list of required dependencies and the
                second element is a list of optional dependencies
        """
        optional_dependencies, required_dependencies = [], []
        fields = config.attributes.fields

        # # Validate required dependencies
        # Validate camera name
        if "camera_name" not in fields:
            raise Exception("missing required camera_name attribute")
        elif not fields["camera_name"].HasField("string_value"):
            raise Exception("camera_name must be a string")
        
        camera_name = fields["camera_name"].string_value
        required_dependencies.append(camera_name)

        # Validate detector model location
        if "model_location" not in fields:
            raise Exception("missing required model_location attribute")
        elif not fields["model_location"].HasField("string_value"):
            raise Exception("model_location must be a string folder path")
        
        LOGGER.info(f"FIELDS {fields}")
        # # Optional dependencies 
        if "tracker_config_location" in fields:
            LOGGER.info(f"Tracking enabled") 

            # Validate tracker file type 
            if not fields["tracker_config_location"].HasField("string_value"):
                raise Exception("tracker_config_location must be a string")

    
        LOGGER.info(f"DEPENDENCIES {required_dependencies} {optional_dependencies}")
        
        return required_dependencies, []



    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        # self.dependencies = dependencies

        attrs = struct_to_dict(config.attributes) 

        # Camera 
        camera_component = dependencies[Camera.get_resource_name(attrs["camera_name"] )]
        self.camera = cast(Camera, camera_component)

        model_location = str(attrs.get("model_location"))


        self.task = str(attrs.get("task")) or None
        self.enable_tracker = False

        if "/" in model_location:
            if self.is_path(model_location):
                self.MODEL_PATH = model_location
            else:
                model_name = str(attrs.get("model_name", ""))
                if model_name == "":
                    raise Exception(
                        "model_name attribute is required for downloading models from HuggingFace."
                    ) 
                self.MODEL_REPO = model_location
                self.MODEL_FILE = model_name
                self.MODEL_PATH = os.path.abspath(
                    os.path.join(
                        MODEL_DIR,
                        f"{self.MODEL_REPO.replace('/', '_')}_{self.MODEL_FILE}",
                    )
                )
                self.get_model()

            self.model = YOLO(self.MODEL_PATH, task=self.task)
        else:
            self.model = YOLO(model_location, task=self.task)


        try: 
            tracker_config_location = str(attrs.get("tracker_config_location", ""))
            if not tracker_config_location:
                raise Exception("tracker_config_location is required when tracker is enabled")

            self.TRACKER_PATH = os.path.abspath(
                    tracker_config_location 
            )
            self.check_path(self.TRACKER_PATH)
            self.logger.info("Tracker enabled and initialized")

            self.enable_tracker = True

        except Exception as e:
            raise Exception(f"Tracker configuration failed: {str(e)}") 
            

        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.logger.info(f"Using CUDA device: {self.device}")
        # Check for Mac Metal Performance Shaders (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            self.logger.info("Using Mac GPU (Metal Performance Shaders)")
        else:
            self.device = "cpu"
            self.logger.info("Using CPU device")

        self.logger.info(f"Device: {self.device}")

        return


    def is_path(self, path: str) -> bool:
        try:
            Path(path)
            return os.path.exists(path)
        except ValueError:
            return False
        

    def get_model(self):
        if not os.path.exists(self.MODEL_PATH):
            MODEL_URL = f"https://huggingface.co/{self.MODEL_REPO}/resolve/main/{self.MODEL_FILE}"
            self.logger.debug(f"Fetching model {self.MODEL_FILE} from {MODEL_URL}")
            urlretrieve(MODEL_URL, self.MODEL_PATH, self.log_progress)


    def check_path(self, path):
        """ 
            Check if path exists on Viam machine
        """ 
        if not os.path.exists(path):
            self.logger.debug(f"Tracker path expected in model path folder.{path}")
            raise FileExistsError(path)
             

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> CaptureAllResult:
        result = CaptureAllResult()

        result.image = await self.camera.get_image(mime_type="image/jpeg")
        result.detections = await self.get_detections(result.image)

        return result

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Detection]:
        image = await self.camera.get_image(mime_type="image/jpeg")

        return await self.get_detections(image)


    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Detection]:

        detections = []
        results = self.model.track(viam_to_pil_image(image), tracker=self.TRACKER_PATH, persist=True, classes=[0], device=self.device)[0]    # handles per frame detection updates
        
        if results is None:
            return detections

        for i, (xyxy, conf, track_id) in enumerate(
                zip(results.boxes.xyxy,
                    results.boxes.conf, 
                    results.boxes.id)):
            
            # bounding box
            x1, y1, x2, y2 = xyxy.tolist()
            confidence     = round(conf.item(), 4) 
            track_id       = int(track_id.item())

            # If no keypoints in frame, no people in frame
            if results.keypoints is not None:
                kpts = results.keypoints.xy[i].tolist()  # [[x,y,conf], ...]
            else:
                return detections

            detection = {
                "class_name": track_id,  # Overwriting class name with 
                "confidence": confidence,
                "x_min": int(x1) , 
                "y_min": int(y1),
                "x_max": int(x2),
                "y_max": int(y2),
            }

            detections.append(detection)

        return detections
    

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Classification]:
        self.logger.error("`get_classifications_from_camera` is not implemented")
        raise NotImplementedError()

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[Classification]:
        self.logger.error("`get_classifications` is not implemented")
        raise NotImplementedError()

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> List[PointCloudObject]:
        self.logger.error("`get_object_point_clouds` is not implemented")
        raise NotImplementedError()

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None
    ) -> Vision.Properties:

        return GetPropertiesResponse(
            classifications_supported=True,
            detections_supported=True,
            object_point_clouds_supported=False
        )
    
    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`do_command` is not implemented")
        pass

