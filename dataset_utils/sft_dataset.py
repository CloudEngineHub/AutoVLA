import gc
import os
import cv2
import glob
import torch
import base64
import json
import numpy as np
from pathlib import Path
from PIL import Image
from models.action_tokenizer import ActionTokenizer
from torch.utils.data import Dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from navsim.common.dataloader import SceneLoader
from navsim.agents.autovla_agent import AutoVLAAgent
from navsim.common.dataclasses import SceneFilter
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

IGNORE_INDEX = -100

class SFTDataset(Dataset):
    def __init__(self, data_config, model_config, processor, using_cot=True):
        data_paths = data_config['json_dataset_path']
        self.sensor_data_path = data_config['sensor_data_path']

        # Handle both single path (string/Path) and multiple paths (list)
        if isinstance(data_paths, (str, Path)):
            self.data_paths = [Path(data_paths)]
        else:
            self.data_paths = [Path(path) for path in data_paths]
            
        self.processor = processor
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer, model_config=model_config)
        # Flag to control whether to use CoT in training data
        self.using_cot = using_cot
        
        trajectory_sampling = TrajectorySampling(time_horizon=model_config['trajectory']['time_horizon'], 
                                                interval_length=model_config['trajectory']['interval_length'])
        nuplan_agent = AutoVLAAgent(trajectory_sampling=trajectory_sampling,
                                    sensor_data_path=self.sensor_data_path,
                                   codebook_cache_path=model_config['codebook_cache_path'],
                                   skip_model_load=True)

        self._agent = nuplan_agent
        
        # Get all JSON files from all data paths
        self.scenes = []
        for data_path in self.data_paths:
            path_scenes = sorted(list(data_path.glob('*.json')))
            self.scenes.extend(path_scenes)
            
        if not self.scenes:
            raise ValueError(f"No JSON files found in any of the provided data paths: {self.data_paths}")

        # Initialize any other necessary attributes here

    def __len__(self):
        # return len(self._scene_loader.tokens)
        return len(self.scenes)

    def __getitem__(self, idx):
        # Load data from JSON file
        input_features: Dict[str, torch.Tensor] = {}
        target_trajectory: Dict[str, torch.Tensor] = {}
        
        scene_path = self.scenes[idx]
        with open(scene_path, 'r') as f:
            scene_data = json.load(f)
            
        for builder in self._agent.get_feature_builders():
            input_features.update(builder.compute_features(scene_data))
        for builder in self._agent.get_target_builders():
            target_trajectory.update(builder.compute_targets(scene_data))
        
        # image sensor
        images = input_features['images']
        camera_images = {}
        
        # List of all camera types
        camera_types = ['front_camera', 'back_camera', 'front_left_camera', 'front_right_camera', 'left_camera', 'right_camera']
        
        if input_features['sensor_data_path']:
            for camera_type in camera_types:
                camera_images[camera_type] = []
                for i in range(4):
                    img = images[camera_type][i]
                    camera_images[camera_type].append(
                        os.path.join(input_features['sensor_data_path'], img))

        # Assign to individual variables for message formatting
        front_camera_1, front_camera_2, front_camera_3, front_camera_4 = camera_images['front_camera']
        back_camera_1, back_camera_2, back_camera_3, back_camera_4 = camera_images['back_camera']
        front_left_camera_1, front_left_camera_2, front_left_camera_3, front_left_camera_4 = camera_images['front_left_camera']
        front_right_camera_1, front_right_camera_2, front_right_camera_3, front_right_camera_4 = camera_images['front_right_camera']

        # vehicle state
        velocity = input_features["vehicle_velocity"]
        if isinstance(velocity, list) or isinstance(velocity, np.ndarray):
            velocity_x = velocity[0]
            velocity_y = velocity[1]
            velocity = np.sqrt(velocity_x**2 + velocity_y**2)
            
        acceleration = input_features["vehicle_acceleration"]
        if isinstance(acceleration, list) or isinstance(acceleration, np.ndarray):
            acceleration_x = acceleration[0]
            acceleration_y = acceleration[1]
            acceleration = np.sqrt(acceleration_x**2 + acceleration_y**2)

        instruction = input_features["driving_command"].lower()

        # trajectory
        gt_action_idx = target_trajectory['gt_idx']
        gt_raw_trajectory = target_trajectory['gt_pos_raw']


        # get assistent content for different datasets
        gt_cot = scene_data['cot_output']

        
        if not self.using_cot:
            assistant_content = [
                {
                    "type": "text",
                    "text": (
                        "<answer>\n"
                        "The final output action is: " + self.action_tokenizer(gt_action_idx[0]) + "\n"
                        "</answer>"
                    )
                }
            ]
            system_content = [
                {
                    "type": "text",
                    "text": (
                        "You are an Advanced Driver Assistance and Full Self-Driving System. "
                        "You will be provided with video observations from the ego vehicle's surrounding cameras, along with the vehicle's current dynamic states. "
                        "Your task is to predict the most appropriate driving action for the next five seconds."
                    )
                }
            ]

            has_cot = False
        else:
            # system_content = [
            #     {
            #         "type": "text",
            #         "text": (
            #             "You are an Advanced Driver Assistance and Full Self-Driving System. "
            #             "You will receive visual observations from the ego vehicle's cameras and dynamic information about the vehicle's current state. "
            #             "Your task is to predict the optimal driving action for the next five seconds.\n\n"
            #             "First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.\n\n"
            #             "If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.\n\n"
            #             "Structure your reasoning as follows:\n"
            #             "1. **Scene Analysis**: Describe the traffic situation, including relevant environmental cues such as traffic lights, lane markings, and the behaviors of surrounding vehicles or pedestrians.\n"
            #             "2. **Identification of Critical Objects**: Identify two to three critical road users or obstacles, specifying their relative positions to the ego vehicle.\n"
            #             "3. **Prediction of Critical Object Behavior**: Predict the potential movements of the identified critical objects.\n"
            #             "4. **Ego Vehicle Intent Reasoning**: Based on the observed environment and current vehicle state, reason about the desired intent of the ego vehicle.\n"
            #             "5. **Final Action Decision**: Select one lateral action and one longitudinal action:\n"
            #             "- **Lateral actions** (choose exactly one): [move forward, turn left, change lane to left, turn right, change lane to right]\n"
            #             "- **Longitudinal actions** (choose exactly one): [stop, deceleration to zero, maintain constant speed, quick deceleration, deceleration, quick acceleration, acceleration]\n\n"
            #             "Present the final action clearly after your reasoning steps."
            #         )
            #     }
            # ]

            system_content = [
                {
                    "type": "text",
                    "text": (
                        "You are an Advanced Driver Assistance and Full Self-Driving System. "
                        "You will receive visual observations from the ego vehicle's cameras and dynamic information about the vehicle's current state. "
                        "Your task is to predict the optimal driving action for the next five seconds.\n\n"
                        "First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.\n\n"
                        "If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.\n\n"
                        "Present the final action clearly after your reasoning steps."
                    )
                }
            ]

            # Otherwise, check dataset type and CoT availability
            if scene_data["dataset_name"] == "nuplan" or scene_data["dataset_name"] == "waymo" :
                has_cot = False
                if isinstance(gt_cot, str):
                    assistant_content = [
                        {
                            "type": "text",
                            "text": (
                                "<think>\n"
                                "This is a complex scenario requiring additional reasoning.\n"
                                f"{gt_cot}\n"
                                "</think>\n"
                                "<answer>\n"
                                "The final output action is: " + self.action_tokenizer(gt_action_idx[0]) + "\n"
                                "</answer>"
                            )
                        }
                    ]
                    has_cot = True
                else:
                    assistant_content = [
                        {
                            "type": "text",
                            "text": (
                                "<think>\n"
                                "This is a straightforward scenario, and a direct decision can be made.\n"
                                "</think>\n"
                                "<answer>\n"
                                "The final output action is: " + self.action_tokenizer(gt_action_idx[0]) + "\n"
                                "</answer>"
                            )
                        }
                    ]
            else:
                print(scene_data['dataset_name'])
                exit()


        user_content = [
            {
                "type": "text",
                "text": (
                    "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment."
                )
            },
            {
                "type": "text",
                "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": 28 * 28 * 140,
                "max_pixels": 28 * 28 * 140,
                "video": [
                    f"file://{front_camera_1}",
                    f"file://{front_camera_2}",
                    f"file://{front_camera_3}",
                    f"file://{front_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": 28 * 28 * 140,
                "max_pixels": 28 * 28 * 140,
                "video": [
                    f"file://{front_left_camera_1}",
                    f"file://{front_left_camera_2}",
                    f"file://{front_left_camera_3}",
                    f"file://{front_left_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."
            },
            {
                "type": "video",
                "min_pixels": 28 * 28 * 140,
                "max_pixels": 28 * 28 * 140,
                "video": [
                    f"file://{front_right_camera_1}",
                    f"file://{front_right_camera_2}",
                    f"file://{front_right_camera_3}",
                    f"file://{front_right_camera_4}",
                ]
            },
            {
                "type": "text",
                "text": (
                    f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is {acceleration:.3f} m/s². "
                    f"The driving instruction is: {instruction}. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."
                )
            },
        ]

        # create messages
        messages = [
            {   
                "role": "system",
                "content": system_content
            },

            {
                "role": "user",
                "content": user_content
            },

            # assistant response
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]


        # process the images and messages
        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
    
        inputs = {'text': text, 'image_inputs': image_inputs, 'video_inputs': video_inputs}

        # trajectory information
        inputs['gt_trajectory'] = gt_raw_trajectory
        inputs['gt_action'] = gt_action_idx
        inputs['has_cot'] = has_cot
        inputs['data_path'] = scene_path
        # Force garbage collection to free temporary objects
        # gc.collect()

  
        return inputs


@dataclass
class DataCollator:
    processor: AutoProcessor
    ignore_index: int = -100
    assistant_id: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.assistant_id is None:
            self.assistant_id = [151644, 77091]  # default value for Qwen2.5-VL

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract text and image inputs
        text = [batch["text"] for batch in features]
        image_inputs = [batch["image_inputs"] for batch in features]
        
        # Process all 6 videos
        video_inputs = []
        image_inputs = []
        has_cot = []
        data_path = []
        for batch in features:
            # Each batch has 6 videos (one per camera)
            batch_videos = batch["video_inputs"]
            video_inputs.extend(batch["video_inputs"])
            batch_images = batch["image_inputs"]
            image_inputs.append(batch_images)
            batch_has_cot = batch["has_cot"]
            has_cot.append(batch_has_cot)
            batch_data_path = batch["data_path"]
            data_path.append(batch_data_path)
        
        
        batch = self.processor(
            text=text,
            images=image_inputs if image_inputs[0] is not None else None,
            videos=video_inputs if video_inputs[0] is not None else None,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        # Find the start of the assistant response
        assistant_id = torch.tensor(self.assistant_id)
        for i in range(labels.shape[0]):
            for j in range(len(labels[i]) - len(assistant_id) + 1):
                if torch.equal(labels[i][j:j + len(assistant_id)], assistant_id):
                    start_idx = j
                    break
            
            # [CRITICAL] We take the losses on the assistant response only         
            labels[i, :start_idx] = self.ignore_index
        
        # add labels and gt action to the batch
        batch['labels'] = labels
        batch['gt_trajectory'] = torch.stack([batch['gt_trajectory'] for batch in features])
        batch['gt_action'] = torch.stack([batch['gt_action'] for batch in features])
        batch['has_cot'] = torch.tensor(has_cot)
        # batch['data_path'] = data_path

        return batch