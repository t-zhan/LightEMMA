import numpy as np
from pyquaternion import Quaternion
import cv2
import json
import pandas as pd
import glob
import os
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from utils import compute_speed, compute_curvature, quaternion_to_yaw, global_to_ego_frame


def _convert_to_python_types(obj):
    """将numpy类型转换为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, list):
        return [_convert_to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_python_types(item) for item in obj)
    else:
        return obj


def _format_positions_string(positions):
    """将位置数据格式化为带3位小数的字符串"""
    formatted_positions = []
    for pos in positions:
        formatted_pos = [round(float(coord), 3) for coord in pos]
        formatted_positions.append(formatted_pos)
    return str(formatted_positions)


class nuscFormatter:
    def __init__(self, data_path, qa_path, instruction_path):
        self.data_path = data_path
        self.qa_path = qa_path
        self.instruction_path = instruction_path
        self.load_from_path()

    def nusc2qwen(self, json_path=None):
        annotations = []
        
        for sample in tqdm(self.samples):
            try: 
                scene = list(filter(lambda x: x.get("token") == sample["scene_token"], self.scenes))[0]
                scene_number = int(scene["name"][-4:])

                instance = {}
                instance["sample_token"] = sample["token"]
                instance["scene_token"] = sample["scene_token"]
                instance["scene_name"] = scene["name"]

                img_paths = self.get_imgs(self.nusc_trainval, sample["token"])
                instance["image"] = img_paths

                instance["conversations"] = []
                
                # 第一轮对话
                scene_description = scene["description"]
                instance["conversations"].append({"from": "human", "value": self.prompts["scene_prompt"]})
                instance["conversations"].append({"from": "gpt", "value": scene_description})

                # 第二轮对话
                prev_speed, prev_curvatures, _ = self.get_actions(sample["token"], prev=True)
                intent_prompt = self.prompts["intent_prompt"].format(
                    scene_description=scene_description,
                    prev_speed=prev_speed, 
                    prev_curvatures=prev_curvatures
                )
                gt_speed, gt_curvatures, gt_positions = self.get_actions(sample["token"], prev=False)
                instruction = self.get_instructions(scene_number)
                intent_response = self.get_intent_response(prev_speed, prev_curvatures, gt_speed, gt_curvatures, instruction)
                instance["conversations"].append({"from": "human", "value": intent_prompt})
                instance["conversations"].append({"from": "gpt", "value": intent_response})

                # 第三轮对话
                waypoint_prompt = self.prompts["waypoint_prompt"].format(
                    scene_description=scene_description,
                    prev_speed=prev_speed,
                    prev_curvatures=prev_curvatures,
                    driving_intent=intent_response
                )
                instance["conversations"].append({"from": "human", "value": waypoint_prompt})
                instance["conversations"].append({"from": "gpt", "value": gt_positions})
                
                annotations.append(instance)
            except:
                continue

        if json_path is not None:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=4)

        return annotations

    def load_from_path(self):
        self.nusc_trainval = NuScenes(version='v1.0-trainval', dataroot=self.data_path, verbose=True)

        sample_path = os.path.join(self.data_path, "v1.0-trainval", "sample.json")
        with open(sample_path, 'r') as file:
            self.samples = json.load(file)
        
        prompts_path = os.path.join("data", "utils", "nusc2qwen_prompts.json")
        with open(prompts_path, "r", encoding="utf-8") as f:
            self.prompts = json.load(f)

        scene_desc_path = os.path.join(self.data_path, "v1.0-trainval", "scene.json")
        with open(scene_desc_path, 'r') as file:
            self.scenes = json.load(file)

        csv_files = glob.glob(os.path.join(self.instruction_path, '*.csv'))
        instructions = []
        for file_path in csv_files:
            df = pd.read_csv(file_path)
            instructions.append(df)
        self.instructions = pd.concat(instructions, ignore_index=True)

    def get_actions(self, sample_token, obs_len=6, prev=True):
        """
        获取速度和曲率数据，数据不完整时抛出异常
        
        :param sample_token: 当前样本token
        :param obs_len: 观测长度，默认6
        :param prev: True表示计算历史数据，False表示计算未来数据
        :return: 速度和曲率列表
        """
        # 获取位置数据
        ego_positions = []
        obs_time = []
        
        # 收集样本 (历史或未来)
        sample_tokens = []
        current_token = sample_token
        
        # 根据prev参数决定遍历方向
        direction = 'prev' if prev else 'next'
        
        for _ in range(obs_len + 2):
            if current_token:
                sample_tokens.append(current_token)
                sample = self.nusc_trainval.get('sample', current_token)
                current_token = sample.get(direction)
            else:
                break
        
        # 数据不完整时直接抛出异常
        if len(sample_tokens) < obs_len + 1:
            data_type = "historical" if prev else "future"
            raise ValueError(f"Insufficient {data_type} data: got {len(sample_tokens)} samples, need {obs_len + 1}")
        
        # 根据prev参数决定是否需要反转
        if prev:
            # 历史数据需要反转获得时间正序
            sample_tokens = sample_tokens[::-1]
        # 未来数据已经是时间正序，无需反转
        
        # 获取位置和时间戳
        for token in sample_tokens:
            sample = self.nusc_trainval.get('sample', token)
            cam_front_data = self.nusc_trainval.get('sample_data', sample['data']['CAM_FRONT'])
            ego_state = self.nusc_trainval.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_positions.append(tuple(ego_state['translation'][0:2]))
            obs_time.append(ego_state['timestamp'])
        
        # 获取当前帧信息用于坐标转换
        current_sample = self.nusc_trainval.get('sample', sample_token)
        cam_front_data = self.nusc_trainval.get('sample_data', current_sample['data']['CAM_FRONT'])
        ego_state = self.nusc_trainval.get('ego_pose', cam_front_data['ego_pose_token'])
        cur_pos = tuple(ego_state['translation'][0:2])
        cur_heading = quaternion_to_yaw(ego_state['rotation'])
        
        # 转换到ego坐标系
        obs_pos = global_to_ego_frame(cur_pos, cur_heading, ego_positions)
        
        # 计算速度和曲率
        speed = compute_speed(obs_pos, obs_time)
        curvatures = compute_curvature(obs_pos)

        # 提取位置数据（移除前2个用于计算的点）
        positions = obs_pos[2:]
        
        # 转换为Python原生类型
        speed = _convert_to_python_types(speed)
        curvatures = _convert_to_python_types(curvatures)
        positions = _convert_to_python_types(positions)
        
        positions = _format_positions_string(positions)
        
        return speed, curvatures, positions

    def get_intent_response(self, prev_speed, prev_curvatures, gt_speed, gt_curvatures, instruction):
        """
        根据历史和未来数据生成驾驶意图响应
        """
        # 分析历史速度变化
        speed_change = prev_speed[-1] - prev_speed[0]
        if speed_change > 0.5:
            speed_intent = f"accelerating from {prev_speed[0]:.3f} m/s to {prev_speed[-1]:.3f} m/s (by {speed_change:.3f} m/s)"
        elif speed_change < -0.5:
            speed_intent = f"decelerating from {prev_speed[0]:.3f} m/s to {prev_speed[-1]:.3f} m/s (by {abs(speed_change):.3f} m/s)"
        else:
            speed_intent = f"maintaining speed around {prev_speed[-1]:.3f} m/s"
        
        # 分析历史转向行为
        curvature_change = prev_curvatures[-1] - prev_curvatures[0] if prev_curvatures else 0
        if curvature_change > 0.02:
            turn_intent = f"turning more left with curvature changing from {prev_curvatures[0]:.3f} to {prev_curvatures[-1]:.3f} (curvature change {curvature_change:.3f})"
        elif curvature_change < -0.02:
            turn_intent = f"turning more right with curvature changing from {prev_curvatures[0]:.3f} to {prev_curvatures[-1]:.3f} (curvature change {curvature_change:.3f})"
        else:
            turn_intent = f"following the lane with stable curvature around {prev_curvatures[-1]:.3f}"
        
        # 分析未来建议 (基于ground truth)
        future_speed_change = gt_speed[-1] - gt_speed[0]
        if future_speed_change > 0.5:
            future_speed = f"accelerate from {gt_speed[0]:.3f} m/s to {gt_speed[-1]:.3f} m/s (by {future_speed_change:.3f} m/s)"
        elif future_speed_change < -0.5:
            future_speed = f"decelerate from {gt_speed[0]:.3f} m/s to {gt_speed[-1]:.3f} m/s (by {abs(future_speed_change):.3f} m/s)"
        else:
            future_speed = f"maintain speed around {gt_speed[-1]:.3f} m/s"
        
        # 分析未来转向建议
        future_curvature_change = gt_curvatures[-1] - gt_curvatures[0] if gt_curvatures else 0
        if future_curvature_change > 0.02:
            future_turn = f"turn more left with curvature changing from {gt_curvatures[0]:.3f} to {gt_curvatures[-1]:.3f} (curvature change {future_curvature_change:.3f})"
        elif future_curvature_change < -0.02:
            future_turn = f"turn more right with curvature changing from {gt_curvatures[0]:.3f} to {gt_curvatures[-1]:.3f} (curvature change {future_curvature_change:.3f})"
        else:
            future_turn = f"follow the lane with stable curvature around {gt_curvatures[-1]:.3f}"
        
        # 生成完整响应
        if instruction:
            response = (
                f"The ego vehicle's intent is to {instruction}. "
                f"Based on the historical data, the ego vehicle was previously {speed_intent} and {turn_intent}. "
                f"Considering the current situation from the surround-view analysis, the vehicle's intent, and the historical data, "
                f"the ego should {future_speed} and {future_turn} in the next 3 seconds."
            )
        else:
            response = (
                f"Based on the historical data, the ego vehicle was previously {speed_intent} and {turn_intent}. "
                f"Considering the current situation from the surround-view analysis and the historical data, "
                f"the ego should {future_speed} and {future_turn} in the next 3 seconds."
            )
        
        return response
    
    # Get image by sample token, using this function
    def get_imgs(self, nusc, current_sample_token):
        # Get the current sample information
        current_sample = nusc.get('sample', current_sample_token)

        # Get the front camera data
        cams = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        img_paths = []
        for cam in cams:
            cam_data_token = current_sample['data'][cam]
            # cam_front_data = nusc.get('sample_data', cam_front_data_token)
            
            # ADD MORE FIGURES AND LIDAR IF NEEDED
            
            # Save and read the front camera image
            img_paths.append(nusc.get_sample_data_path(cam_data_token))
        # img = cv2.imread(cam_front_path)
        
        return img_paths
    
    def get_instructions(self, scene_number):
        valid_instructions = self.instructions[self.instructions['Scene Number'] == scene_number]['Instruction'] \
        .dropna() \
        .unique() \
        .tolist()

        if not valid_instructions:
            return ''
        
        # 处理每个指令：所有字母转小写，句号转逗号，去掉末尾标点
        processed_instructions = []
        for instruction in valid_instructions:
            # 所有字母转小写
            instruction = instruction.lower()
            
            # 句号转逗号
            instruction = instruction.replace('.', ',')
            
            # 去掉末尾的标点符号
            instruction = instruction.strip()
            while instruction and instruction[-1] in ',!?;:':
                instruction = instruction[:-1].strip()
            
            processed_instructions.append(instruction)
        
        return ', '.join(processed_instructions)

    # High level instruction part
    # This function is used to find the scene token corresponding to a given sample token in a JSON file.
    def find_scene_token(self, json_file_path, sample_token):
        try:
            # Open the JSON file and load its content.
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            # Iterate through each item in the JSON data.
            for item in data:
                # Check if the 'token' field of the current item matches the given sample token.
                if item.get('token') == sample_token:
                    # If a match is found, return the corresponding scene token.
                    return item.get('scene_token')

            # If no match is found, return None.
            return None

        # Handle the case where the file is not found.
        except FileNotFoundError:
            print(f"Error: File {json_file_path} not found.")
        # Handle the case where the file is not in valid JSON format.
        except json.JSONDecodeError:
            print(f"Error: File {json_file_path} is not in valid JSON format.")


if __name__ == "__main__":
    data_path = "data/raw/nusc/nuscenes"
    qa_path = "data/raw/nusc/nuScenes_QA"
    instruction_path = "data/raw/nusc/doScenes"
    dst_path = "data/processed/nusc/nusc2qwen.json"
    nuscFormatter(data_path, qa_path, instruction_path).nusc2qwen(dst_path)