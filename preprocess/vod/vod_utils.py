import os
import torch
import json
from tqdm import tqdm


class vodFormatter:
    scene_split = [
        ['00000', '00543'],
        ['00544', '01311'],
        ['01312', '01802'],
        ['01803', '02199'],
        ['02200', '02531'],
        ['02532', '02797'],
        ['02798', '03276'],
        ['03277', '03574'],
        ['03575', '03609'],
        ['03610', '04047'],
        ['04049', '04386'],
        ['04387', '04651'],
        ['04652', '05085'],
        ['06334', '06570'],
        ['06571', '06758'],
        ['06759', '07542'],
        ['07543', '07899'],
        ['07900', '08197'],
        ['08198', '08480'],
        ['08481', '08748'],
        ['08749', '09095'],
        ['09096', '09517'],
        ['09518', '09775'],
        ['09776', '09930']
    ]
    skip_scenes = ['01262', '01272', '01282', '01292', '01302', '01312']

    def __init__(self, data_path, prompt_path, instruction_path):
        self.data_path = data_path
        self.prompt_path = prompt_path
        self.instruction_path = instruction_path
        self.load_from_path()

    def vod2sharegpt(self, json_path=None):
        annotations = []

        for frame_number in tqdm(self.frame_numbers):
            if not self.check_number_in_range(frame_number) or frame_number in self.skip_scenes:
                continue 

            instance = {}
            instance["image"] = frame_number + ".jpg"
            instance["conversations"] = [{"from": "human"}, {"from": "gpt"}]
            instance["conversations"][0]["value"] = f"""The car is {self.instruction_map[self.instructions[frame_number]]}. {self.prompts["waypoint_description"]}"""
            instance["conversations"][1]["value"] = self.format_output(self.organize_next_steps(frame_number))
            

            
            annotations.append(instance)

        if json_path is not None:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=4)

        return annotations
    

    def vod2xxx(self, json_path=None):

        annotations = []

        for frame_number in tqdm(self.frame_numbers):
            if not self.check_number_in_range(frame_number) or frame_number in self.skip_scenes:
                continue 

            instance = {}
            instance["messages"] = [{"role": "user", 
                                    "content": [{"type": "image", "image": None},
                                                {"type": "text", "text": None}]}, 
                                    {"role": "assistant",
                                     "content": [{"type": "text", "text": None}]}]
            
            instance["messages"][0]["content"][0]["image"] = os.path.join("data/raw/view_of_delft_PUBLIC/lidar/training/image_2", frame_number + ".jpg")
            instance["messages"][0]["content"][1]["text"] = f"""The car is {self.instruction_map[self.instructions[frame_number]]}. {self.prompts["waypoint_description"]}"""
            # instance["messages"][1]["content"][0]["text"] = self.format_output(frame_number, self.organize_next_steps(frame_number))
            instance["messages"][1]["content"][0]["text"] = self.format_output_2(self.organize_next_steps(frame_number))

            annotations.append(instance)

        if json_path is not None:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=4)

        return annotations
    

    def load_from_path(self):
        self.prompts = self.load_prompts(self.prompt_path)

        frame_path = os.path.join(self.data_path, "radar_5frames", "ImageSets", "train.txt")
        # frame_path = os.path.join(self.data_path, "radar_5frames", "ImageSets", "test.txt")
        self.frame_numbers = [line.strip() for line in open(frame_path)]

        self.instruction_map = {"0": "driving straight", "1": "turning left", "2": "turning right", "3": "waiting"}
        self.instructions = {line.split(',')[0]: line.split(',')[1].strip() for line in open(self.instruction_path)}

    
    def load_prompts(self, prompts_path):
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        return prompts

    def check_number_in_range(self, n, diff=50):
        for low_bound, high_bound in self.scene_split:
            low_bound, high_bound, n = int(low_bound), int(high_bound), int(n)
            if low_bound <= n <= high_bound and low_bound <= n + diff <= high_bound:
                return True
        return False

    def format_output(self, frame_number, tensor_var):
        tensor_str = ", ".join([f"[{', '.join([f'{x.item():.2f}' for x in row])}]" \
            for row in tensor_var])
        return f"[{tensor_str}]"
    
    def format_output_2(self, tensor_var):
        result = []
    
        for i, point in enumerate(tensor_var):
            point_dict = {
                "point_2d": [round(point[0].item(), 2), round(point[1].item(), 2)],
                "label": i
            }
            # print(point_dict)
            result.append(point_dict)

        result = json.dumps(result)
        # print(result)
        
        return result

    def organize_next_steps(
            self, 
            first_frame_number, 
            upper_interval=5, 
            dt=0.1, 
            steps=10):

        total_frame_interval = int(upper_interval // (dt * steps))
        frame_numbers = [f"{(int(first_frame_number) + i * total_frame_interval):05d}" 
                         for i in range(1, steps + 1)]
        points = torch.stack([self.next_frame_in_base_frame(first_frame_number, ii) \
            for ii in frame_numbers])
        xy_form_tensor = torch.round(points[:, [0, 2]], decimals=2)
        return xy_form_tensor

    def next_frame_in_base_frame(
            self, 
            base_frame_number, 
            next_frame_number, 
            matrix_label="odomToCamera"
        ):
        odomToCamera_base = self.read_matrix_from_json(base_frame_number, matrix_label)
        odomToCamera_next = self.read_matrix_from_json(next_frame_number, matrix_label)
        todom_base = odomToCamera_base[:3, 3]
        todom_next = odomToCamera_next[:3, 3]
        odom_displacement = todom_next - todom_base
        R_base = odomToCamera_base[:3, :3]
        R_base_inv = torch.inverse(R_base)
        transformed_displacement = R_base_inv @ odom_displacement
        return transformed_displacement

    def read_matrix_from_json(self, frame_number, matrix_key):
        file_path = os.path.join(self.data_path, "radar_5frames/training/pose", frame_number+".json")
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        for line in lines:
            try:
                data = json.loads(line)
                if matrix_key in data:
                    matrix_data = data[matrix_key]
                    if len(matrix_data) == 16:
                        return torch.tensor(matrix_data, dtype=torch.float32).view(4, 4)
            except json.JSONDecodeError:
                continue
        
        raise ValueError(f"Cannot find the {matrix_key} data")
    

if __name__ == "__main__":
    data_path = "data/raw/view_of_delft_PUBLIC"
    prompt_path = "data/utils/prompts.json"
    instruction_path = "data/utils/vod_scene_mapping.txt"
    json_path = "data/processed/vod/vod.json"
    # json_path = "data/processed/vod/vod_test.json"
    vodFormatter(data_path, prompt_path, instruction_path).vod2xxx(json_path)