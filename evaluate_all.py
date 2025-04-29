import os
import re
import glob
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from collections import OrderedDict, defaultdict
from matplotlib import font_manager

from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="LightEMMA: Evaluate All Models and Generate Comparisons")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration file (default: config.yaml)")
    parser.add_argument("--no_vis", action="store_true",
                        help="Disable merged frame visualization generation")
    return parser.parse_args()

def generate_error_all(model_dirs, baseline_dir, output_dir):
    """
    Generate error_all.txt by combining error frames from all models.

    Args:
        model_dirs (dict): Dictionary of {model_name: model_path}
        baseline_dir (str): Path to baseline results directory
        output_dir (str): Directory to save error_all.txt

    Returns:
        set: Set of (scene, frame) tuples representing error frames
    """
    # Dictionary to track which models have errors for each (scene, frame)
    error_frames = defaultdict(list)

    # Add baseline to the model directories for processing
    all_dirs = model_dirs.copy()
    if baseline_dir:
        all_dirs["baseline"] = baseline_dir

    # Process each model's error.txt
    for model_name, model_path in all_dirs.items():
        error_file_path = os.path.join(model_path, "analysis", "errors.txt")

        if not os.path.exists(error_file_path):
            print(f"Warning: No errors.txt found for model {model_name}")
            continue

        # Read error.txt
        with open(error_file_path, "r") as f:
            # Skip header line
            header = f.readline()

            # Process each line
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    scene = parts[0]
                    frame = parts[1]

                    # Add model to list of models with error for this frame
                    error_frames[(scene, frame)].append(model_name)

    # Sort error frames by scene number and frame
    sorted_error_frames = sorted(
        error_frames.items(), key=lambda x: (get_scene_number(x[0][0]), int(x[0][1]))
    )

    # Write error_all.txt
    error_all_path = os.path.join(output_dir, "error_all.txt")
    with open(error_all_path, "w") as f:
        f.write("scene\tframe\tmodels\n")

        for (scene, frame), models in sorted_error_frames:
            models_str = ", ".join(models)
            f.write(f"{scene}\t{frame}\t{models_str}\n")

    print(f"Generated error_all.txt with {len(sorted_error_frames)} error frames")

    # Return set of error frames for filtering
    return {(scene, frame) for (scene, frame), _ in sorted_error_frames}


def evaluate_model_with_error_filtering(model_path, error_frames_set, model_name, is_baseline=False):
    """
    Evaluate a model by excluding error frames.

    Args:
        model_path (str): Path to model results directory
        error_frames_set (set): Set of (scene, frame) tuples representing error frames
        model_name (str): Name of the model
        is_baseline (bool): Whether this is the baseline model (different JSON structure)

    Returns:
        dict: Dictionary containing filtered evaluation results
    """
    # Load original evaluation
    eval_path = os.path.join(model_path, "analysis", "evaluation.json")
    if not os.path.exists(eval_path):
        print(f"Warning: No evaluation.json found for model {model_name}")
        return None

    eval_data = load_json_file(eval_path)

    # Re-process scene files with error frames filtered out
    output_dir = os.path.join(model_path, "output")
    scene_files = glob.glob(os.path.join(output_dir, "*.json"))
    
    if not scene_files:
        print(f"No scene JSON files found in {output_dir}")
        return None

    # Initialize metrics
    metrics_lists = {
        "L2_1s": [],
        "L2_2s": [],
        "L2_3s": [],
        "ADE": [],
        "FDE": [],
        "missRate_2": [],
    }
    
    # Initialize filtered evaluation
    eval_filtered = {"overall": {}, "per_scene": OrderedDict()}
    
    # Token and time usage (only for VLM models)
    if not is_baseline:
        token_usage = {
            "scene_prompt": {"input": 0, "output": 0},
            "intent_prompt": {"input": 0, "output": 0},
            "waypoint_prompt": {"input": 0, "output": 0},
            "total": {"input": 0, "output": 0},
        }
        
        time_usage = {
            "scene_prompt": 0,
            "intent_prompt": 0,
            "waypoint_prompt": 0,
            "total": 0,
        }
    
    # Track statistics
    total_frames = 0
    successful_frames = 0
    
    # Process each scene file
    for scene_file in scene_files:
        scene_name = os.path.basename(scene_file).split(".")[0]
        
        # Load scene data
        scene_data = load_json_file(scene_file)
        
        if not scene_data or "frames" not in scene_data or not scene_data["frames"]:
            continue
            
        # Sort frames
        scene_data["frames"] = sorted(scene_data["frames"], key=lambda x: x.get("frame_index", float("inf")))
        
        # Initialize for this scene
        scene_metrics_lists = {k: [] for k in metrics_lists}
        filtered_frame_results = []
        
        scene_frames = 0
        scene_successful = 0
        
        if not is_baseline:
            scene_token_usage = {
                "scene_prompt": {"input": 0, "output": 0},
                "intent_prompt": {"input": 0, "output": 0},
                "waypoint_prompt": {"input": 0, "output": 0},
                "total": {"input": 0, "output": 0},
            }
            
            scene_time_usage = {
                "scene_prompt": 0,
                "intent_prompt": 0,
                "waypoint_prompt": 0,
                "total": 0,
            }
        
        # Process each frame
        for frame in scene_data["frames"]:
            frame_idx = str(frame.get("frame_index", ""))
            
            # Skip if frame is in error frames set
            if (scene_name, frame_idx) in error_frames_set:
                continue
                
            # Count frames
            total_frames += 1
            scene_frames += 1
            
            # Create frame result
            frame_result = {
                "frame_index": frame["frame_index"],
                "sample_token": frame["sample_token"]
            }
            
            # Check if predictions exist
            if "predictions" in frame and "trajectory" in frame["predictions"]:
                successful_frames += 1
                scene_successful += 1
                
                # Get ground truth and prediction
                gt_trajectory = frame["ego_info"]["gt_positions"]
                pred_trajectory = frame["predictions"]["trajectory"]
                
                # Recalculate metrics
                metrics = compute_metrics(pred_trajectory, gt_trajectory)
                
                # Store metrics for this frame
                frame_result["metrics"] = metrics
                
                # Add to aggregates
                for metric_name in metrics_lists:
                    if metrics[metric_name] is not None:
                        metrics_lists[metric_name].append(metrics[metric_name])
                        scene_metrics_lists[metric_name].append(metrics[metric_name])
                
                # Process token and time usage for VLM models
                if not is_baseline:
                    if "token_usage" in frame:
                        frame_result["token_usage"] = frame["token_usage"]
                        
                        # Accumulate for scene
                        for prompt_type, counts in frame["token_usage"].items():
                            scene_token_usage[prompt_type]["input"] += counts["input"]
                            scene_token_usage[prompt_type]["output"] += counts["output"]
                            scene_token_usage["total"]["input"] += counts["input"]
                            scene_token_usage["total"]["output"] += counts["output"]
                            
                            # Accumulate for overall
                            token_usage[prompt_type]["input"] += counts["input"]
                            token_usage[prompt_type]["output"] += counts["output"]
                            token_usage["total"]["input"] += counts["input"]
                            token_usage["total"]["output"] += counts["output"]
                    
                    if "time_usage" in frame:
                        frame_result["time_usage"] = frame["time_usage"]
                        
                        # Accumulate for scene
                        for prompt_type, time_val in frame["time_usage"].items():
                            scene_time_usage[prompt_type] += time_val
                            scene_time_usage["total"] += time_val
                            
                            # Accumulate for overall
                            time_usage[prompt_type] += time_val
                            time_usage["total"] += time_val
            
            # Add to frame results
            filtered_frame_results.append(frame_result)
        
        # Add scene to filtered evaluation if it has successful frames
        if scene_successful > 0:
            # Calculate scene-level metrics
            scene_metrics = {
                k: np.mean(v).item() if v else None
                for k, v in scene_metrics_lists.items()
            }
            
            # Create scene entry
            scene_entry = {
                "frames_total": scene_frames,
                "frames_successful": scene_successful,
                "success_rate": scene_successful / scene_frames if scene_frames > 0 else 0,
                "metrics": scene_metrics,
                "frame_results": filtered_frame_results
            }
            
            # Add token and time usage for VLM models
            if not is_baseline:
                scene_entry["token_usage"] = scene_token_usage
                scene_entry["time_usage"] = scene_time_usage
            
            # Add to per-scene results
            eval_filtered["per_scene"][scene_name] = scene_entry
    
    # Calculate overall metrics
    overall_metrics = {
        k: np.mean(v).item() if v else None for k, v in metrics_lists.items()
    }
    
    # Create overall entry
    eval_filtered["overall"] = {
        "frames_total": total_frames,
        "frames_successful": successful_frames,
        "success_rate": successful_frames / total_frames if total_frames > 0 else 0,
        "metrics": overall_metrics
    }
    
    # Add token and time usage for VLM models
    if not is_baseline:
        eval_filtered["overall"]["token_usage"] = token_usage
        eval_filtered["overall"]["time_usage"] = time_usage
    
    return eval_filtered


def create_comparison_grid(model_dirs, scene, frame, output_path):
    """
    Create a visual comparison grid for a specific scene and frame.

    Args:
        model_dirs (dict): Dictionary of {model_name: model_path}
        scene (str): Scene name
        frame (int): Frame index
        output_path (str): Path to save the comparison grid

    Returns:
        bool: Success status
    """
    # Get model names and sort them
    model_names = list(model_dirs.keys())
    
    # Define the grid layout based on number of models
    n_models = len(model_names)
    if n_models <= 3:
        rows, cols = 1, n_models
    elif n_models <= 6:
        rows, cols = 2, 3
    elif n_models <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 3  # Maximum 12 models in a 4x3 grid
    
    # Check if we have images for the models
    available_images = []
    for model_name in model_names:
        if model_name != "baseline":  # Skip baseline in visualization
            model_path = model_dirs[model_name]
            img_path = os.path.join(model_path, "frame", f"{scene}_frame{frame}.png")
            if os.path.exists(img_path):
                available_images.append((model_name, img_path))
    
    if not available_images:
        print(f"Error: No images available for {scene}_frame{frame}")
        return False
    
    # Get sample image dimensions
    sample_img = cv2.imread(available_images[0][1])
    img_height, img_width = sample_img.shape[:2]
    
    # Create a blank canvas for the grid
    grid_width = img_width * cols
    grid_height = img_height * rows
    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255  # White background
    
    # Place images in the grid
    for i, (model_name, img_path) in enumerate(available_images):
        row = i // cols
        col = i % cols
        
        # Check if we've exceeded grid dimensions
        if row >= rows or col >= cols:
            break
        
        # Read and place the image
        img = cv2.imread(img_path)
        y_start = row * img_height
        x_start = col * img_width
        
        # Make sure we don't go out of bounds
        if y_start + img_height <= grid_height and x_start + img_width <= grid_width:
            # Place the image in the grid
            grid_img[y_start : y_start + img_height, x_start : x_start + img_width] = img
            
            # Add model name as text overlay
            # Convert to PIL for better text rendering
            pil_img = Image.fromarray(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # Try to get a nice font
            try:
                # Find a font that exists on the system
                font_path = font_manager.findfont(
                    font_manager.FontProperties(family="DejaVu Sans")
                )
                font = ImageFont.truetype(font_path, 36)
            except:
                # Fall back to default
                font = ImageFont.load_default()
            
            # Draw model name at the top left of its section
            text_x = x_start + 10
            text_y = y_start + 10
            draw.text((text_x, text_y), model_name, font=font, fill=(255, 0, 0))
            
            # Convert back to OpenCV format
            grid_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Save the grid image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, grid_img)
    print(f"Saved comparison grid to {output_path}")
    return True


def get_scene_frames(model_dirs):
    """
    Get all scenes and their frames available across all models.

    Args:
        model_dirs (dict): Dictionary of {model_name: model_path}

    Returns:
        dict: Dictionary of {scene_name: [frame_indices]}
    """
    scenes = {}
    
    # Check all models to get available scenes and frames
    for model_name, model_path in model_dirs.items():
        if model_name == "baseline":
            continue  # Skip baseline in visualization
            
        frame_dir = os.path.join(model_path, "frame")
        
        if os.path.exists(frame_dir):
            # Get all scene names from visualization filenames
            for file in os.listdir(frame_dir):
                if file.endswith(".png"):
                    # Extract scene name and frame number
                    match = re.match(r"(.+)_frame(\d+)\.png", file)
                    if match:
                        scene_name, frame_num = match.groups()
                        if scene_name not in scenes:
                            scenes[scene_name] = set()
                        scenes[scene_name].add(int(frame_num))
    
    # Convert sets to sorted lists
    for scene in scenes:
        scenes[scene] = sorted(list(scenes[scene]))
    
    return scenes


def generate_comparison_visualizations(model_dirs, output_dir):
    """
    Generate comparison visualizations for all models.

    Args:
        model_dirs (dict): Dictionary of {model_name: model_path}
        output_dir (str): Directory to save comparison visualizations
    """
    # Get available scenes and frames
    scenes = get_scene_frames(model_dirs)
    if not scenes:
        print("Error: No visualization files found")
        return
    
    # Create comparison grids for each scene and frame
    total_frames = sum(len(frames) for frames in scenes.values())
    processed_frames = 0
    
    for scene, frames in scenes.items():
        scene_output_dir = os.path.join(output_dir, scene)
        os.makedirs(scene_output_dir, exist_ok=True)
        
        for frame in frames:
            output_path = os.path.join(scene_output_dir, f"comparison_frame{frame}.png")
            success = create_comparison_grid(model_dirs, scene, frame, output_path)
            
            processed_frames += 1
            print(f"Processed {processed_frames}/{total_frames} frames")
    
    print(f"Complete! Comparison visualizations saved to {output_dir}")


def find_model_dirs(results_dir):
    """
    Find all model directories including VLMs and baseline.
    
    Args:
        results_dir (str): Base directory containing model results
        
    Returns:
        dict: Dictionary of {model_name: model_path}
        str: Path to baseline directory
    """
    model_dirs = {}
    baseline_dir = None
    
    # Get all immediate subdirectories
    subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    for d in subdirs:
        full_path = os.path.join(results_dir, d)
        
        # Check for required subdirectories
        has_output = os.path.exists(os.path.join(full_path, "output"))
        has_analysis = os.path.exists(os.path.join(full_path, "analysis"))
        
        if has_output and has_analysis:
            if d.lower() == "baseline":
                baseline_dir = full_path
            else:
                # Extract model name (remove timestamp if present)
                model_name = d.split('_')[0] if '_' in d else d
                model_dirs[model_name] = full_path
    
    return model_dirs, baseline_dir


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get results directory from config
    results_dir = config["data"]["results"]
    
    # Find all model directories
    model_dirs, baseline_dir = find_model_dirs(results_dir)
    
    if not model_dirs:
        print("Error: No model directories found")
        return
        
    print(f"Found {len(model_dirs)} model directories:")
    for model_name, path in model_dirs.items():
        print(f"  - {model_name}: {path}")
        
    if baseline_dir:
        print(f"Found baseline directory: {baseline_dir}")
    
    # Create output directory for merged frames
    frames_merge_dir = os.path.join(results_dir, "frames_merge")
    os.makedirs(frames_merge_dir, exist_ok=True)
    
    # Generate error_all.txt for all models including baseline
    error_frames_set = generate_error_all(model_dirs, baseline_dir, results_dir)
    
    # Process baseline evaluation if available
    if baseline_dir:
        print(f"Processing baseline evaluation...")
        baseline_evaluation = evaluate_model_with_error_filtering(
            baseline_dir, error_frames_set, "baseline", is_baseline=True
        )
        
        if baseline_evaluation:
            # Save filtered evaluation
            eval_path = os.path.join(baseline_dir, "analysis", "evaluation_all.json")
            save_dict_to_json(baseline_evaluation, eval_path)
            print(f"Saved filtered baseline evaluation to {eval_path}")
    
    # Process each VLM model
    for model_name, model_path in model_dirs.items():
        print(f"Processing {model_name} evaluation...")
        
        # Evaluate with error filtering
        vlm_evaluation = evaluate_model_with_error_filtering(
            model_path, error_frames_set, model_name, is_baseline=False
        )
        
        if vlm_evaluation:
            # Save filtered evaluation
            eval_path = os.path.join(model_path, "analysis", "evaluation_all.json")
            save_dict_to_json(vlm_evaluation, eval_path)
            print(f"Saved filtered evaluation for {model_name} to {eval_path}")
    
    # Generate comparison visualizations if not disabled
    if not args.no_vis:
        # Create model_dirs dictionary without baseline for visualization
        viz_model_dirs = {k: v for k, v in model_dirs.items()}
        generate_comparison_visualizations(viz_model_dirs, frames_merge_dir)
    
    print("All evaluation tasks completed successfully")


if __name__ == "__main__":
    main()