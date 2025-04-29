import os
import numpy as np
import argparse
from collections import OrderedDict
from nuscenes import NuScenes

from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="LightEMMA: Baseline Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration file (default: config.yaml)")
    parser.add_argument("--no_vis", action="store_true",
                        help="Disable visualization generation")
    return parser.parse_args()

def constant_extrapolation(obs_actions):
    """
    Simple extrapolation by copying the last observed action
    
    Args:
        obs_actions: List of (speed, curvature) tuples for the observed window
        
    Returns:
        list: Predicted actions for the next 6 timesteps
    """
    last_action = obs_actions[-1]
    return [last_action] * 6

def evaluate_baseline(config, generate_vis=True):
    """
    Evaluate constant extrapolation baseline
    
    Args:
        config: Configuration dictionary
        generate_vis: Whether to generate visualizations
        
    Returns:
        dict: Dictionary containing evaluation results
    """
    # Extract paths from config
    data_root = config["data"]["root"]
    data_version = config["data"]["version"]
    results_dir = os.path.join(config["data"]["results"], "baseline")
    
    print(f"Evaluating constant extrapolation baseline, saving to {results_dir}")
    
    # Create output directory structure
    output_dir = os.path.join(results_dir, "output")
    frame_dir = os.path.join(results_dir, "frame")
    analysis_dir = os.path.join(results_dir, "analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    if generate_vis:
        os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Initialize NuScenes dataset
    nusc = NuScenes(version=data_version, dataroot=data_root, verbose=True)
    
    # Initialize metrics
    total_frames = 0
    successful_frames = 0
    
    # Metrics overall
    all_l2_1s = []
    all_l2_2s = []
    all_l2_3s = []
    all_ade = []
    all_fde = []
    all_miss_rates = []
    
    # Metrics per scene
    scene_metrics = {}
    
    # Process each scene in NuScenes
    for scene in nusc.scene:
        scene_name = scene["name"]
        print(f"Processing scene {scene_name}")
        
        first_sample_token = scene["first_sample_token"]
        last_sample_token = scene["last_sample_token"]
        description = scene["description"]
        
        # Initialize baseline scene data structure
        baseline_scene_data = {
            "scene_info": {
                "name": scene_name,
                "description": description,
                "first_sample_token": first_sample_token,
                "last_sample_token": last_sample_token
            },
            "frames": [],
            "metadata": {
                "model": "constant_extrapolation_baseline",
                "timestamp": "",
                "total_frames": 0
            }
        }
        
        # Collect scene data
        camera_params = []
        front_camera_images = []
        ego_positions = []
        ego_headings = []
        timestamps = []
        sample_tokens = []
        
        curr_sample_token = first_sample_token
        
        # Retrieve all frames in the scene
        while curr_sample_token:
            sample = nusc.get("sample", curr_sample_token)
            sample_tokens.append(curr_sample_token)
            
            # Get camera data
            cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            front_camera_images.append(
                os.path.join(nusc.dataroot, cam_front_data["filename"])
            )
            
            # Get the camera parameters
            camera_params.append(
                nusc.get("calibrated_sensor", cam_front_data["calibrated_sensor_token"])
            )
            
            # Get ego vehicle state
            ego_state = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
            ego_positions.append(tuple(ego_state["translation"][0:2]))
            ego_headings.append(quaternion_to_yaw(ego_state["rotation"]))
            timestamps.append(ego_state["timestamp"])
            
            # Move to next sample or exit loop if at the end
            curr_sample_token = (
                sample["next"] if curr_sample_token != last_sample_token else None
            )
        
        num_frames = len(front_camera_images)
        
        # Determine obs_len and fut_len from config
        OBS_LEN = config["prediction"]["obs_len"]
        FUT_LEN = config["prediction"]["fut_len"]
        EXT_LEN = config["prediction"]["ext_len"]
        TTL_LEN = OBS_LEN + FUT_LEN + EXT_LEN
        
        # Check if we have enough frames
        if num_frames < TTL_LEN:
            print(f"Skipping '{scene_name}', insufficient frames ({num_frames} < {TTL_LEN}).")
            continue
        
        # Initialize metrics for this scene
        scene_l2_1s = []
        scene_l2_2s = []
        scene_l2_3s = []
        scene_ade = []
        scene_fde = []
        scene_miss_rates = []
        
        scene_frames = 0
        scene_successful = 0
        
        # Initialize frame results for this scene
        frame_results = []
        
        # Process each frame in the scene
        for i in range(0, num_frames - TTL_LEN, 1):
            try:
                cur_index = i + OBS_LEN + 1
                frame_index = i  # The relative index in the processed subset
                
                # Get current position and heading
                cur_pos = ego_positions[cur_index]
                cur_heading = ego_headings[cur_index]
                
                # Get observation data (past positions and timestamps)
                obs_pos = ego_positions[cur_index - OBS_LEN - 1 : cur_index + 1]
                obs_pos_ego = global_to_ego_frame(cur_pos, cur_heading, obs_pos)
                obs_time = timestamps[cur_index - OBS_LEN - 1 : cur_index + 1]
                
                # Calculate past speeds and curvatures
                prev_speed = compute_speed(obs_pos_ego, obs_time)
                prev_curvatures = compute_curvature(obs_pos_ego)
                prev_actions = list(zip(prev_speed, prev_curvatures))
                
                # Get future positions and timestamps (ground truth)
                fut_pos = ego_positions[cur_index - 1 : cur_index + FUT_LEN + 1]
                fut_pos_ego = global_to_ego_frame(cur_pos, cur_heading, fut_pos)
                fut_time = timestamps[cur_index - 1 : cur_index + FUT_LEN + 1]
                
                # Calculate ground truth speeds and curvatures
                gt_speed = compute_speed(fut_pos_ego, fut_time)
                gt_curvatures = compute_curvature(fut_pos_ego)
                gt_actions = list(zip(gt_speed, gt_curvatures))
                
                # Remove extra indices used for speed and curvature calculation
                fut_pos_ego = fut_pos_ego[2:]
                
                # Generate predicted actions using constant extrapolation
                pred_actions = constant_extrapolation(prev_actions)
                
                # Convert actions to trajectory
                pred_trajectory = integrate_driving_commands(pred_actions, dt=0.5)
                
                # Calculate metrics
                metrics = compute_metrics(pred_trajectory, fut_pos_ego)
                
                # Store metrics
                all_l2_1s.append(metrics["L2_1s"])
                all_l2_2s.append(metrics["L2_2s"])
                all_l2_3s.append(metrics["L2_3s"])
                all_ade.append(metrics["ADE"])
                all_fde.append(metrics["FDE"])
                all_miss_rates.append(metrics["missRate_2"])
                
                scene_l2_1s.append(metrics["L2_1s"])
                scene_l2_2s.append(metrics["L2_2s"])
                scene_l2_3s.append(metrics["L2_3s"])
                scene_ade.append(metrics["ADE"])
                scene_fde.append(metrics["FDE"])
                scene_miss_rates.append(metrics["missRate_2"])
                
                # Create baseline frame data
                frame_data = {
                    "frame_index": frame_index,
                    "sample_token": sample_tokens[cur_index],
                    "image_path": front_camera_images[cur_index],
                    "timestamp": timestamps[cur_index],
                    "camera_params": {
                        "rotation": camera_params[cur_index]["rotation"],
                        "translation": camera_params[cur_index]["translation"],
                        "camera_intrinsic": camera_params[cur_index]["camera_intrinsic"]
                    },
                    "ego_info": {
                        "position": cur_pos,
                        "heading": cur_heading,
                        "obs_positions": obs_pos_ego,
                        "obs_actions": prev_actions,
                        "gt_positions": fut_pos_ego,
                        "gt_actions": gt_actions
                    },
                    "predictions": {
                        "pred_actions": pred_actions,
                        "trajectory": pred_trajectory
                    },
                }
                
                # Add frame to scene data
                baseline_scene_data["frames"].append(frame_data)
                
                # Create visualization if requested
                if generate_vis:
                    image_path = front_camera_images[cur_index]
                    
                    # Visualization filename
                    viz_filename = f"{scene_name}_frame{frame_index}.png"
                    viz_path = os.path.join(frame_dir, viz_filename)
                    
                    # Create visualization
                    OverlayTrajectory(
                        img_path=image_path,
                        wp_world1=fut_pos_ego,  # Ground truth
                        wp_world2=pred_trajectory,  # Prediction
                        cam_to_ego=frame_data["camera_params"],
                        ego_pos=(0, 0),
                        ego_heading=0.0,
                        save_path=viz_path
                    )
                
                # Update counters
                total_frames += 1
                scene_frames += 1
                successful_frames += 1
                scene_successful += 1
                
                # Add to frame results
                frame_result = {
                    "frame_index": frame_index,
                    "sample_token": sample_tokens[cur_index],
                    "metrics": metrics
                }
                frame_results.append(frame_result)
                
            except Exception as e:
                print(f"Error processing frame {i} in scene {scene_name}: {e}")
                continue
        
        # Update scene metadata
        baseline_scene_data["metadata"]["total_frames"] = len(baseline_scene_data["frames"])
        
        # Save scene data if any frames were processed
        if scene_successful > 0:
            scene_file_path = os.path.join(output_dir, f"{scene_name}.json")
            save_dict_to_json(baseline_scene_data, scene_file_path)
            
            # Calculate scene-level metrics
            scene_metrics[scene_name] = {
                "frames_total": scene_frames,
                "frames_successful": scene_successful,
                "success_rate": scene_successful / scene_frames,
                "metrics": {
                    "L2_1s": np.mean(scene_l2_1s).item() if scene_l2_1s else None,
                    "L2_2s": np.mean(scene_l2_2s).item() if scene_l2_2s else None,
                    "L2_3s": np.mean(scene_l2_3s).item() if scene_l2_3s else None,
                    "ADE": np.mean(scene_ade).item() if scene_ade else None,
                    "FDE": np.mean(scene_fde).item() if scene_fde else None,
                    "missRate_2": np.mean(scene_miss_rates).item() if scene_miss_rates else None
                },
                "frame_results": sorted(frame_results, key=lambda x: x.get("frame_index", float('inf')))
            }
    
    # Calculate overall metrics
    overall_metrics = {
        "frames_total": total_frames,
        "frames_successful": successful_frames,
        "success_rate": successful_frames / total_frames if total_frames > 0 else 0,
        "metrics": {
            "L2_1s": np.mean(all_l2_1s).item() if all_l2_1s else None,
            "L2_2s": np.mean(all_l2_2s).item() if all_l2_2s else None,
            "L2_3s": np.mean(all_l2_3s).item() if all_l2_3s else None,
            "ADE": np.mean(all_ade).item() if all_ade else None,
            "FDE": np.mean(all_fde).item() if all_fde else None,
            "missRate_2": np.mean(all_miss_rates).item() if all_miss_rates else None
        }
    }
    
    # Create evaluation results dictionary
    evaluation_results = {
        "method": "constant_extrapolation_baseline",
        "overall": overall_metrics,
        "per_scene": OrderedDict(sorted(scene_metrics.items(), key=lambda x: x[0]))
    }
    
    # Save evaluation results
    evaluation_file = os.path.join(analysis_dir, "evaluation.json")
    save_dict_to_json(evaluation_results, evaluation_file)
    print(f"Baseline evaluation results saved to {evaluation_file}")
    
    # Print overall summary
    print("\nOverall Baseline Metrics:")
    print(f"Total frames: {overall_metrics['frames_total']}")
    print(f"Successful frames: {overall_metrics['frames_successful']}")
    print(f"Success rate: {overall_metrics['success_rate']*100:.2f}%")
    print(f"L2 1s avg: {overall_metrics['metrics']['L2_1s']:.4f}")
    print(f"L2 2s avg: {overall_metrics['metrics']['L2_2s']:.4f}")
    print(f"L2 3s avg: {overall_metrics['metrics']['L2_3s']:.4f}")
    print(f"ADE avg: {overall_metrics['metrics']['ADE']:.4f}")
    print(f"FDE avg: {overall_metrics['metrics']['FDE']:.4f}")
    print(f"missRate_2 avg: {overall_metrics['metrics']['missRate_2']:.4f}")
    
    return evaluation_results


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Evaluate baseline
    evaluate_baseline(config, generate_vis=not args.no_vis)

if __name__ == "__main__":
    main()