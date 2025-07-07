import glob
import os
import shutil

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from IPython.display import HTML, clear_output, display


def cleanup_videos(video_folder, video_prefix=None):
    """
    Clean up existing video files before starting new training.

    Args:
        video_folder: Path to video folder
        video_prefix: Optional prefix to filter which videos to delete (None deletes all)
    """
    try:
        if not os.path.exists(video_folder):
            print(f"Video folder {video_folder} doesn't exist, nothing to clean up.")
            return

        if video_prefix:
            # Delete only videos with specific prefix
            pattern = os.path.join(video_folder, f"{video_prefix}-episode-*.mp4")
            video_files = glob.glob(pattern)
            if video_files:
                print(
                    f"🗑️ Cleaning up {len(video_files)} existing videos with prefix '{video_prefix}'..."
                )
                for video_file in video_files:
                    os.remove(video_file)
                    print(f"   Deleted: {os.path.basename(video_file)}")
        else:
            # Delete entire folder and recreate
            if os.listdir(video_folder):  # Check if folder has any content
                print(f"🗑️ Cleaning up entire video folder: {video_folder}")
                shutil.rmtree(video_folder)
                os.makedirs(video_folder, exist_ok=True)
                print(f"   Video folder cleaned and recreated.")
            else:
                print(f"Video folder {video_folder} is already empty.")

    except Exception as e:
        print(f"Warning: Error cleaning up videos: {e}")


def create_env_with_wrappers(
    config,
    is_continuous,
    record_videos=False,
    video_prefix="training",
    cleanup_existing=True,
):
    """Create environment with standard wrappers."""
    env = gym.make(
        config["env_id"],
        continuous=is_continuous,
        render_mode="rgb_array",
        **config["env_kwargs"],
    )

    # Add video recording wrapper for training
    if record_videos:
        # Clean up existing videos first
        if cleanup_existing:
            cleanup_videos(config["video_folder"], video_prefix)

        # Ensure video folder exists
        os.makedirs(config["video_folder"], exist_ok=True)

        # Get video recording interval from config
        video_record_interval = config.get("video_record_interval")

        # Custom episode trigger that uses actual episode numbers
        def episode_trigger(episode_id):
            # episode_id starts from 0, so we add 1 to match our episode numbering
            actual_episode = episode_id + 1
            
            return actual_episode % video_record_interval == 0

        env = RecordVideo(
            env,
            video_folder=config["video_folder"],
            name_prefix=video_prefix,
            episode_trigger=episode_trigger,
        )

    # Add episode statistics tracking
    env = RecordEpisodeStatistics(env)

    return env


def display_videos_grid(video_folder, video_prefix, max_videos=None):
    """
    Display all videos from the training session in a horizontal grid with infinite looping.

    Args:
        video_folder: Path to video folder
        video_prefix: Prefix used for video files
        max_videos: Maximum number of videos to display (None for all)
    """
    try:
        # Find all video files matching the pattern
        pattern = os.path.join(video_folder, f"{video_prefix}-episode-*.mp4")
        video_files = glob.glob(pattern)

        if not video_files:
            print(f"No videos found in {video_folder} with prefix {video_prefix}")
            return

        # Sort video files by episode number
        import re

        def extract_episode_num(filename):
            match = re.search(r"episode-(\d+)\.mp4", filename)
            return int(match.group(1)) if match else 0

        video_files.sort(key=extract_episode_num)

        # Limit number of videos if specified
        if max_videos:
            video_files = video_files[-max_videos:]  # Show most recent videos

        # Convert absolute paths to relative paths for web display
        relative_videos = []
        for video_file in video_files:
            if os.path.isabs(video_file):
                relative_videos.append(os.path.relpath(video_file))
            else:
                relative_videos.append(video_file)

        # Extract episode numbers for display
        episode_nums = [extract_episode_num(os.path.basename(v)) for v in video_files]

        print(
            f"\n📹 Displaying {len(relative_videos)} training videos (episodes: {episode_nums}):"
        )

        # Create HTML for horizontal video grid with infinite looping
        videos_per_row = min(3, len(relative_videos))  # Max 3 videos per row
        video_width = max(300, 900 // videos_per_row)  # Responsive width

        html_content = """
        <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">
        """

        for i, (video_path, episode_num) in enumerate(
            zip(relative_videos, episode_nums)
        ):
            html_content += f"""
            <div style="text-align: center;">
                <p style="margin: 5px 0; font-weight: bold;">Episode {episode_num}</p>
                <video width="{video_width}" height="{int(video_width * 0.75)}" controls loop autoplay muted>
                    <source src="{video_path}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            """

            # Start new row after every 3 videos
            if (i + 1) % 3 == 0 and i < len(relative_videos) - 1:
                html_content += """
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin-top: 10px;">
                """

        html_content += """
        </div>
        <style>
            video {
                border: 2px solid #ccc;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
        </style>
        """

        display(HTML(html_content))

    except Exception as e:
        print(f"Error displaying videos: {e}")


def display_latest_video(video_folder, video_prefix, episode_num):
    """
    Display all videos collected so far in a grid format with infinite looping.
    This replaces the single video display to show training progression.

    Args:
        video_folder: Path to video folder
        video_prefix: Prefix used for video files
        episode_num: Current episode number (for reference)
    """
    # Clear previous output to avoid stacking
    clear_output(wait=True)

    # Display all videos collected so far
    display_videos_grid(
        video_folder, video_prefix, max_videos=10
    )  # Limit to last 10 videos

    # Print video info without interfering with tqdm
    try:
        pattern = os.path.join(video_folder, f"{video_prefix}-episode-*.mp4")
        video_files = glob.glob(pattern)
        print(f"📹 {len(video_files)} training videos available in {video_folder}")
    except Exception as e:
        print(f"Note: {e}")


def show_final_video_summary(video_folder, video_prefix, action_type):
    """
    Show a final summary of all training videos after training completion.

    Args:
        video_folder: Path to video folder
        video_prefix: Prefix used for video files
        action_type: "Discrete" or "Continuous" for labeling
    """
    print(f"\n{'='*60}")
    print(f"FINAL VIDEO SUMMARY - {action_type.upper()} TRAINING")
    print(f"{'='*60}")

    # Display all videos without limit
    display_videos_grid(video_folder, video_prefix, max_videos=None)
