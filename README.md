# Chef Robot Assistant

This is a group project for the **WQF7010 Robotics** subject in the **Universiti Malaya, Master of Artificial of Intelligence 2025/26** program.

The **Chef Robot Assistant** helps users by detecting food ingredients and then generating a recipe based on the detected items.

## Prerequisites

Before setting up this repository, ensure the following requirements are met:

| Component | Version |
|-----------|---------|
| ROS Distro | Noetic |
| Operating System | Ubuntu 20.04.x |
| Python | 3.8.x |
| Python package manager | uv |

### Setup Instructions

1. Install the required Ubuntu packages:

   ```bash
   sudo apt-get update
   sudo apt-get install -y alsa-utils espeak
   ```

   For the speech branch, install `alsa-utils` so the `arecord` microphone capture
   command is available on Ubuntu.
   For the recipe branch, add `GEMINI_API_KEY` to your local `.env`.
   For the output/image branch, install `espeak`, run inside an active desktop/X
   session so OpenCV windows can open, and add `HF_TOKEN` to your local `.env`.

2. Create a ROS workspace if you haven't already. The folder name is up to you, but the recommended name is `catkin_ws`, following the standard convention:

   ```bash
   mkdir -p ~/catkin_ws/src
   ```

3. Navigate to your workspace's `src` directory and clone this repository:

   ```bash
   cd ~/catkin_ws/src
   git clone <repository-url>
   ```

4. Install ROS dependencies:

   ```bash
   cd ~/catkin_ws
   sudo apt install python3-rosdep
   sudo rosdep init # only run once after installing ROS.
   rosdep update --include-eol-distros # only run once after installing ROS.
   rosdep install --from-paths src --ignore-src -r -y
   ```

5. Install `uv` if you do not already have it:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

6. Create the Python environment and sync dependencies:

   ```bash
   cd ~/catkin_ws/src/WQF7010\ Robotics
   uv sync
   source .venv/bin/activate
   ```

7. Create your local environment file from the template and fill in the required API keys:

   ```bash
   cp .env.example .env
   ```

   Required keys:
   - `ROBOFLOW_API_KEY`
   - `GEMINI_API_KEY`
   - `HF_TOKEN`

8. Build the package:

   ```bash
   cd ~/catkin_ws
   catkin_make
   ```

### Running the System

After setup:

```bash
cd ~/catkin_ws
source devel/setup.bash
source src/WQF7010\ Robotics/.venv/bin/activate
roslaunch chef_robot_assistant foundation.launch
```

### How to Use

1. Place at least `2` supported ingredients in front of the camera.
2. Wait for the robot to detect and announce the ingredients.
3. Say one cuisine clearly:
   - `malay`
   - `western`
   - `chinese`
4. When asked about image generation, say:
   - `yes`
   - `no`
5. After the cycle finishes, the robot may ask:
   - `Do you want me to start again?`
6. Say:
   - `yes` to start another cycle
   - `no` to leave the robot idle

### Stopping the System

To stop the app normally, press `Ctrl+C` in the terminal running `roslaunch`.

If the stack gets stuck, force stop it with:

```bash
pkill -f "foundation.launch|rosmaster|rosout|interaction_manager.py|vision_node.py|speech_node.py|recipe_node.py|media_node.py|webcam_node.py"
```

### Optional Cleanup

To remove generated dish images:

```bash
rm -f /tmp/chef_robot_assistant_generated_media/*
```
