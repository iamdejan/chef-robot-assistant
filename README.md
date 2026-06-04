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
   sudo apt update
   sudo apt install -y alsa-utils espeak ros-noetic-usb-cam
   ```

   `alsa-utils` is required so the `arecord` microphone capture command is available on Ubuntu.

2. Create a ROS workspace if you haven't already. The folder name is up to you, but the recommended name is `catkin_ws`, following the standard convention:

   ```bash
   mkdir -p ~/catkin_ws/src
   ```

3. Navigate to your workspace's `src` directory and clone this repository:

   ```bash
   cd ~/catkin_ws/src
   git clone https://github.com/iamdejan/chef-robot-assistant.git chef_robot_assistant
   ```

4. Install ROS dependencies, but only if ROS has never been installed before:

   ```bash
   cd ~/catkin_ws
   sudo apt install python3-rosdep
   sudo rosdep init # only run once after installing ROS.
   rosdep update --include-eol-distros # only run once after installing ROS.
   rosdep install --from-paths src --ignore-src -r -y
   ```

   If in doubt, do not initialize `rosdep`.


6. Install dependencies:

   ```bash
   cd ~/catkin_ws/src/chef_robot_assistant
   pip3 install -r requirements.txt
   ```

7. Create your local environment file from the template and fill in the required API keys:

   ```bash
   cp .env.example .env
   ```

   Required keys:
   - `GEMINI_API_KEY`
   - `HF_TOKEN`
   - `OPENAI_API_KEY`

   In the Juno robot, we already put the API keys which can be used for this project. Note that the API keys will be invalidated when Semester 2 ends.

8. Build the package:

   ```bash
   cd ~/catkin_ws
   catkin_make
   ```

### Running the System

After setup, run these commands in sequence:
```bash
cd ~
source /opt/ros/noetic/setup.bash
source $HOME/catkin_ws/devel/setup.bash
```

Then, run either:
```bash
roslaunch chef_robot_assistant foundation_with_usb_cam.launch
```
or
```bash
roslaunch chef_robot_assistant foundation.launch
```

### How to Use

1. Place at least `2` supported ingredients in front of the camera.
2. Wait for the robot to detect and announce the ingredients.
3. Answer the questions from the robot. List of questions:
   a. Are you able to buy missing ingredients? (yes / no)
   b. Do you have health preferences, such as low fat, high protein, or low carbs? (low fat / high fat / low protein / high protein / low fiber / high fiber / low carbs / high carbs)
   c. Do you have any food allergies? (yes / no)
   d. Do you have a specific dish in mind? (just say dish name without "I want")
   e. I am generating a recipe for you. Do you want Malay, Western or Chinese food? (Malay / Chinese / Western)
4. Recipe will be generated, and the recipe summary will be spoken by the robot.
5. Answer the question from the robot: "Do you want me to generate an image of the dish?" (yes / no)
   a. If the image of the dish is successfully generated, the robot will say "Here is an example image of the dish. Thank you for using me."
6. The program is finished. The robot will ask: "Do you want me to start again?" (yes / no)

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
