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

### Setup Instructions

1. Make sure the ROS workspace is installed in an environment matching the versions listed above.

   For the speech branch, install `alsa-utils` so the `arecord` microphone capture
   command is available on Ubuntu.
   For the recipe branch, add `GEMINI_API_KEY` to your local `.env`.

2. Create a ROS workspace if you haven't already. The folder name is up to you, but the recommended name is `catkin_ws`, following the standard convention:

   ```bash
   mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```

3. Navigate to your workspace's `src` directory and clone this repository:

   ```bash
   cd ~/catkin_ws/src
   git clone <repository-url>
   ```

4. Build the package:

   ```bash
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```

5. Install Python dependencies listed in `requirements.txt`:

   ```bash
   pip3 install -r requirements.txt
   ```
