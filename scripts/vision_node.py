#!/usr/bin/env python3

import rospy
import cv2
import tkinter as tk

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PIL import Image as PILImage
from PIL import ImageTk


NODE_NAME = "vision_node"


class WebcamROSNode:
    def __init__(self, window):
        self.window = window
        self.window.title("Webcam Stream")

        # ROS setup
        rospy.init_node(NODE_NAME)

        self.bridge = CvBridge()

        # Publisher
        self.image_pub = rospy.Publisher(
            '/chef_robot_assistant/image_raw',
            Image,
            queue_size=10
        )

        # Open webcam.
        # Explicitly force the Video4Linux2 backend so it bypasses GStreamer entirely.
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        # Force MJPG compression to prevent WSL USB bandwidth issues
        # At a high level, this line translates to:
        # "Hey camera, please stop sending me massive, uncompressed raw images, and
        # instead compress each frame as a JPEG before sending it over the USB cable."
        #
        # More explanations:
        # - cv2.CAP_PROP_FOURCC (The Property ID): This tells OpenCV what property we are modifying.
        #   FOURCC stands for Four-Character Code.
        # - cv2.VideoWriter_fourcc(*'MJPG'): This is the new format we are applying.
        #   OpenCV cannot just accept the string "MJPG";
        #   it requires a specific 32-bit integer that represents those four letters.
        #   - cv2.VideoWriter_fourcc() is a helper function that converts four individual characters into that exact 32-bit integer.
        #   - *'MJPG' uses Python's unpacking operator (the *). It takes the string 'MJPG' and unpacks it into four separate arguments.
        #     So, cv2.VideoWriter_fourcc(*'MJPG') is just a cleaner way of writing cv2.VideoWriter_fourcc('M', 'J', 'P', 'G').
        #
        # Why did this fix your WSL issue?
        # By default, most webcams default to a format called YUYV (a type of raw, uncompressed video).
        #
        # Uncompressed video requires a massive amount of USB bandwidth. A standard Windows machine can usually handle this fine.
        # However, you are running WSL.
        # Because WSL is a virtual machine, the USB connection is being artificially bridged through software (via usbipd).
        # This virtual bridge creates a bottleneck.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Set a safe, standard resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # -----------------------

        if not self.cap.isOpened():
            rospy.logerr("Cannot open webcam")
            raise RuntimeError("Cannot open webcam")

        # GUI elements
        self.label = tk.Label(window)
        self.label.pack()

        self.btn_snapshot = tk.Button(
            window,
            text="Take Snapshot",
            command=self.take_snapshot
        )
        self.btn_snapshot.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_quit = tk.Button(
            window,
            text="Quit",
            command=self.close
        )
        self.btn_quit.pack(side=tk.RIGHT, padx=10, pady=10)

        # Start update loop
        self.update_frame()

    def update_frame(self):
        if rospy.is_shutdown():
            self.close()
            return

        ret, frame = self.cap.read()

        if ret:
            # Publish ROS image
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.image_pub.publish(ros_image)

            # Convert for Tkinter display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = PILImage.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        # Refresh every 10 ms
        self.window.after(10, self.update_frame)

    def take_snapshot(self):
        ret, frame = self.cap.read()

        if ret:
            filename = "snapshot.jpg"
            cv2.imwrite(filename, frame)
            rospy.loginfo(f"Saved {filename}")

    def close(self):
        rospy.loginfo(f"Shutting down {NODE_NAME}")

        if self.cap.isOpened():
            self.cap.release()

        self.window.destroy()



def main():
    print("1. Initializing Tkinter...")
    root = tk.Tk()

    print("2. Starting WebcamROSNode...")
    app = WebcamROSNode(root)

    print("3. Entering mainloop (Window should appear)...")
    try:
        root.mainloop()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
