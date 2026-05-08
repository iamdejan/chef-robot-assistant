#!/usr/bin/env python3

import rospy
import cv2
import tkinter as tk

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PIL import Image as PILImage
from PIL import ImageTk


class WebcamROSNode:
    def __init__(self, window):

        self.window = window
        self.window.title("ROS Webcam Stream")

        # ROS setup
        rospy.init_node('webcam_gui_node', anonymous=True)

        self.bridge = CvBridge()

        # Publisher
        self.image_pub = rospy.Publisher(
            '/webcam/image_raw',
            Image,
            queue_size=10
        )

        # Open webcam
        self.cap = cv2.VideoCapture(0)

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

        rospy.loginfo("Shutting down webcam node")

        if self.cap.isOpened():
            self.cap.release()

        self.window.destroy()


if __name__ == '__main__':

    root = tk.Tk()

    app = WebcamROSNode(root)

    try:
        root.mainloop()
    except rospy.ROSInterruptException:
        pass
