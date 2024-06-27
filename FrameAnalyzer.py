import cv2
import time
import psutil
import GPUtil
import numpy as np

class FrameAnalyzer:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.cpu_usage = 0
        self.gpu_usage = 0
        self.num_humans = 0

    def update_frame_count(self):
        self.frame_count += 1

    def get_frame_rate(self):
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time
        return fps

    def update_human_count(self, num_humans):
        self.num_humans = num_humans

    def get_cpu_usage(self):
        self.cpu_usage = psutil.cpu_percent()
        return self.cpu_usage

    def get_gpu_usage(self):
        gpus = GPUtil.getGPUs()
        if gpus:
            self.gpu_usage = gpus[0].load * 100
        return self.gpu_usage

    def display_info(self, img):
        fps = self.get_frame_rate()
        cpu = self.get_cpu_usage()
        gpu = self.get_gpu_usage()

        info = [
            f"FPS: {fps:.2f}",
            f"Humans: {self.num_humans}",
            f"CPU: {cpu:.2f}%",
            f"GPU: {gpu:.2f}%"
        ]

        colors = [
            (255, 0, 0),  # Blue for FPS
            (0, 255, 0),  # Green for Humans
            (0, 0, 255),  # Red for CPU
            (255, 255, 0)  # Cyan for GPU
        ]

        for i, (text, color) in enumerate(zip(info, colors)):
            cv2.putText(img, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img
