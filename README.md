# Smart Queue Monitor (Wait-Time-Detection)

![Python](https://img.shields.io/badge/Language-Python-blue.svg)
![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-yellow.svg)
![DeepSORT](https://img.shields.io/badge/Tracking-DeepSORT-lightgrey.svg)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green.svg)

## 1. Description
The **Smart Queue Monitor** is a real-time computer vision system engineered to monitor queues, track individuals, and calculate predictive wait times. Designed to optimize crowd management and operational efficiency, the system analyzes live video feeds to extract actionable analytics, moving beyond basic object counting to full service-duration tracking.

## 2. Key Features
*   **Real-Time Object Detection:** Implements YOLOv8 for high-accuracy human detection, optimized to perform consistently across varied lighting and crowd densities.
*   **Robust Subject Tracking:** Utilizes the DeepSORT algorithm to assign and maintain unique persistent IDs for individuals, effectively handling occlusions and overlapping paths.
*   **Dynamic ROI Configuration:** Allows for customizable Regions of Interest (ROI) to isolate specific queue and service boundaries, filtering out background movement and reducing false positives.
*   **Live Analytics Output:** Computes and renders real-time metrics directly onto the video feed, including active queue length and average service duration.

## 3. Technology Stack
*   **Language:** Python
*   **Detection Architecture:** YOLOv8 (You Only Look Once)
*   **Tracking Algorithm:** DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric)
*   **Computer Vision Framework:** OpenCV

## 4. Algorithmic Logic
The system utilizes live extracted data to predict the estimated wait time for new arrivals. By continually averaging the actual service duration of previous individuals, the system calculates future wait times using the following logic:

> **Estimated Wait Time = Queue Length × Average Service Time**

## 5. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Piyush-Sambyall/Wait-Time-Detection.git](https://github.com/Piyush-Sambyall/Wait-Time-Detection.git)
