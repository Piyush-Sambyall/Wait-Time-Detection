import cv2
import time
import csv
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# =============================================================================
# CONFIG
# =============================================================================
class Config:
    def __init__(self):
        self.source = 0  # CAMERAS
        self.model_path = "yolov8n.pt"
        self.use_gpu = True
        self.yolo_conf = 0.45
        self.min_person_height = 80
        self.frame_skip = 2

        self.queue_roi = [50, 200, 400, 500]
        self.service_roi = [420, 200, 650, 450]

        self.deep_sort_max_age = 40
        self.avg_service_time = None
        self.alpha = 0.3

        self.log_csv = True


# =============================================================================
# MAIN CLASS
# =============================================================================
class SmartQueueMonitor:
    def __init__(self, cfg):
        self.cfg = cfg

        self.device = "cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu"
        print("[INFO] Using:", self.device)

        self.model = YOLO(cfg.model_path).to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()

        self.tracker = DeepSort(max_age=cfg.deep_sort_max_age)

        # CAMERA FIX
        self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            print("[WARNING] MSMF failed, switching...")
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError("[ERROR] Camera not opening")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.entry_times = {}
        self.total_served = 0
        self.frame_count = 0
        self.start_time = time.time()

        if cfg.log_csv:
            self.csv_file = open("queue_log.csv", "w", newline="")
            self.writer = csv.writer(self.csv_file)
            self.writer.writerow(["time", "queue", "service", "wait"])

        print("[INFO] Controls: Queue A/D W/S Z/X C/V | Service Arrows/IJKL | ESC")

    # =============================================================================
    # PROFESSIONAL INFO PANEL
    # =============================================================================
    def draw_info(self, frame, queue, wait, status, status_color, service):
        x, y = 10, 10
        box_w, box_h = 260, 170

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (200, 200, 200), 1)

        def put(label, value, yy, color):
            text = f"{label}: {value}"
            cv2.putText(frame, text, (x + 12, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        yy = y + 28
        step = 24

        put("Queue", queue, yy, (0, 120, 255)); yy += step
        put("Wait", f"{int(wait)} sec", yy, (255, 180, 0)); yy += step
        put("Status", status, yy, status_color); yy += step
        put("In Service", service, yy, (0, 255, 255)); yy += step
        put("Served", self.total_served, yy, (0, 255, 0)); yy += step

        if self.cfg.avg_service_time:
            put("Avg Time", f"{self.cfg.avg_service_time:.1f}s", yy, (200, 200, 0))

    # =============================================================================
    def fix_roi(self, roi):
        roi[0] = max(0, roi[0])
        roi[1] = max(0, roi[1])
        roi[2] = max(roi[0] + 10, roi[2])
        roi[3] = max(roi[1] + 10, roi[3])
        return roi

    # =============================================================================
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (960, 540))
            self.frame_count += 1

            if self.frame_count % self.cfg.frame_skip != 0:
                continue

            results = self.model(frame, imgsz=480, conf=self.cfg.yolo_conf, verbose=False)

            detections = []
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) != 0:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if (y2 - y1) < self.cfg.min_person_height:
                        continue

                    conf = float(box.conf[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

            tracks = self.tracker.update_tracks(detections, frame=frame)

            queue_count = 0
            service_count = 0
            now = time.time()

            for t in tracks:
                if not t.is_confirmed() or t.time_since_update > 1:
                    continue

                track_id = t.track_id
                x1, y1, x2, y2 = map(int, t.to_ltrb())

                cx, cy = (x1+x2)//2, (y1+y2)//2
                q, s = self.cfg.queue_roi, self.cfg.service_roi

                # ZONE DETECTION
                if q[0]<cx<q[2] and q[1]<cy<q[3]:
                    zone = "Q"
                    color = (255, 100, 0)
                    queue_count += 1

                    if track_id not in self.entry_times:
                        self.entry_times[track_id] = now

                elif s[0]<cx<s[2] and s[1]<cy<s[3]:
                    zone = "S"
                    color = (0, 255, 255)
                    service_count += 1

                    if track_id in self.entry_times:
                        st = now - self.entry_times[track_id]

                        if 2 < st < 120:
                            if self.cfg.avg_service_time is None:
                                self.cfg.avg_service_time = st
                            else:
                                self.cfg.avg_service_time = (
                                    self.cfg.alpha * st +
                                    (1 - self.cfg.alpha) * self.cfg.avg_service_time
                                )

                            self.total_served += 1
                            del self.entry_times[track_id]
                else:
                    zone = ""
                    color = (0, 255, 0)

                # DRAW BOX
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # ID LABEL (HIGH VISIBILITY)
                label = f"ID {track_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw + 8, y1), color, -1)

                cv2.putText(frame, label, (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 2, cv2.LINE_AA)

                # CENTER DOT
                cv2.circle(frame, (cx, cy), 4, color, -1)

            # WAIT TIME
            wait = 0
            if self.cfg.avg_service_time and queue_count > 0:
                wait = queue_count * self.cfg.avg_service_time / max(service_count, 1)

            # STATUS
            if queue_count > 8:
                status, status_color = "HIGH", (0, 0, 255)
            elif queue_count > 4:
                status, status_color = "MEDIUM", (0, 200, 255)
            else:
                status, status_color = "LOW", (0, 255, 0)

            # DRAW ROIs
            cv2.rectangle(frame, tuple(self.cfg.queue_roi[:2]),
                          tuple(self.cfg.queue_roi[2:]), (255,0,0), 2)

            cv2.rectangle(frame, tuple(self.cfg.service_roi[:2]),
                          tuple(self.cfg.service_roi[2:]), (0,255,255), 2)

            # DRAW UI
            self.draw_info(frame, queue_count, wait, status, status_color, service_count)

            cv2.putText(frame, "Queue: A/D W/S | Z/X/C/V resize",
                        (10,510), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

            cv2.putText(frame, "Service: Arrows or IJKL | ESC",
                        (10,530), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

            # KEYBOARD CONTROL
            key = cv2.waitKeyEx(1)

            # Queue
            if key == ord('a'): self.cfg.queue_roi[0]-=5; self.cfg.queue_roi[2]-=5
            if key == ord('d'): self.cfg.queue_roi[0]+=5; self.cfg.queue_roi[2]+=5
            if key == ord('w'): self.cfg.queue_roi[1]-=5; self.cfg.queue_roi[3]-=5
            if key == ord('s'): self.cfg.queue_roi[1]+=5; self.cfg.queue_roi[3]+=5
            if key == ord('z'): self.cfg.queue_roi[2]-=5
            if key == ord('x'): self.cfg.queue_roi[2]+=5
            if key == ord('c'): self.cfg.queue_roi[3]-=5
            if key == ord('v'): self.cfg.queue_roi[3]+=5

            # Service
            if key in [2490368, ord('i')]:
                self.cfg.service_roi[1]-=5; self.cfg.service_roi[3]-=5
            if key in [2621440, ord('k')]:
                self.cfg.service_roi[1]+=5; self.cfg.service_roi[3]+=5
            if key in [2424832, ord('j')]:
                self.cfg.service_roi[0]-=5; self.cfg.service_roi[2]-=5
            if key in [2555904, ord('l')]:
                self.cfg.service_roi[0]+=5; self.cfg.service_roi[2]+=5

            self.cfg.queue_roi = self.fix_roi(self.cfg.queue_roi)
            self.cfg.service_roi = self.fix_roi(self.cfg.service_roi)

            if key == 27:
                print("Queue ROI:", self.cfg.queue_roi)
                print("Service ROI:", self.cfg.service_roi)
                break

            cv2.imshow("Smart Queue Monitor", frame)

        self.cap.release()
        cv2.destroyAllWindows()
        if self.cfg.log_csv:
            self.csv_file.close()


# =============================================================================
if __name__ == "__main__":
    cfg = Config()
    SmartQueueMonitor(cfg).run()