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
        self.source = 0   # 0 laptop, 1 USB, or "http://IP:PORT/video"
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

        self.cap = self.init_camera()

        self.entry_times = {}
        self.wait_times = {}

        self.total_served = 0
        self.frame_count = 0

        self.id_map = {}
        self.next_id = 1

    # =============================================================================
    def init_camera(self):
        src = self.cfg.source

        if isinstance(src, str):
            cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src, cv2.CAP_MSMF)
            if not cap.isOpened():
                cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            raise RuntimeError("Camera not opening")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap

    # =============================================================================
    def draw_info(self, frame, queue, wait, status, status_color, service):
        x, y = 10, 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (270, 170), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        def put(label, value, yy, color):
            cv2.putText(frame, f"{label}: {value}", (x+12, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        yy = y + 28
        step = 24

        put("Queue", queue, yy, (0,120,255)); yy += step
        put("Wait", f"{int(wait)}s", yy, (255,180,0)); yy += step
        put("Status", status, yy, status_color); yy += step
        put("Service", service, yy, (0,255,255)); yy += step
        put("Served", self.total_served, yy, (0,255,0)); yy += step

        if self.cfg.avg_service_time:
            put("Avg", f"{self.cfg.avg_service_time:.1f}s", yy, (200,200,0))

    # =============================================================================
    def fix_roi(self, roi):
        roi[0] = max(0, roi[0])
        roi[1] = max(0, roi[1])
        roi[2] = max(roi[0]+10, roi[2])
        roi[3] = max(roi[1]+10, roi[3])
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
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))

            tracks = self.tracker.update_tracks(detections, frame=frame)

            queue_count = 0
            service_count = 0
            now = time.time()

            for t in tracks:
                if not t.is_confirmed() or t.time_since_update > 1:
                    continue

                track_id = t.track_id

                if track_id not in self.id_map:
                    self.id_map[track_id] = self.next_id
                    self.next_id += 1

                display_id = self.id_map[track_id]

                x1, y1, x2, y2 = map(int, t.to_ltrb())
                cx, cy = (x1+x2)//2, (y1+y2)//2

                q, s = self.cfg.queue_roi, self.cfg.service_roi

                zone = ""
                color = (0,255,0)

                if q[0]<cx<q[2] and q[1]<cy<q[3]:
                    zone = "Q"
                    color = (255,100,0)
                    queue_count += 1

                    if track_id not in self.entry_times:
                        self.entry_times[track_id] = now
                        self.wait_times[track_id] = 0

                elif s[0]<cx<s[2] and s[1]<cy<s[3]:
                    zone = "S"
                    color = (0,255,255)
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
                            if track_id in self.wait_times:
                                del self.wait_times[track_id]

                # WAIT TIME
                if track_id in self.entry_times:
                    self.wait_times[track_id] = now - self.entry_times[track_id]

                wait_t = int(self.wait_times.get(track_id, 0))

                # ESTIMATION
                est_t = 0
                if self.cfg.avg_service_time and zone == "Q":
                    est_t = int(wait_t + self.cfg.avg_service_time * queue_count)

                # DRAW BOX
                cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)

                # LABEL
                label = f"P{display_id} | W:{wait_t}s | E:{est_t}s"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                cv2.rectangle(frame, (x1,y1-30),(x1+tw+10,y1), color, -1)

                cv2.putText(frame, label, (x1+5,y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,0,0), 2, cv2.LINE_AA)

                cv2.circle(frame, (cx,cy), 4, color, -1)

            # GLOBAL WAIT
            wait = 0
            if self.cfg.avg_service_time and queue_count > 0:
                wait = queue_count * self.cfg.avg_service_time / max(service_count,1)

            # STATUS
            if queue_count > 8:
                status, color = "HIGH", (0,0,255)
            elif queue_count > 4:
                status, color = "MEDIUM", (0,200,255)
            else:
                status, color = "LOW", (0,255,0)

            cv2.rectangle(frame, tuple(self.cfg.queue_roi[:2]), tuple(self.cfg.queue_roi[2:]), (255,0,0),2)
            cv2.rectangle(frame, tuple(self.cfg.service_roi[:2]), tuple(self.cfg.service_roi[2:]), (0,255,255),2)

            self.draw_info(frame, queue_count, wait, status, color, service_count)

            key = cv2.waitKeyEx(1)

            if key == 27:
                break

            cv2.imshow("Smart Queue Monitor", frame)

        self.cap.release()
        cv2.destroyAllWindows()


# =============================================================================
if __name__ == "__main__":
    cfg = Config()
    SmartQueueMonitor(cfg).run()