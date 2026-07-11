import cv2
import time
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict


# =============================================================================
class Config:
    def __init__(self):
        self.source = 0
        self.model_path = "yolov8n.pt"
        self.use_gpu = True

        self.yolo_conf = 0.5
        self.min_person_height = 80
        self.roi_step = 15

        self.queue_roi = [50, 200, 400, 500]
        self.service_roi = [420, 200, 650, 450]

        self.deep_sort_max_age = 60
        self.avg_service_time = None
        self.alpha = 0.3


# =============================================================================
class SmartQueueMonitor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu"

        self.model = YOLO(cfg.model_path).to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()

        # ✅ Improved tracking stability
        self.tracker = DeepSort(max_age=cfg.deep_sort_max_age, n_init=3)

        self.cap = cv2.VideoCapture(cfg.source)

        self.entry_times = {}
        self.service_times = {}
        self.wait_times = {}

        self.total_served = 0
        self.id_map = {}
        self.next_id = 1
        self.track_state = {}

        self.smooth_boxes = defaultdict(list)

    # ================= TIME FORMAT =================
    def format_time_label(self, seconds):
        if seconds < 60:
            return f"{int(seconds)}s"
        else:
            return f"{round(seconds/60,1)}m"

    def format_time_hms(self, seconds):
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ================= SMOOTHING =================
    def smooth(self, tid, box):
        self.smooth_boxes[tid].append(box)
        if len(self.smooth_boxes[tid]) > 5:
            self.smooth_boxes[tid].pop(0)
        avg = [int(sum(x)/len(x)) for x in zip(*self.smooth_boxes[tid])]
        return avg

    # =============================================================================
    def draw_info(self, frame, queue, wait, status, status_color, service):
        x, y = 10, 10
        current_time = time.strftime("%H:%M:%S")

        lines = [
            ("Time", current_time, (255,255,255)),
            ("Queue", queue, (200,100,0)),
            ("Wait", self.format_time_hms(wait), (255,200,0)),
            ("Status", status, status_color),
            ("Service", service, (0,255,255)),
            ("Served", self.total_served, (0,255,0))
        ]

        if self.cfg.avg_service_time:
            avg = self.cfg.avg_service_time
            if avg < 60:
                avg_text = f"{int(avg)}s"
            else:
                m = int(avg // 60)
                s = int(avg % 60)
                avg_text = f"{m:02d}:{s:02d}"

            lines.append(("Avg Service", avg_text, (200,200,0)))

        max_width = 0
        for text, val, _ in lines:
            (tw, _), _ = cv2.getTextSize(f"{text}: {val}",
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            max_width = max(max_width, tw)

        w = max_width + 40
        h = 20 + len(lines)*25

        overlay = frame.copy()
        cv2.rectangle(overlay, (x,y), (x+w, y+h), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        yy = y + 25
        for text, val, color in lines:
            cv2.putText(frame, f"{text}: {val}",
                        (x+10, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            yy += 25

    # =============================================================================
    def run(self):

        LEFT = [81, 2424832, 65361]
        RIGHT = [83, 2555904, 65363]
        UP = [82, 2490368, 65362]
        DOWN = [84, 2621440, 65364]

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (960,540))
            key = cv2.waitKeyEx(1)

            if key == 27:
                break

            # Queue controls
            if key == ord('a'):
                self.cfg.queue_roi[0] -= 15; self.cfg.queue_roi[2] -= 15
            elif key == ord('d'):
                self.cfg.queue_roi[0] += 15; self.cfg.queue_roi[2] += 15
            elif key == ord('w'):
                self.cfg.queue_roi[1] -= 15; self.cfg.queue_roi[3] -= 15
            elif key == ord('s'):
                self.cfg.queue_roi[1] += 15; self.cfg.queue_roi[3] += 15
            elif key == ord('z'):
                self.cfg.queue_roi[2] -= 30
            elif key == ord('x'):
                self.cfg.queue_roi[2] += 30
            elif key == ord('c'):
                self.cfg.queue_roi[3] -= 30
            elif key == ord('v'):
                self.cfg.queue_roi[3] += 30

            # Service controls
            elif key in LEFT:
                self.cfg.service_roi[0] -= 15; self.cfg.service_roi[2] -= 15
            elif key in RIGHT:
                self.cfg.service_roi[0] += 15; self.cfg.service_roi[2] += 15
            elif key in UP:
                self.cfg.service_roi[1] -= 15; self.cfg.service_roi[3] -= 15
            elif key in DOWN:
                self.cfg.service_roi[1] += 15; self.cfg.service_roi[3] += 15
            elif key == ord('i'):
                self.cfg.service_roi[3] += 30
            elif key == ord('k'):
                self.cfg.service_roi[3] -= 30
            elif key == ord('j'):
                self.cfg.service_roi[2] -= 30
            elif key == ord('l'):
                self.cfg.service_roi[2] += 30

            results = self.model(frame, conf=self.cfg.yolo_conf, verbose=False)

            detections = []
            for r in results:
                for b in r.boxes:
                    if int(b.cls[0]) != 0:
                        continue

                    x1,y1,x2,y2 = map(int,b.xyxy[0])

                    # ✅ detection filtering (fix multiple detection issue)
                    conf = float(b.conf[0])
                    if conf < 0.5:
                        continue

                    if (y2 - y1) < self.cfg.min_person_height:
                        continue

                    detections.append(([x1,y1,x2-x1,y2-y1], conf, 'person'))

            tracks = self.tracker.update_tracks(detections, frame=frame)

            queue_count = 0
            service_count = 0
            now = time.time()

            for t in tracks:
                if not t.is_confirmed():
                    continue

                tid = t.track_id

                if tid not in self.id_map:
                    self.id_map[tid] = self.next_id
                    self.next_id += 1

                pid = self.id_map[tid]

                x1,y1,x2,y2 = self.smooth(tid, list(map(int,t.to_ltrb())))
                cx,cy = (x1+x2)//2,(y1+y2)//2

                in_q = self.cfg.queue_roi[0]<cx<self.cfg.queue_roi[2] and self.cfg.queue_roi[1]<cy<self.cfg.queue_roi[3]
                in_s = self.cfg.service_roi[0]<cx<self.cfg.service_roi[2] and self.cfg.service_roi[1]<cy<self.cfg.service_roi[3]

                if tid not in self.track_state:
                    self.track_state[tid] = "new"

                if in_q and self.track_state[tid]=="new":
                    self.entry_times[tid] = now
                    self.track_state[tid]="queue"

                if in_s and self.track_state[tid]=="queue":
                    self.wait_times[tid] = now - self.entry_times.get(tid, now)
                    self.service_times[tid] = now
                    self.track_state[tid]="service"

                #  prevent duplicate service updates
                if self.track_state[tid]=="service" and not in_s:
                    if tid in self.service_times:
                        st = now - self.service_times.get(tid, now)

                        if 3 < st < 120:
                            if self.cfg.avg_service_time is None:
                                self.cfg.avg_service_time = st
                            else:
                                self.cfg.avg_service_time = self.cfg.alpha*st + (1-self.cfg.alpha)*self.cfg.avg_service_time

                            self.total_served += 1

                        self.service_times.pop(tid, None)

                    self.track_state[tid]="done"

                if in_q: queue_count+=1
                if in_s: service_count+=1

                # ✅ real-time wait update
                if self.track_state.get(tid) == "queue":
                    wait_t = now - self.entry_times.get(tid, now)
                elif self.track_state.get(tid) == "service":
                    wait_t = self.wait_times.get(tid, 0)
                else:
                    wait_t = 0

                label = f"ID:{pid} W:{self.format_time_label(wait_t)}"

                if in_q:
                    color = (255,150,100)
                elif in_s:
                    color = (100,100,255)
                else:
                    color = (0,255,0)

                overlay = frame.copy()
                cv2.rectangle(overlay,(x1,y1-30),(x1+180,y1),(0,0,0),-1)
                cv2.addWeighted(overlay,0.7,frame,0.3,0,frame)

                cv2.putText(frame,label,(x1+5,y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)

            wait_global = queue_count*(self.cfg.avg_service_time or 5)

            if queue_count>8:
                status,color_s="HIGH",(0,0,255)
            elif queue_count>4:
                status,color_s="MEDIUM",(0,200,255)
            else:
                status,color_s="LOW",(0,255,0)

            cv2.rectangle(frame,tuple(self.cfg.queue_roi[:2]),tuple(self.cfg.queue_roi[2:]),(139,0,0),5)
            cv2.rectangle(frame,tuple(self.cfg.service_roi[:2]),tuple(self.cfg.service_roi[2:]),(0,0,139),5)

            self.draw_info(frame, queue_count, wait_global, status, color_s, service_count)

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,500),(960,540),(0,0,0),-1)
            cv2.addWeighted(overlay,0.6,frame,0.4,0,frame)

            cv2.putText(frame,
                "Queue: WASD | Z/X width | C/V height   ||   Service: Arrows | I/K height | J/L width",
                (10,525), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            cv2.imshow("Smart Queue Monitor", frame)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = Config()
    SmartQueueMonitor(cfg).run()