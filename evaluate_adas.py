"""
VinFast ADAS - Pedestrian Collision Warning (PCW)
Evaluation & Benchmark Script (Đánh giá & Xuất Log Kiểm chứng)
Hackathon Capstone - Track 4 (Computer Vision & Robotics)
"""

import cv2
import numpy as np
import time
import os
import sys
import json
import csv
import argparse
from ultralytics import YOLO

class Kalman1DDistance:
    """1D Kalman Filter to track distance and estimate relative velocity (v_rel = dd/dt)."""
    def __init__(self, init_d=3.0, process_noise=0.1, measurement_noise=0.25):
        self.d = init_d
        self.v = 0.0
        self.last_t = time.time()
        self.P = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float)
        self.Q = np.array([[process_noise, 0.0], [0.0, process_noise * 2]], dtype=float)
        self.R = float(measurement_noise)

    def update(self, measured_d):
        now = time.time()
        dt = max(1e-3, now - self.last_t)
        self.last_t = now

        # Prediction step
        pred_d = self.d + self.v * dt
        pred_v = self.v
        
        P00 = self.P[0,0] + dt * (self.P[1,0] + self.P[0,1]) + dt*dt * self.P[1,1] + self.Q[0,0]
        P01 = self.P[0,1] + dt * self.P[1,1] + self.Q[0,1]
        P10 = self.P[1,0] + dt * self.P[1,1] + self.Q[1,0]
        P11 = self.P[1,1] + self.Q[1,1]

        # Update step
        y = measured_d - pred_d
        S = P00 + self.R
        K0 = P00 / S
        K1 = P10 / S

        self.d = pred_d + K0 * y
        self.v = pred_v + K1 * y

        self.P[0,0] = (1 - K0) * P00
        self.P[0,1] = (1 - K0) * P01
        self.P[1,0] = -K1 * P00 + P10
        self.P[1,1] = -K1 * P01 + P11

        return self.d, self.v

class ADASSafetyEvaluator:
    def __init__(self, source="0", focal_length=680.0, real_height=1.70, car_speed_kmh=45.0, headless=False):
        self.source = int(source) if str(source).isdigit() else str(source)
        self.focal_length = focal_length
        self.real_height = real_height
        self.car_speed_kmh = car_speed_kmh
        self.v_car_mps = (car_speed_kmh * 1000.0) / 3600.0
        self.headless = headless
        
        print(f"[EVALUATION INIT] Loading YOLOv8n model...")
        self.model = YOLO('yolov8n.pt')
        # Warmup model to avoid 1000ms JIT latency spike on frame 1
        _dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model.predict(_dummy, imgsz=320, verbose=False)
        print("[EVALUATION INIT] Model warmup complete.")
        self.trackers = {}

        # Log storage
        self.frame_records = []
        self.infer_times = []
        self.total_times = []
        self.raw_distances = []
        self.kalman_distances = []
        self.alert_counts = {"SAFE": 0, "WARNING": 0, "CRITICAL": 0, "NONE": 0}
        self.corridor_hits = 0
        self.ttc_list = []

    def estimate_distance(self, bbox_height_px):
        if bbox_height_px <= 5:
            return 99.0
        return (self.focal_length * self.real_height) / bbox_height_px

    def check_in_corridor(self, foot_x, foot_y, w, h):
        top_y = int(h * 0.48)
        top_w = int(w * 0.26)
        bot_y = h
        bot_w = int(w * 0.70)
        cx = w // 2
        top_left = cx - top_w // 2
        top_right = cx + top_w // 2
        bot_left = cx - bot_w // 2
        bot_right = cx + bot_w // 2

        if foot_y < top_y or foot_y > bot_y:
            return False
        ratio = (foot_y - top_y) / (bot_y - top_y)
        cur_left = top_left + ratio * (bot_left - top_left)
        cur_right = top_right + ratio * (bot_right - top_right)
        return cur_left <= foot_x <= cur_right

    def run_evaluation(self):
        print(f"[EVALUATION] Starting video source: {self.source}")
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {self.source}")
            return None

        total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(self.source, str) else -1
        fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0

        os.makedirs("recordings", exist_ok=True)
        csv_file_path = "recordings/eval_telemetry.csv"
        csv_fp = open(csv_file_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow([
            "frame_idx", "timestamp_s", "latency_infer_ms", "latency_total_ms", 
            "ped_count", "min_raw_dist_m", "min_kalman_dist_m", "kalman_v_rel_mps", 
            "min_ttc_s", "in_corridor", "threat_level"
        ])

        frame_idx = 0
        t_eval_start = time.time()
        print(f"[EVALUATION] Processing frames... Press 'q' to stop early if running in GUI mode.\n")

        while True:
            t_frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            h, w, _ = frame.shape

            # 1. Inference Latency (with imgsz=320 for real-time CPU speed)
            t0 = time.time()
            results = self.model.predict(frame, classes=[0], imgsz=320, verbose=False, conf=0.45)
            t1 = time.time()
            infer_ms = (t1 - t0) * 1000.0
            self.infer_times.append(infer_ms)

            # 2. Extract Pedestrian Telemetry
            boxes = results[0].boxes
            ped_count = len(boxes)
            peds = []

            for i, box in enumerate(boxes):
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = coords
                box_h = y2 - y1
                ped_cx = (x1 + x2) // 2
                ped_foot_y = y2

                raw_d = self.estimate_distance(box_h)
                self.raw_distances.append(raw_d)

                if i not in self.trackers:
                    self.trackers[i] = Kalman1DDistance(init_d=raw_d)
                smooth_d, v_rel = self.trackers[i].update(raw_d)
                self.kalman_distances.append(smooth_d)

                closing_speed = max(0.5, self.v_car_mps + (-v_rel))
                ttc = smooth_d / closing_speed
                in_corr = self.check_in_corridor(ped_cx, ped_foot_y, w, h)

                if in_corr and (smooth_d <= 2.2 or ttc <= 1.8):
                    threat = "CRITICAL"
                elif in_corr and (smooth_d <= 4.0 or ttc <= 3.0):
                    threat = "WARNING"
                else:
                    threat = "SAFE"

                peds.append({
                    "raw_d": raw_d,
                    "smooth_d": smooth_d,
                    "v_rel": v_rel,
                    "ttc": ttc,
                    "in_corr": in_corr,
                    "threat": threat,
                    "conf": conf
                })

            # Frame level summary
            if peds:
                crit_ped = min(peds, key=lambda p: (not p['in_corr'], p['ttc']))
                frame_threat = crit_ped["threat"]
                min_raw = crit_ped["raw_d"]
                min_kalman = crit_ped["smooth_d"]
                v_rel = crit_ped["v_rel"]
                min_ttc = crit_ped["ttc"]
                in_corr_flag = crit_ped["in_corr"]
                if in_corr_flag:
                    self.corridor_hits += 1
                self.ttc_list.append(min_ttc)
            else:
                frame_threat = "NONE"
                min_raw = 99.0
                min_kalman = 99.0
                v_rel = 0.0
                min_ttc = 99.0
                in_corr_flag = False

            self.alert_counts[frame_threat] += 1

            t_frame_end = time.time()
            total_frame_ms = (t_frame_end - t_frame_start) * 1000.0
            self.total_times.append(total_frame_ms)

            # Write to CSV
            csv_writer.writerow([
                frame_idx, round(t_frame_end - t_eval_start, 3), round(infer_ms, 2), round(total_frame_ms, 2),
                ped_count, round(min_raw, 2), round(min_kalman, 2), round(v_rel, 2),
                round(min_ttc, 2), in_corr_flag, frame_threat
            ])

            # Optional display
            if not self.headless:
                cv2.putText(frame, f"EVAL FRAME #{frame_idx} | INFER: {infer_ms:.1f}ms | THREAT: {frame_threat}", 
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)
                cv2.imshow("VinFast ADAS Evaluation Stream", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[EVALUATION] User interrupted evaluation.")
                    break

            if frame_idx % 30 == 0:
                print(f"  -> Processed {frame_idx} frames... (Current Threat: {frame_threat}, Infer: {infer_ms:.1f}ms)")

        cap.release()
        csv_fp.close()
        if not self.headless:
            cv2.destroyAllWindows()

        t_eval_total = time.time() - t_eval_start
        print(f"\n[EVALUATION FINISHED] Processed {frame_idx} frames in {t_eval_total:.2f}s.")
        
        # Calculate Aggregated Metrics
        return self.compute_and_save_summary(frame_idx, t_eval_total)

    def compute_and_save_summary(self, total_frames, total_time_s):
        if total_frames == 0 or not self.infer_times:
            print("[ERROR] No frames evaluated.")
            return None

        avg_infer = float(np.mean(self.infer_times))
        p95_infer = float(np.percentile(self.infer_times, 95))
        p99_infer = float(np.percentile(self.infer_times, 99))
        
        avg_total_ms = float(np.mean(self.total_times))
        effective_fps = float(total_frames / max(1e-4, total_time_s))

        # Variance reduction (Kalman smoothness vs raw noise)
        if len(self.raw_distances) > 5:
            raw_var = float(np.var(self.raw_distances))
            kalman_var = float(np.var(self.kalman_distances))
            var_reduction_pct = max(0.0, (1.0 - kalman_var / max(1e-4, raw_var)) * 100.0)
        else:
            var_reduction_pct = 0.0

        min_ttc_observed = float(np.min(self.ttc_list)) if self.ttc_list else 99.0
        avg_ttc_observed = float(np.mean(self.ttc_list)) if self.ttc_list else 99.0

        summary = {
            "evaluation_metadata": {
                "source": str(self.source),
                "total_frames_processed": total_frames,
                "duration_seconds": round(total_time_s, 2),
                "simulated_car_speed_kmh": self.car_speed_kmh
            },
            "system_performance_metrics": {
                "mean_fps": round(effective_fps, 2),
                "mean_inference_latency_ms": round(avg_infer, 2),
                "p95_inference_latency_ms": round(p95_infer, 2),
                "p99_inference_latency_ms": round(p99_infer, 2),
                "mean_pipeline_latency_ms": round(avg_total_ms, 2)
            },
            "perception_and_geometry_metrics": {
                "total_detections": len(self.raw_distances),
                "kalman_noise_variance_reduction_pct": round(var_reduction_pct, 2),
                "corridor_intrusion_frames": self.corridor_hits,
                "corridor_intrusion_ratio_pct": round((self.corridor_hits / total_frames) * 100.0, 2)
            },
            "adas_safety_threat_metrics": {
                "frames_safe_pct": round((self.alert_counts["SAFE"] / total_frames) * 100.0, 2),
                "frames_warning_pct": round((self.alert_counts["WARNING"] / total_frames) * 100.0, 2),
                "frames_critical_pct": round((self.alert_counts["CRITICAL"] / total_frames) * 100.0, 2),
                "frames_no_pedestrian_pct": round((self.alert_counts["NONE"] / total_frames) * 100.0, 2),
                "min_ttc_observed_seconds": round(min_ttc_observed, 2),
                "avg_ttc_observed_seconds": round(avg_ttc_observed, 2),
                "ncap_safety_compliance": "PASS" if min_ttc_observed >= 1.4 else "WARNING_BRAKE_LATE"
            }
        }

        # Save JSON summary
        json_path = "recordings/eval_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        # Print Markdown / Terminal Report
        self.print_report(summary, json_path, "recordings/eval_telemetry.csv")
        return summary

    def print_report(self, summary, json_file, csv_file):
        m = summary["system_performance_metrics"]
        g = summary["perception_and_geometry_metrics"]
        s = summary["adas_safety_threat_metrics"]
        meta = summary["evaluation_metadata"]

        print("\n" + "=" * 70)
        print("          VINFAST ADAS - PCW PERCEPTION BENCHMARK REPORT         ")
        print("         Hackathon Capstone Track 4 (Evidence & Evaluation)      ")
        print("=" * 70)
        print(f" Source Video / Camera   : {meta['source']}")
        print(f" Total Frames Evaluated  : {meta['total_frames_processed']} frames ({meta['duration_seconds']}s)")
        print(f" Vehicle Speed Setting   : {meta['simulated_car_speed_kmh']} km/h")
        print("-" * 70)
        print(" [1] HIỆU NĂNG THỜI GIAN THỰC (REAL-TIME PERFORMANCE)")
        print(f"  * Average Frame Rate    : {m['mean_fps']} FPS (Target: >= 30 FPS)")
        print(f"  * Mean Inference Latency: {m['mean_inference_latency_ms']} ms")
        print(f"  * P95 Inference Latency : {m['p95_inference_latency_ms']} ms (Trễ phân vị 95%)")
        print(f"  * P99 Inference Latency : {m['p99_inference_latency_ms']} ms")
        print(f"  * Total Pipeline Latency: {m['mean_pipeline_latency_ms']} ms / frame")
        print("-" * 70)
        print(" [2] HÌNH HỌC & BẤT ĐỊNH (GEOMETRY & KALMAN UNCERTAINTY)")
        print(f"  * Số phát hiện đối tượng: {g['total_detections']} detections")
        print(f"  * Độ mượt Kalman Filter : Giảm {g['kalman_noise_variance_reduction_pct']}% phương sai nhiễu đo")
        print(f"  * Xâm phạm hành lang lái: {g['corridor_intrusion_frames']} frames ({g['corridor_intrusion_ratio_pct']}%)")
        print("-" * 70)
        print(" [3] CHỈ SỐ AN TOÀN ADAS & TIÊU CHUẨN NCAP (THREAT & SAFETY)")
        print(f"  * Tỉ lệ KHÔNG CÓ người  : {s['frames_no_pedestrian_pct']}%")
        print(f"  * Tỉ lệ AN TOÀN (SAFE)  : {s['frames_safe_pct']}%")
        print(f"  * CẢNH BÁO SỚM (WARNING): {s['frames_warning_pct']}%")
        print(f"  * PHANH KHẨN CẤP (CRIT) : {s['frames_critical_pct']}%")
        print(f"  * Thời gian va chạm nhỏ : Min TTC = {s['min_ttc_observed_seconds']}s (Chuẩn an toàn >= 1.4s)")
        print(f"  * Đánh giá chuẩn NCAP   : {s['ncap_safety_compliance']}")
        print("=" * 70)
        print(f" [VẬN HÀNH THÀNH CÔNG] Dữ liệu log chi tiết đã lưu vào:")
        print(f"  -> File CSV Frame-by-frame: {csv_file}")
        print(f"  -> File JSON Tổng hợp     : {json_file}")
        print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VinFast ADAS Evaluation and Benchmarking")
    parser.add_argument("--source", type=str, default="recordings/demo_backup.mp4", 
                        help="Video file path or camera index (0, 1) to evaluate")
    parser.add_argument("--focal", type=float, default=680.0, help="Camera focal length (default: 680)")
    parser.add_argument("--speed", type=float, default=45.0, help="Simulated vehicle speed in km/h (default: 45)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI popup window for fast benchmarking")
    args = parser.parse_args()

    evaluator = ADASSafetyEvaluator(
        source=args.source,
        focal_length=args.focal,
        car_speed_kmh=args.speed,
        headless=args.headless
    )
    evaluator.run_evaluation()
