import csv
import os
import time
import numpy as np
from datetime import datetime

class ExperimentLogger:
    def __init__(self, output_dir="experiment_logs", run_name="test_run"):
        """
        Initializes the logger, creating the directory and CSV files with headers.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # timestamp to keep logs unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Define File Paths
        self.det_log_path = os.path.join(self.output_dir, f"detection_log_{run_name}_{timestamp}.csv")
        self.summary_log_path = os.path.join(self.output_dir, f"benchmark_summary_{run_name}_{timestamp}.csv")

        # 2. Initialize Detection Log (Frame-by-Frame)
        self.det_header = [
            'video_id', 'frame_id', 'ground_truth', 'vibe_ratio', 
            'mhi_energy', 'cnn_conf', 'final_decision', 
            'inference_time_ms', 'trigger_stage'
        ]
        self._init_csv(self.det_log_path, self.det_header)

        # 3. Initialize Summary Log (Per Video)
        self.summary_header = [
            'video_id', 'total_frames', 'frames_processed_cnn', 
            'TP', 'FP', 'FN', 'TN',
            'Precision', 'Recall', 'F1_Score', 
            'Avg_FPS', 'Storage_Saved_Pct'
        ]
        self._init_csv(self.summary_log_path, self.summary_header)

        # Internal state for the current video
        self.current_video_stats = self._reset_stats()

    def _init_csv(self, path, header):
        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def _reset_stats(self):
        return {
            'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
            'processed_count': 0,
            'total_time_ms': 0.0,
            'frame_count': 0
        }

    def start_video(self, video_id):
        """Call this before starting a new video loop."""
        print(f"--- Starting Logging for: {video_id} ---")
        self.current_video_stats = self._reset_stats()
        self.current_video_id = video_id

    def log_frame(self, frame_id, ground_truth, vibe_ratio, mhi_energy, 
                  cnn_conf, final_decision, time_ms, trigger_stage):
        """
        Logs a single frame's details and updates running counters.
        
        Args:
            ground_truth (int): 1 if animal present, 0 if empty.
            final_decision (int): 1 if system saved the frame, 0 if discarded.
            trigger_stage (str): 'vibe', 'mhi', 'cnn', or 'none'
        """
        # 1. Write to Detailed Log
        with open(self.det_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_video_id, frame_id, ground_truth, 
                f"{vibe_ratio:.4f}", f"{mhi_energy:.4f}", f"{cnn_conf:.4f}", 
                final_decision, f"{time_ms:.2f}", trigger_stage
            ])

        # 2. Update Running Stats (Confusion Matrix)
        s = self.current_video_stats
        s['frame_count'] += 1
        s['total_time_ms'] += time_ms
        
        # Only count as "processed" if it wasn't rejected early by VIBE
        if trigger_stage != 'none': 
            s['processed_count'] += 1

        # Calculate TP/FP/FN/TN
        if ground_truth == 1 and final_decision == 1:
            s['tp'] += 1
        elif ground_truth == 0 and final_decision == 1:
            s['fp'] += 1
        elif ground_truth == 1 and final_decision == 0:
            s['fn'] += 1
        elif ground_truth == 0 and final_decision == 0:
            s['tn'] += 1

    def end_video(self):
        """
        Computes final metrics for the video and writes to summary log.
        """
        s = self.current_video_stats
        
        # Calculate Metrics
        total = s['tp'] + s['fp'] + s['fn'] + s['tn']
        saved_frames = s['tp'] + s['fp']
        
        # Safe Division
        precision = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 0
        recall = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_fps = (s['frame_count'] / (s['total_time_ms'] / 1000)) if s['total_time_ms'] > 0 else 0
        storage_saved = (1 - (saved_frames / total)) * 100 if total > 0 else 0

        # Write to Summary Log
        with open(self.summary_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_video_id,
                total,
                s['processed_count'],
                s['tp'], s['fp'], s['fn'], s['tn'],
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                f"{avg_fps:.2f}",
                f"{storage_saved:.2f}"
            ])
            
        print(f"Finished {self.current_video_id}. F1: {f1:.3f} | Recall: {recall:.3f} | FPS: {avg_fps:.1f}")
        return self.summary_log_path

# ==========================================
#  Example Usage (Run this file to test it)
# ==========================================
if __name__ == "__main__":
    # 1. Setup Logger
    logger = ExperimentLogger(run_name="vibe_mhi_cnn_test")

    # 2. Simulate Video Processing
    video_name = "test_deer_night"
    logger.start_video(video_name)
    
    # Simulate 100 frames
    import random
    for i in range(100):
        # Fake Inference Data
        gt = 1 if i > 20 and i < 60 else 0  # Animal present frames 20-60
        vibe_score = random.random()
        cnn_score = random.random()
        
        # Fake Logic: If Vibe > 0.2 and CNN > 0.8, save it
        is_animal = 1 if (vibe_score > 0.2 and cnn_score > 0.8) else 0
        stage = "cnn_confirm" if is_animal else "vibe_reject"
        
        # LOG THE FRAME
        logger.log_frame(
            frame_id=i,
            ground_truth=gt,
            vibe_ratio=vibe_score,
            mhi_energy=random.random(), # dummy mhi
            cnn_conf=cnn_score,
            final_decision=is_animal,
            time_ms=25 + random.random() * 10, # Fake 30ms latency
            trigger_stage=stage
        )

    # 3. Save Summary
    summary_file = logger.end_video()
    print(f"\nTest complete. Summary saved to: {summary_file}")