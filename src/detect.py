# src/detect.py
import cv2
import time
import numpy as np
from ultralytics import YOLO
import os
import datetime
import threading
import queue
import torch
import math
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SmartDetector")

class PersonAnalyzer:
    """Class for analyzing and differentiating between men and women"""
    
    def __init__(self):
        # Load specific model for gender classification if available
        # If not using a specific gender model, we'll use visual cues and proportions
        self.gender_confidence_threshold = 0.7
        self.history_buffer = {}  # Track objects across frames
        self.history_size = 10
        
    def analyze_person(self, frame: np.ndarray, bbox: List[float], person_id: int) -> Dict:
        """
        Analyze a detected person to determine gender and other attributes
        Returns a dictionary of attributes with confidence levels
        """
        x1, y1, x2, y2 = map(int, bbox)
        person_img = frame[y1:y2, x1:x2]
        
        if person_img.size == 0:
            return {"gender": "unknown", "confidence": 0.0}
        
        # Create a unique tracking ID based on position and size
        if person_id not in self.history_buffer:
            self.history_buffer[person_id] = []
            
        # Simple heuristic-based gender classification
        # In a production system, you'd use a dedicated gender classification model
        height = y2 - y1
        width = x2 - x1
        aspect_ratio = height / max(width, 1)
        
        # Size of head relative to body can be a hint
        head_region = person_img[0:int(height/4), :]
        
        # Analyze color distribution (men often have less color variation)
        color_std = np.std(person_img, axis=(0, 1))
        color_variety = np.mean(color_std)
        
        # Determine gender based on combined factors
        # This is a simplified approach - a dedicated neural network would be more accurate
        male_indicators = 0
        female_indicators = 0
        
        # Height to width ratio tends to be higher for women
        if aspect_ratio > 3.2:
            female_indicators += 1
        elif aspect_ratio < 2.8:
            male_indicators += 1
            
        # Color variety tends to be higher in women's clothing (simplified heuristic)
        if color_variety > 50:
            female_indicators += 1
        else:
            male_indicators += 1
            
        # Calculate confidence based on agreement between indicators
        total_indicators = male_indicators + female_indicators
        
        if male_indicators > female_indicators:
            gender = "man"
            confidence = male_indicators / total_indicators if total_indicators > 0 else 0.5
        elif female_indicators > male_indicators:
            gender = "woman"
            confidence = female_indicators / total_indicators if total_indicators > 0 else 0.5
        else:
            gender = "unknown"
            confidence = 0.5
            
        # Store result in history buffer
        result = {"gender": gender, "confidence": confidence}
        self.history_buffer[person_id].append(result)
        if len(self.history_buffer[person_id]) > self.history_size:
            self.history_buffer[person_id].pop(0)
            
        # Smooth results across frames for stability
        if len(self.history_buffer[person_id]) >= 3:
            gender_counts = {"man": 0, "woman": 0, "unknown": 0}
            for entry in self.history_buffer[person_id]:
                gender_counts[entry["gender"]] += 1
                
            # Use most common gender with increased confidence
            most_common_gender = max(gender_counts, key=gender_counts.get)
            most_common_count = gender_counts[most_common_gender]
            
            if most_common_gender != "unknown" and most_common_count > len(self.history_buffer[person_id]) / 2:
                confidence = min(0.95, 0.5 + (most_common_count / len(self.history_buffer[person_id])) * 0.5)
                result = {"gender": most_common_gender, "confidence": confidence}
                
        return result

class SmartObjectDetector:
    def __init__(self, 
                 model_path: str, 
                 conf_threshold: float = 0.5, 
                 iou_threshold: float = 0.45,
                 use_gpu: bool = True,
                 person_analysis: bool = True):
        """
        Inisialisasi Smart Object Detector
        
        Args:
            model_path (str): Path ke file model YOLOv8 (misalnya, best.pt)
            conf_threshold (float): Threshold kepercayaan untuk deteksi
            iou_threshold (float): Threshold IOU untuk NMS
            use_gpu (bool): Gunakan GPU jika tersedia
            person_analysis (bool): Aktifkan analisis lebih lanjut untuk orang (gender)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")
            
        logger.info(f"Memuat model dari {model_path}...")
        
        # Determine device
        self.device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Menggunakan device: {self.device}")
        
        # Load model with optimized settings
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Set detection parameters
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Enable half-precision if using GPU
        if self.device != "cpu":
            self.model.half()  # FP16 inference
        
        logger.info("Model berhasil dimuat!")
        
        # Mendapatkan daftar kelas dari model
        self.class_names = self.model.names
        logger.info(f"Kelas yang dapat dideteksi: {self.class_names}")
        
        # Setup person analysis
        self.person_analysis = person_analysis
        if person_analysis:
            self.person_analyzer = PersonAnalyzer()
            logger.info("Analisis orang diaktifkan (deteksi gender)")
        
        # Setup tracking 
        self.next_object_id = 0
        self.objects = {}  # Dictionary to track objects over frames
        
        # Performance monitoring
        self.fps_history = []
        self.max_fps_history = 30
        
        # Buat direktori output jika belum ada
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"))
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "videos"), exist_ok=True)
        
        # Setup threading for parallel processing
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        self.is_running = False
        self.processing_thread = None
        
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        # Box format: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Box areas
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Union area
        union = box1_area + box2_area - intersection
        
        return intersection / max(union, 1e-6)
    
    def _assign_object_ids(self, results, prev_objects):
        """Assign consistent IDs to detected objects across frames"""
        current_objects = {}
        
        if results[0].boxes is not None:
            # Extract current detections
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls.item())
                class_name = self.class_names[cls_id]
                xyxy = box.xyxy.cpu().numpy()[0]
                conf = float(box.conf.item())
                
                # Try to match with previous objects
                matched = False
                for obj_id, obj_data in prev_objects.items():
                    # Match if same class and significant overlap
                    if obj_data["class_name"] == class_name and self._calculate_iou(xyxy, obj_data["box"]) > 0.5:
                        current_objects[obj_id] = {
                            "class_name": class_name,
                            "box": xyxy,
                            "conf": conf,
                            "age": obj_data["age"] + 1,
                            "attributes": obj_data.get("attributes", {})
                        }
                        matched = True
                        break
                
                # If no match found, assign new ID
                if not matched:
                    current_objects[self.next_object_id] = {
                        "class_name": class_name,
                        "box": xyxy,
                        "conf": conf,
                        "age": 1,
                        "attributes": {}
                    }
                    self.next_object_id += 1
        
        return current_objects
    
    def _processing_thread_function(self):
        """Thread function to process frames in parallel"""
        while self.is_running:
            try:
                # Get frame from queue with timeout
                frame_data = self.frame_queue.get(timeout=1.0)
                frame, frame_id = frame_data
                
                # Run inference (this is the most time-consuming part)
                with torch.no_grad():
                    results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
                
                # Track objects across frames
                self.objects = self._assign_object_ids(results, self.objects)
                
                # Analysis for people
                if self.person_analysis:
                    for obj_id, obj_data in self.objects.items():
                        if obj_data["class_name"] == "person" and obj_data["age"] % 3 == 0:  # Only analyze every 3 frames
                            gender_info = self.person_analyzer.analyze_person(frame, obj_data["box"], obj_id)
                            obj_data["attributes"]["gender"] = gender_info
                
                # Put results in queue
                self.result_queue.put((results, frame_id, self.objects.copy()))
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing thread: {e}")
                continue
    
    def detect_from_camera(self, camera_id=0, width=1280, height=720, save_video=False, show_display=True):
        """
        Melakukan deteksi objek dari kamera secara real-time dengan performa yang ditingkatkan.
        
        Args:
            camera_id (int): ID kamera (default: 0 untuk webcam utama)
            width (int): Lebar frame (higher for better detection)
            height (int): Tinggi frame
            save_video (bool): Jika True, rekam dan simpan video
            show_display (bool): Tampilkan preview (disable for headless mode)
        """
        # Initialize camera with higher resolution for better detection
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Try to increase camera buffer size to prevent frame drops
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        if not cap.isOpened():
            logger.error("Error: Tidak dapat membuka kamera.")
            return
            
        logger.info(f"Kamera berhasil dibuka dengan resolusi {width}x{height}")
        
        # Setup video writer if needed
        video_writer = None
        if save_video:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, "videos", f"smart_detection_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
            logger.info(f"Video akan disimpan ke: {output_path}")
        
        # Start processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_thread_function)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Performance monitoring
        fps_start_time = time.time()
        frame_count = 0
        fps = 0
        
        # Frame ID for synchronization
        current_frame_id = 0
        last_processed_frame_id = -1
        last_results = None
        last_objects = {}
        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Tidak dapat menangkap frame.")
                    break
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - fps_start_time
                if elapsed_time >= 1.0:
                    fps = frame_count / elapsed_time
                    self.fps_history.append(fps)
                    if len(self.fps_history) > self.max_fps_history:
                        self.fps_history.pop(0)
                    frame_count = 0
                    fps_start_time = time.time()
                
                # Submit frame for processing if queue not full
                if not self.frame_queue.full():
                    self.frame_queue.put((frame.copy(), current_frame_id))
                    current_frame_id += 1
                
                # Get processed results if available
                try:
                    while not self.result_queue.empty():
                        results, frame_id, objects = self.result_queue.get(block=False)
                        if frame_id > last_processed_frame_id:
                            last_processed_frame_id = frame_id
                            last_results = results
                            last_objects = objects
                        self.result_queue.task_done()
                except queue.Empty:
                    pass
                
                # Display results if available
                if last_results is not None and show_display:
                    # Use last_results to annotate current frame
                    annotated_frame = last_results[0].plot()
                    
                    # Draw additional information
                    avg_fps = sum(self.fps_history) / max(len(self.fps_history), 1)
                    
                    # Status information
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.putText(annotated_frame, f"Device: {self.device}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display gender information for detected people
                    for obj_id, obj_data in last_objects.items():
                        if obj_data["class_name"] == "person" and "attributes" in obj_data:
                            if "gender" in obj_data["attributes"]:
                                gender_info = obj_data["attributes"]["gender"]
                                x1, y1, x2, y2 = map(int, obj_data["box"])
                                
                                if gender_info["confidence"] > 0.6:
                                    gender_text = f"{gender_info['gender'].upper()} ({gender_info['confidence']:.2f})"
                                    
                                    # Choose color based on gender
                                    if gender_info['gender'] == "man":
                                        color = (255, 0, 0)  # Blue for men
                                    elif gender_info['gender'] == "woman":
                                        color = (0, 0, 255)  # Red for women
                                    else:
                                        color = (255, 255, 255)  # White for unknown
                                    
                                    # Draw gender label above the bounding box
                                    cv2.putText(annotated_frame, gender_text, (x1, y1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Class counts with visual indicator
                    class_counts = {}
                    for obj_data in last_objects.values():
                        class_name = obj_data["class_name"]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Display detected classes with counts
                    y_pos = 90
                    for class_name, count in class_counts.items():
                        text = f"{class_name}: {count}"
                        cv2.putText(annotated_frame, text, (10, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_pos += 30
                    
                    # Display frame
                    cv2.imshow("Smart Object Detection", annotated_frame)
                    
                    # Save to video if enabled
                    if save_video and video_writer is not None:
                        video_writer.write(annotated_frame)
                
                # Process key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Keluar dari deteksi...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = os.path.join(self.output_dir, "images", f"smart_detection_{timestamp}.jpg")
                    if last_results is not None:
                        cv2.imwrite(img_path, annotated_frame)
                        logger.info(f"Frame disimpan ke: {img_path}")
        
        except Exception as e:
            logger.error(f"Error selama deteksi: {str(e)}")
        
        finally:
            # Cleanup resources
            self.is_running = False
            if self.processing_thread is not None:
                self.processing_thread.join(timeout=1.0)
            
            if save_video and video_writer is not None:
                video_writer.release()
            
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Deteksi selesai!")
    
    def detect_on_image(self, image_path, save_output=False):
        """
        Melakukan deteksi objek pada sebuah gambar dengan analisis tambahan.
        
        Args:
            image_path (str): Path ke gambar
            save_output (bool): Jika True, simpan hasil deteksi
        
        Returns:
            numpy.ndarray: Gambar dengan hasil deteksi.
        """
        if not os.path.exists(image_path):
            logger.error(f"Gambar tidak ditemukan: {image_path}")
            return None
            
        img = cv2.imread(image_path)
        
        # Run inference
        with torch.no_grad():
            results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)
        
        # Process detected objects (including gender detection)
        detected_objects = {}
        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls.item())
                class_name = self.class_names[cls_id]
                xyxy = box.xyxy.cpu().numpy()[0]
                
                detected_objects[i] = {
                    "class_name": class_name,
                    "box": xyxy,
                    "conf": float(box.conf.item()),
                    "attributes": {}
                }
                
                if self.person_analysis and class_name == "person":
                    gender_info = self.person_analyzer.analyze_person(img, xyxy, i)
                    detected_objects[i]["attributes"]["gender"] = gender_info
        
        # Plot base results
        annotated_img = results[0].plot()
        
        # Add gender information
        for obj_id, obj_data in detected_objects.items():
            if obj_data["class_name"] == "person" and "attributes" in obj_data:
                if "gender" in obj_data["attributes"]:
                    gender_info = obj_data["attributes"]["gender"]
                    x1, y1, x2, y2 = map(int, obj_data["box"])
                    
                    if gender_info["confidence"] > 0.6:
                        gender_text = f"{gender_info['gender'].upper()} ({gender_info['confidence']:.2f})"
                        
                        # Choose color based on gender
                        if gender_info['gender'] == "man":
                            color = (255, 0, 0)  # Blue for men
                        elif gender_info['gender'] == "woman":
                            color = (0, 0, 255)  # Red for women
                        else:
                            color = (255, 255, 255)  # White for unknown
                        
                        # Draw gender label above the bounding box
                        cv2.putText(annotated_img, gender_text, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if save_output:
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(self.output_dir, "images", f"{name}_smart_detected{ext}")
            cv2.imwrite(output_path, annotated_img)
            logger.info(f"Hasil deteksi disimpan ke: {output_path}")
            
        return annotated_img

def run_detection(model_path, camera_id=0, save_video=False, use_gpu=True):
    """Run object detection with the enhanced detector"""
    try:
        detector = SmartObjectDetector(model_path, use_gpu=use_gpu, person_analysis=True)
        detector.detect_from_camera(camera_id=camera_id, save_video=save_video)
    except Exception as e:
        logger.error(f"Error dalam menjalankan deteksi: {str(e)}")

if __name__ == "__main__":
    # Asumsikan script dijalankan dari folder src, naik satu level ke root, kemudian ke folder models
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best.pt")
    
    # Aktifkan penggunaan GPU jika tersedia
    use_gpu = torch.cuda.is_available()
    logger.info(f"GPU tersedia: {use_gpu}")
    
    run_detection(model_path=MODEL_PATH, use_gpu=use_gpu, save_video=True)