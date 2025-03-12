# src/detect.py
import cv2
import time
import numpy as np
from ultralytics import YOLO
import os
import datetime

class ObjectDetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):

        """
        Inisialisasi Object Detector
        
        Args:
            model_path (str): Path ke file model YOLOv8 (misalnya, best.pt)
            conf_threshold (float): Threshold kepercayaan untuk deteksi
            iou_threshold (float): Threshold IOU untuk NMS
        """
        # Memastikan file model ada
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")
            
        print(f"Memuat model dari {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        print("Model berhasil dimuat!")
        
        # Mendapatkan daftar kelas dari model
        self.class_names = self.model.names
        print(f"Kelas yang dapat dideteksi: {self.class_names}")
        
        # Membuat direktori output jika belum ada
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"))
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "videos"), exist_ok=True)
        
    def detect_from_camera(self, camera_id=0, width=640, height=480, save_video=False):
        
        """
        Melakukan deteksi objek dari kamera
        
        Args:
            camera_id (int): ID kamera (default: 0 untuk webcam utama)
            width (int): Lebar frame
            height (int): Tinggi frame
            save_video (bool): Apakah akan menyimpan video hasil deteksi
        """
        # Inisialisasi kamera
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not cap.isOpened():
            print("Error: Tidak dapat membuka kamera.")
            return
            
        print(f"Kamera berhasil dibuka dengan resolusi {width}x{height}")
        
        # Inisialisasi video writer jika save_video=True
        video_writer = None
        if save_video:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, "videos", f"detection_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
            print(f"Video akan disimpan ke: {output_path}")
        
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Tidak dapat menangkap frame.")
                    break
                
                # Hitung FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Lakukan deteksi objek pada frame
                results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
                # Hasil deteksi berupa gambar beranotasi
                annotated_frame = results[0].plot()
                
                # Tampilkan FPS
                cv2.putText(annotated_frame, f"FPS: {fps}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Hitung dan tampilkan jumlah objek per kelas
                detected_classes = {}
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls.item())
                        class_name = self.class_names[cls_id]
                        detected_classes[class_name] = detected_classes.get(class_name, 0) + 1
                        
                y_pos = 70
                for class_name, count in detected_classes.items():
                    text = f"{class_name}: {count}"
                    cv2.putText(annotated_frame, text, (10, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_pos += 30
                
                # Tampilkan frame dengan deteksi
                cv2.imshow("Deteksi Objek YOLOv8", annotated_frame)
                
                # Simpan video jika diperlukan
                if save_video and video_writer is not None:
                    video_writer.write(annotated_frame)
                
                # Tekan 's' untuk menyimpan frame, 'q' untuk keluar
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Keluar dari deteksi...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = os.path.join(self.output_dir, "images", f"detection_{timestamp}.jpg")
                    cv2.imwrite(img_path, annotated_frame)
                    print(f"Frame disimpan ke: {img_path}")
        
        except Exception as e:
            print(f"Error selama deteksi: {str(e)}")
        
        finally:
            if save_video and video_writer is not None:
                video_writer.release()
            cap.release()
            cv2.destroyAllWindows()
            print("Deteksi selesai!")
    
    def detect_on_image(self, image_path, save_output=True):
        """
        Melakukan deteksi objek pada gambar
        
        Args:
            image_path (str): Path ke gambar
            save_output (bool): Simpan hasil deteksi ke folder output/images
            
        Returns:
            numpy.ndarray: Gambar dengan hasil deteksi
        """
        if not os.path.exists(image_path):
            print(f"Gambar tidak ditemukan: {image_path}")
            return None
            
        img = cv2.imread(image_path)
        results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)
        annotated_img = results[0].plot()
        
        if save_output:
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(self.output_dir, "images", f"{name}_detected{ext}")
            cv2.imwrite(output_path, annotated_img)
            print(f"Hasil deteksi disimpan ke: {output_path}")
            
        return annotated_img

# Fungsi utama untuk menjalankan deteksi dari kamera
def run_detection(model_path, camera_id=0, save_video=False):
    try:
        detector = ObjectDetector(model_path)
        detector.detect_from_camera(camera_id=camera_id, save_video=save_video)
    except Exception as e:
        print(f"Error dalam menjalankan deteksi: {str(e)}")

if __name__ == "__main__":
    # Path ke model; asumsikan script dijalankan dari folder 'src', jadi naik satu level ke root lalu ke folder models
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best.pt")
    run_detection(model_path=MODEL_PATH)