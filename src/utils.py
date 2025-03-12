# src/utils.py
import os
import yaml
import cv2
import numpy as np
import time
from pathlib import Path

def load_config(config_path):
    """
    Memuat file konfigurasi YAML
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error membaca konfigurasi: {str(e)}")
        return {}

def get_available_cameras():
    """
    Mendeteksi kamera yang tersedia di sistem
    """
    available_cameras = []
    for i in range(10):  # Cek index kamera 0-9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def create_project_structure(base_dir="."):
    """
    Membuat struktur folder proyek yang lengkap
    """
    folders = [
        "models",
        "notebooks",
        "src",
        "config",
        "data/train",
        "data/val",
        "data/test",
        "output/images",
        "output/videos"
    ]
    
    for folder in folders:
        path = os.path.join(base_dir, folder)
        os.makedirs(path, exist_ok=True)
        print(f"Folder dibuat: {path}")
    
    # Buat file __init__.py kosong di folder src
    with open(os.path.join(base_dir, "src", "__init__.py"), 'w') as f:
        pass
    
    # Buat file README.md dasar
    with open(os.path.join(base_dir, "README.md"), 'w') as f:
        f.write("# Proyek Deteksi Objek dengan YOLOv8\n\n")
        f.write("Proyek ini mengimplementasikan deteksi objek real-time menggunakan YOLOv8 dan OpenCV.\n")
    
    print(f"Struktur proyek berhasil dibuat di: {os.path.abspath(base_dir)}")

def calculate_fps(prev_time):
    """
    Menghitung FPS
    """
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    return fps, current_time

if __name__ == "__main__":
    create_project_structure()
    cameras = get_available_cameras()
    if cameras:
        print(f"Kamera tersedia: {cameras}")
    else:
        print("Tidak ada kamera yang terdeteksi!")
