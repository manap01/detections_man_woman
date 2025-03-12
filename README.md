# Object Detection with YOLOv8 and Voice AI

This project implements a real-time object detection system using YOLOv8 with optional voice feedback. It can process camera feeds, save detection results, and provide audio notifications of detected objects.

## Technologies Used

### Core Technologies
- **YOLOv8**: State-of-the-art object detection model
- **OpenCV**: For camera feed processing and image manipulation
- **Python 3.10**: Programming language
- **Conda**: Environment management
- **Roboflow**: Dataset management and preparation
- **Google Colab**: For model training

### Additional Technologies
- **gTTS (Google Text-to-Speech)**: For voice feedback
- **PIL/Pillow**: For image processing
- **PyYAML**: For configuration management
- **TensorBoard**: For training visualization (optional)
- **MobileNet SSD**: Alternative lightweight model for resource-constrained environments

## Quick Start Guide

### Setup Environment
```
conda create -n object-voice-ai python=3.10
conda activate object-voice-ai
```

### Install Dependencies
```
pip install ultralytics opencv-python pillow pyyaml gTTS
```

### Run Detection
```
python src/detect.py
```

## Model Training Process

### Data Collection
1. Use Roboflow to find datasets or create your own
2. Access datasets via:
   - https://app.roboflow.com
   - https://universe.roboflow.com

### Model Training in Google Colab
1. Install Roboflow package
2. Download your dataset
3. Train YOLOv8 model with custom parameters
4. Use TensorBoard for training visualization
5. Keep Colab running during long training sessions
6. Export the best model for your application

### Key Training Parameters
- Model: YOLOv8s (small variant)
- Image size: 800px
- Epochs: 25
- Task: Object Detection

## Additional Resources

### Required Model Files
- For YOLOv8: `best.pt` (in models/ directory)
- For MobileNet SSD:
  - `MobileNetSSD_deploy.prototxt`
  - `MobileNetSSD_deploy.caffemodel`

Download MobileNet SSD files from: [GitHub Repository](https://github.com/nikmart/pi-object-detection/tree/master)

## Usage Instructions

1. Run the application
2. Press 's' to save current frame
3. Press 'r' to start/stop recording
4. Press 'q' to quit

## Contact and Support

If you have questions or need assistance, please open an issue in this repository.

## License

[KELOMPOK1]