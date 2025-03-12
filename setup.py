from setuptools import setup, find_packages

setup(
    name="detections_man_woman",  # Nama package
    version="1.0.0",
    author="Hanif Maulana Arrasyid",
    description="Package untuk deteksi objek menggunakan machine learning",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
        "tensorflow"
    ],
)
