# vehicle-counting-yolov4-deepsort
Vehicle Counting and Velocity Estimation by YOLO v4 and DeepSORT

## Requirements
CUDA Toolkit version 10.1 https://developer.nvidia.com/cuda-10.1-download-archive-update2
tensorflow-gpu==2.3.0rc0
opencv-python==4.1.1.26
lxml
tqdm
absl-py
matplotlib
easydict
pillow

## Sample Video
Download sample video at [here](https://drive.google.com/file/d/1CelYAkWzbIAuUMpEdPxoIYs7YylzWY8x/view?usp=sharing), put the video file into the 'data/video' folder of this repository.

## Run the Vehicle Counting Code with YOLOv4 and DeepSORT

Convert darknet weights to tensorflow model
python save_model.py --model yolov4 

Run the Vehicle Counting Code with YOLOv4 and DeepSORT on video
python vehicle_count_velocity.py --video ./data/video/MVI_2966.MP4 --output ./outputs/output.mp4 --model yolov4

## References:
[yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort) by theAIGuysCode
