from ultralytics import YOLO
import os

os.chdir('E:\Aquafoundry')

model = YOLO('yolov8n-cls.pt')
results = model.train(data='E:\Training-Data\Lettuce', epochs=100, imgsz=64)

