Pipeline:

1. YOLOv8 Object Detection → Detect vùng (roundId, timer, dice, boxes)
2. PaddleOCR/EasyOCR → OCR text trong các vùng
3. Custom Classifier → Phân loại dice màu đỏ/trắng
4. Logic xử lý → Tổng hợp kết quả

# Tạo project

mkdir xocdia-detector
cd xocdia-detector

# Tạo virtual environment

python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

# Install dependencies

pip install ultralytics opencv-python paddleocr numpy pillow torch torchvision

# Optional: nếu dùng GPU

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

cấu trúc thu mực
xocdia-detector/
├── dataset/
│ ├── images/
│ │ ├── train/ # 80% ảnh training
│ │ └── val/ # 20% ảnh validation
│ └── labels/
│ ├── train/ # Labels tương ứng
│ └── val/
├── rounds/ # Lưu kết quả
├── label_tool.py # Tool đánh label
├── train.py # Train model
├── ocr_engine.py # OCR engine
├── detector.py # Main detector
├── realtime_capture.py # Real-time capture
└── xocdia.yaml # Config file

Quy trình đầy đủ

# 1. Collect data (chụp 100-200 ảnh từ game)

# Lưu vào dataset/images/

# 2. Label ảnh

python label_tool.py

# 3. Split train/val (80/20)

# Di chuyển file thủ công hoặc dùng script

# 4. Train model

python train.py

# 5. Test detection

python detector.py

# 6. Run real-time

python realtime_capture.py

💡 Tối Ưu & Tips

Data cần bao nhiêu?

Minimum: 50-100 ảnh
Recommended: 200-500 ảnh
Chụp nhiều trạng thái khác nhau

Augmentation tự động:

# Trong train.py, thêm:

results = model.train( # ...
augment=True,
hsv_h=0.015,
hsv_s=0.7,
hsv_v=0.4,
degrees=0,
translate=0.1,
scale=0.5,
mosaic=1.0
)

Model size options:

python# nano (6MB, fastest)
model = YOLO('yolov8n.pt')

# small (22MB, balanced)

model = YOLO('yolov8s.pt')

# medium (50MB, most accurate)

model = YOLO('yolov8m.pt')
