import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.ops import nms
import torchvision
import torchvision.transforms as T
from ultralytics import YOLO
import torchvision.models.detection as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# YOLO Model Configuration
YOLO_LABEL_MAPPING = {
    0: 'DNoHelmet',
    1: 'DHelmet',
    2: 'DHelmetPHelmet',
    3: 'DHelmetPNoHelmet',
    4: 'DNoHelmetPHelmet',
    5: 'DNoHelmetPNoHelmet'
}

YOLO_COLOR_MAPPING = {
    0: (0, 255, 0),   # Green for DNoHelmet
    1: (255, 0, 0),   # Blue for DHelmet
    2: (0, 0, 255),   # Red for DHelmetPHelmet
    3: (255, 255, 0), # Cyan for DHelmetPNoHelmet
    4: (255, 0, 255), # Magenta for DNoHelmetPHelmet
    5: (0, 255, 255)  # Yellow for DNoHelmetPNoHelmet
}

# Faster R-CNN Model Configuration
FRCNN_LABEL_MAPPING = {
    1: 'DNoHelmet',
    2: 'DHelmet',
    3: 'DHelmetPHelmet',
    4: 'DHelmetPNoHelmet',
    5: 'DNoHelmetPHelmet',
    6: 'DNoHelmetPNoHelmet'
}

FRCNN_COLOR_MAPPING = {
    # 1: (0, 255, 0),   # Green for DNoHelmet
    # 2: (255, 0, 0),   # Blue for DHelmet
    # 3: (0, 0, 255),   # Red for DHelmetPHelmet
    # 4: (255, 255, 0), # Cyan for DHelmetPNoHelmet
    # 5: (255, 0, 255), # Magenta for DNoHelmetPHelmet
    # 6: (0, 255, 255)  # Yellow for DNoHelmetPNoHelmet
    1: (0, 255, 0),   # Green (BGR: G in middle)
    2: (0, 0, 255),   # Blue (BGR: R becomes first)
    3: (255, 0, 0),   # Red (BGR: B becomes last)
    4: (0, 255, 255), # Cyan
    5: (255, 0, 255), # Magenta
    6: (255, 255, 0)  # Yellow
}

# RetinaNet Model Configuration
RETINANET_LABEL_MAPPING = {
    1: 'DNoHelmet',
    2: 'DHelmet',
    3: 'DHelmetPHelmet',
    4: 'DHelmetPNoHelmet',
    5: 'DNoHelmetPHelmet',
    6: 'DNoHelmetPNoHelmet'
}

RETINANET_COLOR_MAPPING = {
    1: (0, 255, 0),   # Green (BGR: G in middle)
    2: (0, 0, 255),   # Blue (BGR: R becomes first)
    3: (255, 0, 0),   # Red (BGR: B becomes last)
    4: (0, 255, 255), # Cyan
    5: (255, 0, 255), # Magenta
    6: (255, 255, 0)  # Yellow
}

# Model Loading Functions
@st.cache_resource
def load_yolo_model():
    """Load YOLO model"""
    model_path = r'C:\Users\HoangVan\Documents\Zô đây mà xem tài liệu học\Kì 5\Thị giác máy tính nâng cao\Đồ án\Test\CS331-Demo\weights\best.pt'
    model = YOLO(model_path)
    return model

@st.cache_resource
def load_faster_rcnn_model():
    """Load Faster R-CNN model"""
    num_classes = 7
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Modify the predictor for the specific number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load the pre-trained weights
    checkpoint = torch.load(r"C:\Users\HoangVan\Documents\Zô đây mà xem tài liệu học\Kì 5\Thị giác máy tính nâng cao\Đồ án\Test\CS331-Demo\weights\fasterrcnn_resnet50_fpn_epoch_8.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    
    return model, device

@st.cache_resource
def load_retinanet_model():
    """Load RetinaNet model"""
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    num_classes = 7
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        pretrained=False, 
        pretrained_backbone=True, 
        num_classes=num_classes
    )
    
    # Load checkpoint
    checkpoint = torch.load(r'C:\Users\HoangVan\Documents\Zô đây mà xem tài liệu học\Kì 5\Thị giác máy tính nâng cao\Đồ án\Test\CS331-Demo\weights\retinanet_epoch_10.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device

# Prediction Functions
def yolo_predict(image, model):
    """Predict using YOLO"""
    # Convert image to numpy array for YOLO
    img_np_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run prediction
    results = model(img_np_bgr)
    
    # Create output image with bounding boxes
    annotated_img = img_np_bgr.copy()
    
    # Draw bounding boxes
    for box in results[0].boxes:
        cls = int(box.cls[0])  # Ensure we're getting the first element
        conf = float(box.conf[0])  # Ensure we're getting the first element
        coords = box.xyxy[0].tolist()  # Get bounding box coordinates
        
        # Map class to label and get color
        label = YOLO_LABEL_MAPPING.get(cls, 'Unknown')
        color = YOLO_COLOR_MAPPING.get(cls, (255, 255, 255))
        
        # Draw rectangle
        start_point = (int(coords[0]), int(coords[1]))
        end_point = (int(coords[2]), int(coords[3]))
        cv2.rectangle(annotated_img, start_point, end_point, color, 2)
        
        # Add label and confidence
        cv2.putText(annotated_img, f'{label} {conf:.2f}', 
                    (int(coords[0]), int(coords[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Convert back to RGB
    return cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

def faster_rcnn_predict(image, model, device):
    """Predict using Faster R-CNN"""
    # Preprocessing
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor()
    ])
    
    # Convert to RGB and transform
    image_rgb = image.convert("RGB")
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Get original image for drawing
    img_np = np.array(image)
    
    # Get predictions
    boxes = predictions[0]['boxes'].cpu()
    labels = predictions[0]['labels'].cpu()
    scores = predictions[0]['scores'].cpu()
    
    # Filter predictions
    threshold = 0.3
    keep = scores > threshold
    filtered_boxes = boxes[keep]
    filtered_labels = labels[keep]
    filtered_scores = scores[keep]
    
    # Áp dụng NMS
    iou_threshold = 0.4  # Ngưỡng IoU để loại bỏ các box dư thừa
    nms_indices = nms(filtered_boxes, filtered_scores, iou_threshold)

    # Lấy các box, label, score sau khi áp dụng NMS
    filtered_boxes = filtered_boxes[nms_indices]
    filtered_labels = filtered_labels[nms_indices]
    filtered_scores = filtered_scores[nms_indices]
    
    # Calculate scaling factors
    original_height, original_width = img_np.shape[:2]
    scale_x = original_width / 512
    scale_y = original_height / 512
    
    # Draw on image
    for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
        xmin, ymin, xmax, ymax = box
        
        # Scale coordinates back to original image size
        xmin_orig = int(xmin * scale_x)
        ymin_orig = int(ymin * scale_y)
        xmax_orig = int(xmax * scale_x)
        ymax_orig = int(ymax * scale_y)
        
        # Get label and color
        label_text = FRCNN_LABEL_MAPPING.get(label.item(), 'Unknown')
        color = FRCNN_COLOR_MAPPING.get(label.item(), (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(img_np,
                      (xmin_orig, ymin_orig),
                      (xmax_orig, ymax_orig),
                      color, 2)
        
        # Put label text
        cv2.putText(img_np, f"{label_text} {score.item():.2f}",
                    (xmin_orig, ymin_orig - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)
    
    return img_np

def retinanet_predict(image, model, device):
    """Predict using RetinaNet"""
    # Preprocess image
    transform = T.Compose([
        T.Resize((800, 800)),  
        T.ToTensor()       
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get original dimensions
    original_width, original_height = image.size
    
    # Resize and transform image for model
    input_image = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(input_image)
    
    # Prepare image for drawing
    img_np = np.array(image)
    img_draw = img_np.copy()
    
    # Process predictions
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    
    # Filter predictions
    conf_threshold = 0.3
    keep = scores >= conf_threshold
    filtered_boxes = boxes[keep]
    filtered_labels = labels[keep]
    filtered_scores = scores[keep]
    
    # Áp dụng NMS
    iou_threshold = 0.4  # Ngưỡng IoU để loại bỏ các box dư thừa
    nms_indices = nms(filtered_boxes, filtered_scores, iou_threshold)

    # Lấy các box, label, score sau khi áp dụng NMS
    filtered_boxes = filtered_boxes[nms_indices]
    filtered_labels = filtered_labels[nms_indices]
    filtered_scores = filtered_scores[nms_indices]
    
    # Calculate scaling factors
    scale_x = original_width / 800
    scale_y = original_height / 800
    
    # Draw bounding boxes
    for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
        # Scale coordinates back to original image size
        xmin, ymin, xmax, ymax = box
        xmin_orig = int(xmin.item() * scale_x)
        ymin_orig = int(ymin.item() * scale_y)
        xmax_orig = int(xmax.item() * scale_x)
        ymax_orig = int(ymax.item() * scale_y)
        
        # Get label and score
        label_text = RETINANET_LABEL_MAPPING.get(label.item(), 'Unknown')
        score_text = f"{score.item():.2f}"
        
        # Select color based on the label
        color = RETINANET_COLOR_MAPPING.get(label.item(), (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(img_draw, 
                      (xmin_orig, ymin_orig), 
                      (xmax_orig, ymax_orig), 
                      color, 2)
        
        # Draw label and score
        cv2.putText(img_draw, f"{label_text} {score_text}",
                    (xmin_orig, ymin_orig - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, color, 2)
    
    return img_draw

def main():
    # Page configuration
    st.set_page_config(page_title="Multi-Model Object Detection", page_icon=":robot_face:")
    
    # Title
    st.title("🔍 Multi-Model Object Detection")
    st.write("Upload an image and select detection models")
    
    # Multi-model selection
    model_choices = st.multiselect(
        "Select Detection Models", 
        ["YOLO", "Faster R-CNN", "RetinaNet"],
        default=["YOLO"]
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Uploaded Image", use_container_width=True)
        
        # Predict button
        if st.button("Detect Objects"):
            # Create columns based on number of selected models
            columns = st.columns(len(model_choices))
            
            with st.spinner('Detecting objects...'):
                # Run predictions for selected models
                for i, model_choice in enumerate(model_choices):
                    with columns[i]:
                        st.subheader(f"{model_choice}")
                        
                        # Load model based on selection
                        if model_choice == "YOLO":
                            model = load_yolo_model()
                            device = None
                            result_image = yolo_predict(original_image, model)
                        elif model_choice == "Faster R-CNN":
                            model, device = load_faster_rcnn_model()
                            result_image = faster_rcnn_predict(original_image, model, device)
                        else:  # RetinaNet
                            model, device = load_retinanet_model()
                            result_image = retinanet_predict(original_image, model, device)
                        
                        # Display result for each model
                        st.image(result_image, caption=f"Detected Objects", use_container_width=True)

if __name__ == "__main__":
    main()