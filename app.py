from flask import Flask, request, send_file
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolov8n-seg.pt")  # or yolov8m-seg.pt for better results

def blur_car(image_np):
    results = model.predict(image_np, save=False, imgsz=640)
    mask_total = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)

    for r in results:
        if r.masks is not None:
            for seg in r.masks.data:
                seg = seg.cpu().numpy()

                # Resize the segmentation mask to the input image size
                seg_resized = cv2.resize(seg, (image_np.shape[1], image_np.shape[0]))
                binary_mask = (seg_resized > 0.5).astype(np.uint8)

                # Dilate the mask to expand coverage (padding)
                kernel = np.ones((25, 25), np.uint8)  # Tune as needed
                dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

                # Combine multiple masks if there are multiple cars
                mask_total = cv2.bitwise_or(mask_total, dilated_mask)

    # Apply Gaussian blur only where mask is present
    blurred_image = cv2.GaussianBlur(image_np, (151, 151), 0)  # Adjust blur intensity

    # Copy non-masked (background) pixels from the original image
    mask_3ch = np.stack([mask_total]*3, axis=-1)
    result = np.where(mask_3ch == 1, blurred_image, image_np)

    return result


@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert("RGB")
    image_np = np.array(image)

    processed = blur_car(image_np)
    _, buffer = cv2.imencode('.jpg', processed)
    return send_file(BytesIO(buffer.tobytes()), mimetype='image/jpeg')

if __name__ == '__main__':
      app.run(debug=True)

