import os
import cv2
from ultralytics import YOLO

MODEL_PATH = r'runs/detect/train2/weights/best.pt' 

TEST_IMAGES_DIR = r'data/test/images'

OUTPUT_DIR = 'app_detections'


CONFIDENCE_THRESHOLD = 0.5

def run_detection_app():
  
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results will be saved in the '{OUTPUT_DIR}' folder.")

    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please check if the file exists at: {MODEL_PATH}")
        return

    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in '{TEST_IMAGES_DIR}'. Please check the path.")
        return

    print(f"Found {len(image_files)} images to process...")

    for image_name in image_files:
        image_path = os.path.join(TEST_IMAGES_DIR, image_name)
        
        print(f"\nProcessing: {image_name}")

        results = model.predict(source=image_path, conf=CONFIDENCE_THRESHOLD)

        annotated_image = results[0].plot()

        window_name = f'Detection Result for {image_name}'
        cv2.imshow(window_name, annotated_image)
        output_path = os.path.join(OUTPUT_DIR, f'detected_{image_name}')
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved result to: {output_path}")

        print("Press any key to continue to the next image...")
        cv2.waitKey(0)

   
        cv2.destroyWindow(window_name)

    cv2.destroyAllWindows()
    print("\nDetection process finished for all images.")


if __name__ == '__main__':
    run_detection_app()
