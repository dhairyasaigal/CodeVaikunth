# UNLOCKING OBJECTS - Space Station Object Detection Model.

**Team Name:** CODE VAIKUNTH
**Project Title:** Cosmic Clarity - Unveiling Objects with Advanced AI

---

## 1. Project Overview

This project is our submission for the Duality AI Space Station Hackathon. Our vision, "Cosmic Clarity - Unveiling Objects with Advanced AI," focuses on leveraging artificial intelligence to achieve highly accurate object detection, crucial for ensuring safety and efficiency aboard space stations. We trained a robust object detection model using a high-quality synthetic dataset from Duality AI's Falcon digital twin simulation platform. Our work aimed to optimize detection accuracy under realistic constraints, including diverse lighting conditions, varied object angles, and instances of occlusion for target objects: "Toolbox," "Oxygen Tank," and "Fire Extinguisher."

---

## 2. Methodology & Iterative Improvement

Our approach centered on an iterative process of training, evaluation, and optimization to enhance our YOLOv8 model's performance.

### 2.1. Environment Setup & Dataset Integration

We established a dedicated Python environment named "EDU" using Anaconda, configured by the `setup_env.bat` script provided in the `ENV_SETUP` subfolder of the `Hackathon_Dataset`. This environment includes all necessary dependencies for the YOLOv8 framework, including PyTorch.

The synthetic dataset, crucial for training our model in a simulated space station environment, was downloaded. This dataset is pre-processed and organized, containing images along with YOLO-compatible labels structured into distinct Train, Validation, and Test folders, designed to replicate various challenging scenarios.

### 2.2. Baseline Model Training & Initial Performance Benchmark (Training Result 1)

We initiated the training of our initial YOLOv8 model using the provided `train.py` script. Upon completion, we evaluated its performance on the unseen test dataset using `predict.py`, obtaining crucial metrics.

**Initial Performance (Training Result 1):**
Our model achieved an overall **mAP@0.5 of 0.805**.
* **FireExtinguisher:** 0.838 mAP@0.5
* **ToolBox:** 0.803 mAP@0.5
* **OxygenTank:** 0.773 mAP@0.5

This highlighted "OxygenTank" detection as our primary focus for optimization.

### 2.3. Challenges Faced: False Detection of Objects

During our initial evaluation, we identified specific failure cases, particularly for "OxygenTank" and "Toolbox," which motivated our optimization strategy:

* **Fire Extinguisher:** Object mostly out of frame, insufficient features for detection. Extreme top-down perspective not well-represented in training data.
* **ToolBox:** Object significantly hidden by foreground elements and dense background. Strong glare and reflections obscured object details.Bright reflections reduced object visibility and differentiation from background. Object's lying position might be underrepresented in training.
* **Object Concealed:** Object almost entirely concealed by a large blue cylinder, with only a tiny sliver visible, making identification nearly impossible.

### 2.4. Solution: Targeted Data Augmentation Strategy

To significantly boost accuracy, especially for "OxygenTank" and "Toolbox," we implemented a **targeted data augmentation strategy**. This involved intelligently generating specific types of new training data where it would have the most impact, rather than just randomly creating more pictures.

The core of our solution involved creating **two distinct sets of data transformations, or "augmentation pipelines"**:

1.  **The "strong" transform:** This pipeline was designed to make the model much more robust to challenging real-world conditions. It included:
    * **Geometric augmentations:** Aggressively rotating, shifting, scaling, and applying perspective changes to teach the model to recognize "Oxygen Tanks" and "Toolboxes" regardless of their angle, distance, or position, even if only partially visible.
    * **Photometric augmentations:** Introducing significant variations in lighting, contrast, and color to help the model identify objects in very bright, dim, or oddly lit environments.
    * **Occlusion augmentations (dropout, cutout):** Deliberately hiding parts of the objects or introducing random "holes" in the image to train the model to detect objects even when they are partially blocked or occluded.

    This "strong" pipeline was applied **multiple times** to every original image that contained an "Oxygen Tank" or "Toolbox." This effectively "oversampled" these underperforming classes, giving the model much more exposure to the difficult scenarios they represent.

2.  **The "light" transform:** For the "FireExtinguisher" class, which was already performing well, we used a much milder set of augmentations. This ensured we still added some diversity to its training data (like simple horizontal flips and minor brightness changes) without over-emphasizing a class the model already understood, thus keeping the training focused on the areas that needed the most improvement.

### 2.5. Improved Performance (Training Result 2)

By applying these targeted changes, we successfully enhanced our model's prediction percentage for all classes, achieving a significant improvement of **5%** in overall mAP@0.5.

**Improved Performance (Training Result 2):**
Our model achieved an overall **mAP@0.5 of 0.933**.
* **FireExtinguisher:** 0.875 mAP@0.5
* **ToolBox:** 0.867 mAP@0.5
* **OxygenTank:** 0.853 mAP@0.5

The model now performs well across all three objects, demonstrating significantly improved robustness.

---


## 4. Bonus: Use Case Application

We have developed a basic Python desktop application to demonstrate the practical application of our trained object detection model.

### 4.1. How to Run the Application

1.  Navigate to your project's root directory: `cd [Your_Project_Path]/HackByte_Dataset`
2.  Activate your `EDU` environment: `conda activate EDU`
3.  Open `detection_app.py` and **update `MODEL_PATH`** to point to your final `best.pt` model weights.
4.  Run the application:
    ```bash
    python detection_app.py
    ```
    The app will process images from the configured `TEST_IMAGES_DIR`, display detections, and save results to the `app_detections` folder.

---

## 5. Conclusion & Future Work

We have successfully developed a highly accurate object detection model for the space station environment, significantly improving detection capabilities for all target classes.

For future work, we propose exploring:

* **Advanced Augmentation:** Leveraging Falcon to design and generate highly specific, complex synthetic scenes with extreme occlusions and diverse lighting. This pushes the model's robustness to real-world challenges even further.
* **Domain Adaptation:** Developing methods to make the model work effectively in new space station modules or with updated equipment designs with minimal retraining. This ensures the model remains relevant as the environment evolves, saving time and resources.
* **Self-Supervised Learning:** Enabling the model to continuously learn and improve from abundant unlabeled video feeds from the station. This reduces reliance on manual data labeling, allowing for autonomous model updates and adaptation over time.

---

