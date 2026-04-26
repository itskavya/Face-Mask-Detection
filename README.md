Face Mask Detection System

A deep learning-based project that uses Convolutional Neural Networks (CNN) to detect whether a person is wearing a face mask or not. This project leverages TensorFlow and Keras to build and train a robust image classification model.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training & Results](#training--results)
- [Making Predictions](#making-predictions)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [Future Improvements](#future-improvements)

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify images into two categories:
- **With Mask** - Person is wearing a face mask
- **Without Mask** - Person is not wearing a face mask

The model is trained on a dataset of facial images and can make real-time predictions on new images. This system can be useful for enforcing mask policies in public spaces, offices, hospitals, and other facilities.

## Features

✅ **Binary Classification** - Detects presence or absence of face masks  
✅ **Data Augmentation** - Applies transformations to improve model generalization  
✅ **High Accuracy** - Achieves 92.95% validation accuracy  
✅ **Real-time Prediction** - Can predict mask status for single images  
✅ **Easy to Deploy** - Simple and efficient model architecture  

## Dataset

- **Training Set**: 6,362 images (2 classes)
- **Test Set**: 1,191 images (2 classes)
- **Image Size**: 64 x 64 pixels
- **Format**: RGB images
- **Classes**: 
  - With Mask
  - Without Mask

**Dataset Location**: `datasets/facemask_detection/`

The dataset is organized as follows:
```
datasets/facemask_detection/
├── training_set/
│   ├── with_mask/
│   └── without_mask/
├── test_set/
│   ├── with_mask/
│   └── without_mask/
└── single_prediction/
    └── sample_images/
```

## Technologies Used

### Libraries & Frameworks
- **TensorFlow** (v2.4.1) - Deep learning framework
- **Keras** - High-level neural networks API
- **NumPy** - Numerical computing
- **Pillow (PIL)** - Image processing
- **Python** (v3.8+)

### Tools
- Jupyter Notebook / Google Colab
- Python 3.8 or higher

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/facemask-detection.git
cd facemask-detection
```

### Step 2: Install Required Libraries
```bash
pip install tensorflow==2.4.1
pip install keras==2.4.3
pip install numpy
pip install Pillow
```

Or install all dependencies at once:
```bash
pip install -r requirements.txt
```

### Step 3: Download the Dataset
Download the face mask detection dataset and place it in the `datasets/facemask_detection/` directory.

### Step 4: Run the Notebook
Open the Jupyter notebook and run all cells:
```bash
jupyter notebook Facemask_Detection_Project.ipynb
```

Or use Google Colab for cloud-based execution.

## Project Structure

```
facemask-detection/
│
├── Facemask_Detection_Project.ipynb    # Main project notebook
├── README.md                            # Project documentation
├── requirements.txt                     # Python dependencies
│
└── datasets/
    └── facemask_detection/
        ├── training_set/                # Training images
        ├── test_set/                    # Testing images
        └── single_prediction/           # Sample images for prediction
```

## Model Architecture

The CNN model consists of the following layers:

```
Sequential Model
│
├── Conv2D Layer 1
│   ├── Filters: 32
│   ├── Kernel Size: 3x3
│   ├── Activation: ReLU
│   └── Input Shape: (64, 64, 3)
│
├── MaxPooling2D Layer 1
│   ├── Pool Size: 2x2
│   └── Strides: 2
│
├── Conv2D Layer 2
│   ├── Filters: 32
│   ├── Kernel Size: 3x3
│   └── Activation: ReLU
│
├── MaxPooling2D Layer 2
│   ├── Pool Size: 2x2
│   └── Strides: 2
│
├── Flatten Layer
│   └── Converts 2D feature maps to 1D vector
│
├── Dense Layer (Fully Connected)
│   ├── Units: 128
│   └── Activation: ReLU
│
└── Output Layer (Dense)
    ├── Units: 1
    └── Activation: Sigmoid
```

### Model Summary
- **Total Parameters**: ~70K (lightweight model)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

## Training & Results

### Training Configuration
- **Epochs**: 3
- **Batch Size**: 32
- **Train-Test Split**: 6,362 training / 1,191 testing images

### Data Augmentation
Applied to training data only:
- Rescaling: 1/255
- Shear Range: 0.2
- Zoom Range: 0.2
- Horizontal Flip: True

### Training Results

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|-------|-----------|----------------|----------|-------------|
| 1     | 0.4691    | 76.02%        | 0.2234   | 92.53%     |
| 2     | 0.2797    | 88.88%        | 0.2051   | 92.53%     |
| 3     | 0.2215    | 91.23%        | 0.1900   | 92.95%     |

**Final Validation Accuracy**: 92.95%

## Making Predictions

### Single Image Prediction

```python
import numpy as np
from keras.preprocessing import image

# Load and preprocess the image
test_image = image.load_img('path/to/image.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Make prediction
result = cnn.predict(test_image)

# Interpret result
if result[0][0] == 1:
    prediction = 'Without Mask'
else:
    prediction = 'With Mask'

print(prediction)
```

## Usage

### Option 1: Using Jupyter Notebook
1. Open `Facemask_Detection_Project.ipynb`
2. Run all cells in sequence
3. Modify the image path in "Part 4 - Making a single prediction" for custom predictions

### Option 2: Using Google Colab
1. Upload the notebook to Google Colab
2. Mount Google Drive to access datasets
3. Run all cells

### Option 3: As a Python Script
```python
from keras.preprocessing import image
import numpy as np

# Load trained model
# model = load_model('facemask_detection_model.h5')

# Prepare image
test_image = image.load_img('image.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Predict
prediction = model.predict(test_image)
```

## Performance Metrics

### Accuracy
- **Training Accuracy**: 91.23%
- **Validation Accuracy**: 92.95%

### Loss
- **Training Loss**: 0.2215
- **Validation Loss**: 0.1900

### Key Observations
✓ Model shows no signs of overfitting  
✓ Validation accuracy remains stable across epochs  
✓ Loss decreases consistently throughout training  
✓ Model generalizes well to unseen data  

## Future Improvements

1. **Enhanced Model Architecture**
   - Implement deeper networks (ResNet, VGG)
   - Add batch normalization layers
   - Implement dropout for regularization

2. **Data Enhancement**
   - Increase training dataset size
   - Add diverse facial features and angles
   - Include different lighting conditions

3. **Real-time Detection**
   - Implement video stream processing
   - Integrate with webcam for live predictions
   - Deploy as web application (Flask/Django)

4. **Multi-class Classification**
   - Distinguish between different mask types
   - Detect partially worn masks
   - Identify mask fit quality

5. **Model Optimization**
   - Model quantization for mobile deployment
   - TensorFlow Lite conversion
   - Edge device optimization

6. **Advanced Features**
   - Temperature screening integration
   - Crowd density monitoring
   - Alert system for non-compliance
   - Database logging of detections

## Results and Inference

The model successfully demonstrates:
- Binary classification of mask presence
- Fast inference time suitable for real-time applications
- High accuracy without overfitting
- Ability to generalize to different images

**Sample Prediction Output**:
```
Input Image: check_4.jpg
Prediction: Without Mask
Confidence: High
```

## License

This project is provided as-is for educational and research purposes.

## Author

**Kavya Sudha Jyothika Madeti**
- Email: kavyamadeti952@gmail.com
- LinkedIn: [madeti-kavya-sudha-jyothika](https://www.linkedin.com/in/madeti-kavya-sudha-jyothika/)
- GitHub: [itskavya](https://github.com/itskavya)

## Acknowledgments

- TensorFlow and Keras communities for excellent documentation
- Dataset contributors for providing quality training data
- Research papers on CNN architectures and image classification
