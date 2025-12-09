# FairPlayAI: Multi-View Video Transformer for Football Foul Analysis

## 🎯 Project Goal and Overview

**FairPlayAI** is an advanced deep learning solution engineered to provide objective, data-driven analysis of foul events in football (soccer). The system leverages a **Multi-View Video Transformer (MViT)** architecture to process video streams captured from multiple camera perspectives of the same action. The primary function is to accurately predict two critical outcomes: the **Action Class** (type of foul) and the corresponding **Card Severity** (e.g., No card, Yellow card, Red card).

This project serves as a robust proof-of-concept, integrating a state-of-the-art multi-view deep learning model, trained on the **SoccerNet** dataset, with a user-friendly desktop application built using **PyQt6** for practical, real-time inference.

## ✨ Core Technical Features

The architecture is designed for high performance and multi-modal data fusion:

*   **Multi-View Feature Fusion:** The model is explicitly structured to aggregate spatio-temporal features from multiple video views of a single action, significantly enhancing the model's robustness and predictive accuracy compared to single-view systems.
*   **MViT-V2-S Backbone:** Utilizes the Mobile Video Transformer (MViT-V2-S) as the foundational feature extractor. This architecture is highly efficient for video understanding, capturing complex spatio-temporal dependencies within the action clips.
*   **Flexible Aggregation Strategies:** Implements multiple feature aggregation techniques, including **Max Pooling** (default for inference), **Average Pooling**, and a sophisticated **Weighted Attention Mechanism**, allowing for comparative analysis and optimization of multi-view fusion.
*   **Dual-Head Classification:** The model features two independent classification heads, enabling simultaneous prediction of the fine-grained **Action Class** (8 categories) and the official **Card Severity** (4 categories).
*   **PyQt6 Desktop Interface:** A standalone, cross-platform graphical user interface (GUI) is provided for easy video file selection, model execution, and clear visualization of the prediction probabilities.

## ⚙️ Deep Learning Architecture

The system's core is the **Multi-View Network (MVNetwork)**, which extends the MViT-V2-S model for multi-view input processing.

### 1. MVNetwork Structure

The `MVNetwork` class in `gui/output.py` manages the end-to-end inference pipeline:

| Component | Technology | Function |
| :--- | :--- | :--- |
| **Backbone** | `mvit_v2_s` (Pre-trained) | Extracts a **768-dimensional** feature vector for each individual video view. The final classification head of the pre-trained model is removed. |
| **Aggregation Module** | `MVAggregate` | Fuses the multiple 768-dimensional feature vectors (one per view) into a single, comprehensive feature representation. |
| **Action Head** | `fc_action` (Linear Layers) | Predicts the probability distribution over the **8 Action Classes**. |
| **Card Head** | `fc_offence` (Linear Layers) | Predicts the probability distribution over the **4 Card Severity Classes**. |

### 2. Multi-View Aggregation Mechanisms

The `MVAggregate` module is critical for combining features from $V$ views into a unified feature vector.

| Aggregation Type | Implementation Class | Technical Mechanism |
| :--- | :--- | :--- |
| **Max Pooling** | `ViewMaxAggregate` | Selects the maximum feature value across all views for each of the 768 dimensions. This is effective when the most discriminative feature is present in a single view. |
| **Average Pooling** | `ViewAvgAggregate` | Computes the element-wise mean of the feature vectors across all views, providing a simple, robust feature fusion. |
| **Weighted Attention** | `WeightedAggregate` | Learns a dynamic attention matrix to assign weights to each view's contribution based on its relevance to the final prediction, using matrix multiplication (`torch.bmm`) for view-to-view interaction. |

### 3. Prediction Labels

The model is trained to classify actions into the following discrete categories:

| Category | Labels |
| :--- | :--- |
| **Card Severity** (4 Classes) | `No card`, `Yellow card`, `Red card`, `Yellow->Red card` |
| **Action Class** (8 Classes) | `Tackling`, `Standing tackling`, `High leg`, `Holding`, `Pushing`, `Elbowing`, `Challenge`, `Dive` |

## 📦 Data Pipeline and Preprocessing

The data pipeline is responsible for transforming raw SoccerNet annotations and video files into the multi-view tensor format required by the MViT model.

### 1. Annotation Parsing and Filtering

The `DataLoader.py` script processes the `annotations.json` file, applying a strict filtering logic to ensure data quality and relevance:

*   **Exclusion Criteria:** Actions with ambiguous or unknown labels (e.g., `Action class` as `''` or `"Dont know"`, or ambiguous `Offence`/`Severity` values) are systematically filtered out to maintain a clean training set.
*   **Composite Label Generation:** The final card severity label (`data['card']`) is a composite value derived from the `Severity` and `Offence` attributes, calculated as $\text{Severity} \times \text{Offence}$ (where $\text{Offence}$ is binarized to 1.0 or 0.0).

| Card Label Value | Source Attributes | Interpretation |
| :--- | :--- | :--- |
| **0.0** | $\text{Offence} = 0.0$ | No Offence |
| **1.0** | $\text{Severity} = 1.0, \text{Offence} = 1.0$ | Minor Offence (Warning/Yellow Card) |
| **3.0** | $\text{Severity} = 3.0, \text{Offence} = 1.0$ | Major Offence (Red Card) |
| **5.0** | $\text{Severity} = 5.0, \text{Offence} = 1.0$ | Severe Offence (Red Card) |

### 2. Video Frame Extraction and Normalization

The `frames_extract` function handles the conversion of video clips into the required tensor format:

1.  **Temporal Sampling:** **16 frames** are uniformly sampled from a fixed temporal window (frames **60 to 90**) of the video clip. This ensures a consistent input length ($T=16$) for the MViT model.
2.  **Spatial Transformation:** Frames are resized to ensure the smaller dimension is 256 pixels, followed by a **224x224** center crop. This standardizes the spatial input ($H=224, W=224$).
3.  **Normalization:** Frames are converted to RGB, scaled to the range $[0, 1]$, and then standardized using the **ImageNet mean and standard deviation**. This is crucial for leveraging the pre-trained weights of the MViT backbone.
4.  **Final Tensor Shape:** The output for a single view is a PyTorch tensor of shape **(C, T, H, W)**, or **(3, 16, 224, 224)**.

## 🛠️ Installation and Setup

### Prerequisites

The project requires a Python 3.x environment with the following key dependencies:

*   **Deep Learning:** PyTorch, Torchvision
*   **Video Processing:** OpenCV (`opencv-python`)
*   **GUI:** PyQt6
*   **Data Handling:** Pandas, NumPy

### Environment Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/FairPlayAI/FairPlayAI
    cd FairPlayAI
    ```

2.  **Install Dependencies:**
    ```bash
    # Note: PyTorch installation is platform-dependent. Consult the official PyTorch website for the correct command based on your OS and CUDA version.
    # pip install torch torchvision torchaudio
    
    # Install other required packages
    pip install pandas numpy opencv-python PyQt6
    ```

3.  **Data and Model Weights:**
    *   **Data:** The project is dependent on the **SoccerNet** dataset structure. Ensure the `annotations.json` and video clips are correctly organized.
    *   **Model Weights:** The trained model weights (`model_mvit.pth`) are required for inference. This file is not included in the repository and must be acquired separately. Place the weights file in the project's root directory.

4.  **Configuration:**
    *   Update the `TRAIN_PATH` variable in `config.py` to point to the root directory of your SoccerNet training data.
    ```python
    # config.py
    TRAIN_PATH = "/path/to/your/soccernet/data/train/" 
    ```

## 🚀 Usage and Inference

### Running the Desktop Application

The primary method for running inference is via the PyQt6 GUI:

```bash
python gui/app.py
```

The application will launch, allowing the user to select a video file, which is then processed by the loaded `MVNetwork` model.

### Programmatic Inference

The core prediction logic can be accessed directly through the `predict` function in `gui/output.py`:

```python
from gui.output import predict

# Specify the absolute path to the video clip for analysis
VIDEO_PATH = "/path/to/your/video.mp4" 

# The function returns a list of strings detailing the top card and action prediction with confidence scores.
results = predict(VIDEO_PATH) 
# Example output: ['Yellow card: 95.20%', 'Tackling: 88.50%']
```

## 📂 Project Structure Reference

| File/Directory | Description |
| :--- | :--- |
| `DataLoader.py` | Custom script for parsing `annotations.json`, applying filtering, and generating the composite card labels for training data preparation. |
| `dataset.py` | PyTorch `Dataset` implementation, handling multi-view clip selection, frame extraction, and tensor creation for model training. |
| `soccernet_loader.py` | An alternative, more comprehensive PyTorch `Dataset` and `DataLoader` implementation for the SoccerNet data, providing a robust data loading utility. |
| `config.py` | Global configuration file, used primarily to define the `TRAIN_PATH` for the dataset location. |
| `gui/` | Contains all source code for the PyQt6 desktop application. |
| `gui/app.py` | The main GUI application file, managing the user interface, video playback, and interaction with the prediction logic. |
| `gui/output.py` | Contains the complete model definition (`MVNetwork`), video preprocessing functions (`frames_extract`), model loading, and the `predict` inference function. |
| `model_mvit.pth` | Placeholder for the required trained model weights file. |

## 👥 Contributors

This project was developed by:

*   [AliNowia](https://github.com/AliNowia)
*   [Moaz715](https://github.com/Moaz715)
*   [LowkeyAhmed](https://github.com/LowkeyAhmed)
