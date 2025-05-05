# 🍎 Apple Analysis WeChat Mini Program

This WeChat Mini Program utilizes an ONNX model to detect the ripeness and identify rot in apples.

## ✨ Features

*   📱 **Intuitive Interface:** Three main screens: Home, Camera Capture, and Image Upload.
*   📷 **Real-time Analysis:** Capture apple images directly using the device camera for immediate analysis.
*   🖼️ **Upload Analysis:** Select apple images from the device's photo album for analysis.
*   🧠 **AI-Powered Detection:** Employs an ONNX model to classify apple ripeness and detect rot.
*   📊 **Visual Results:** Displays detection outcomes, including bounding boxes around detected apples and confidence scores for each classification.

## 🍏 Model Categories

The model can classify apples into the following categories:

*   100% Ripeness
*   75% Ripeness
*   50% Ripeness
*   Rotten Apple

## 🛠️ Technology Stack

*   **Frontend:** Native WeChat Mini Program Development (WXML, WXSS, JavaScript)
*   **Image Processing:** Utilizes the `<canvas>` element for image scaling and format conversion before model inference.
*   **Model Inference:** Leverages ONNX.js to run the AI model directly within the Mini Program environment.
*   **Result Visualization:** Draws bounding boxes and displays classification results overlaid on the analyzed image.

## 🚀 Getting Started

1.  Clone or download the project repository.
2.  Open WeChat DevTools.
3.  Click on "Import Project" and select the project folder.
4.  Ensure the ONNX model file (`best.onnx`) is present in the correct directory (see Project Structure below).
5.  Click "Compile" or "Preview" to run the Mini Program.

## 📝 Developer Notes

*   **Model Customization:** You might need to adjust the model loading and post-processing logic in `utils/modelService.js` depending on your specific model or requirements.
*   **Preprocessing Tuning:** The image preprocessing parameters in `utils/onnxUtils.js` may need tweaking for optimal performance with different models or image sources.
*   **Deployment:** For use on actual devices, especially if performance is critical or the model is large, consider deploying the model to a cloud backend and calling it via an API instead of running it client-side. The included `backend/assets/flask_inference.py` provides a starting point for a Flask-based backend.

## 📁 Project Structure

```plaintext
.
├── app.js                      # App entry point logic
├── app.json                    # App configuration (pages, window, etc.)
├── app.wxss                    # Global app styles
├── backend/
│   └── assets/                 # Backend assets (including model and scripts)
│       ├── best.onnx           # ONNX model file
│       ├── flask_inference.py  # Flask backend inference script example
│       └── ...                 # Other backend related files (inference helpers, etc.)
├── data.yaml                   # Likely model or configuration data
├── pages/                      # Mini Program pages
│   ├── index/                  # Home page
│   ├── camera/                 # Camera capture page (check if 'realtime' is used instead/also)
│   ├── upload/                 # Image upload page
│   └── realtime/               # Real-time analysis page (verify usage)
│   └── logs/                   # Log viewing page (standard utility)
├── project.config.json         # Developer tool project configuration
├── project.private.config.json # Developer tool private configuration
├── sitemap.json                # Sitemap configuration for search indexing
└── utils/                      # Utility functions
    ├── modelService.js         # Service for loading and running the model
    ├── onnxUtils.js            # Utility functions for ONNX operations and image preprocessing
    └── util.js                 # General utility functions (e.g., date formatting)
```

*Note: The exact purpose of `data.yaml` and the distinction between `camera/` and `realtime/` pages might require inspecting the code.*