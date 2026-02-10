# AI Multi-Tool Platform 🚀

A comprehensive, state-of-the-art AI dashboard that brings together multiple high-performance machine learning and deep learning models for vision, NLP, and regression tasks.

## 🌟 Key Features

### 🖼️ Computer Vision
- **Pneumonia Detector (X-RAY)**: Detects signs of Pneumonia in chest X-ray images with Grad-CAM visual heatmaps.
- **Cartoonify**: Transforms your photos into vibrant, black-outlined cartoon art.
- **City Image Segmentation**: Uses SegFormer (SegFormer-B5) to segment objects in city scene images.
- **Background Removal**: Instantly removes backgrounds using the RMBG-1.4 model.

### ✍️ Natural Language Processing
- **SMS Spam Detection**: Classifies messages as Spam or Ham using TF-IDF vectorization and machine learning.
- **News Classifier**: Categorizes news headlines into Business, Sci/Tech, Sports, or World categories using a Bi-LSTM model.

### 📊 Predictive Analytics (Regression)
- **House Pricing Predictor**: Estimates property values based on area, rooms, and amenities.
- **Insurance Predictor**: Predicts medical insurance premiums based on personal demographic data.
- **Student Performance**: Predicts academic scores based on study hours.

## 🛠️ Technology Stack
- **Backend**: FastAPI (Python)
- **Frontend**: HTML5, Jinja2, Vanilla CSS (Modern Glassmorphism Design)
- **Deep Learning Libraries**: PyTorch, TensorFlow, Transformers (Hugging Face)
- **Computer Vision**: OpenCV, Pillow
- **Data Science**: Scikit-Learn, Pandas, NumPy

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- ffmpeg (required for internal media processing)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd ai-multi-tool-platform
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # On Windows: 
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Setup**:
   Create a `.env` file for any required tokens (e.g., Hugging Face Token if needed for specific models).

5. **Run the Application**:
   ```bash
   python main.py
   ```
   Open your browser at `http://localhost:8000`.

## 📁 Project Structure
- `main.py`: The core FastAPI application logic and model loading.
- `templates/`: Jinja2 HTML templates for the dashboard and individual tools.
- `static/`: CSS, static assets, and temporary upload storage.
- `requirements.txt`: Python package dependencies.
- `[Project Folders]`: Directories containing saved `.pkl`, `.keras`, and `.h5` models.

## 📄 License
This project is licensed under the MIT License.

---
Developed with ❤️ by [Shahla12](https://github.com/Shahla12)
