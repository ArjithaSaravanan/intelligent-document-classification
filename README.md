# Intelligent-document-classification
A simple deep learning project that classifies business documents such as invoices and receipts using a Convolutional Neural Network (CNN) and exposes predictions via a REST API

# Why I built this
In real-world applications, documents like invoices, fuel bills and receipts come in many formats. Before extracting any useful information, they first need to be classified correctly.
I built this project to understand:
- how image-based document classification works
- how to use deep learning (CNNs) with small datasets
- how to turn a trained model into a usable API

# Project Overview
This project builds an end-to-end pipeline to:
1. Convert PDF documents into images
2. Train a CNN model to classify document types
3. Returns predictions with confidence scored for new unseen documents
4. Expose the model via a FastAPI REST API
Example output:
```json
{
    "file_name": "sample_invoice_001.pdf",
    "document_type": "fuel_receipt",
    "confidence": 0.74
}
```
# How I approached it
# 1. Data Preparation
- Collected a small set of sample invoices and receipts
- Converted PDFs into images using `pdf2image`
- Organized them into folders by class:
    - financial_invoice
    - fuel_receipt
    - hospitality_receipt
    - retail_receipt

# 2. Model
- Used **ResNet18 (pretrained)** from PyTorch
- Applied **transfer learning**
- Initially tranined the full model, but observed overfitting
- Improved results by **freezing most layers and training only the final layer**

# 3. Training
- Split the data into **train (80%) and validation (20%)**
- Added **basic augmentation** (rotation, brightness changes)
- Tracked both train and validation loss
This helped me understand:
  - how overfitting happens
  - why small datasets are challenging

# 4. Inference
- It supports both
  - images(.jpg, .png)
  - PDFs (auto-converted to image)
- Returns predicted class along with confidence score

# 5. API (FastAPI)
I exposed the model as a REST API so it can be used like a real service.
**Endpoint**
```http
POST /v1/documents/classify
```
# Tech Stack
- Python
- PyTorch
- torchvision
- FastAPI
- pdf2image
- Pillow

# Limitations
- very small dataset (~20 samples)
- some document types look visually similar
- Model performance on unseen data is limited

# What I learned
This project helped me understand:
- how to build an end-to-end ML pipeline
- how CNNs work for image classification
- how transfer learning helps with small datasets
- how to detect and reduce overfitting
- how to expose ML models via APIs

# What I would improve next
- collect more training data
- refine class definitions
- add proper evaluation metrics (accuracy, confusion matrix)
- integrate this with my invoice extraction pipeline
# Prerequisites
Before running project, make sure the following are installed:
# 1. Python
- Python 3.9+ recomended
# 2. Poppler (Required for PDF processing)
This project used `pdf2image` to convert PDFs into images.
For this to work, **Poppler must be installed on your system**
# Windows
1. Download Poppler from:
   [Poppler-Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
2. Extract the folder
3. Add the `bin` folder to your system PATH
   Example:
   ```
   C:\poppler\Library\bin
   ```
# macOS
```bash
brew install poppler
```
# Linus (Ubuntu)
```
sudo apt-get install poppler-utils
```
# How to run
```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```
Open:
```
http://127.0.0.1:8000/docs
```



   

