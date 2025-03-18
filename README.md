# Fine-Tuning-LayoutLm-Model-on-FUNSD-Dataset-for-document-extraction
# Fine-Tuning LayoutLM for Document Understanding using Keras & Hugging Face Transformers

This project demonstrates how to fine-tune LayoutLM (v1) for document image understanding and information extraction using TensorFlow, Keras, and Hugging Face Transformers.  In this project, we use the FUNSD dataset—a collection of 199 fully annotated forms—to train, evaluate, and run inference on document images.

---

## Project Overview

- **Objective:**  
  Fine-tune LayoutLM to extract structured information (e.g., headers, questions, answers) from form images. The model learns to leverage both textual content and layout information (bounding boxes) for accurate document understanding.

- **Key Steps:**  
  1. **Setup Development Environment:**  
     Install necessary libraries including Hugging Face Transformers, Datasets, TensorFlow, and others.
  2. **Load and Prepare FUNSD Dataset:**  
     Use the Hugging Face Datasets library to load the FUNSD dataset, which contains 149 training and 50 test examples. Preprocess the images, OCR text (if needed), words, and bounding boxes into the required format.
  3. **Fine-Tune LayoutLM:**  
     Utilize the `TFLayoutLMForTokenClassification` class from Hugging Face Transformers to fine-tune the pre-trained `microsoft/layoutlm-base-uncased` model. Configure the model with the appropriate number of labels and the label mappings derived from the FUNSD dataset.
  4. **Training and Evaluation:**  
     Convert the processed dataset into TensorFlow datasets, define an optimizer (with optional mixed precision), and train the model using callbacks (TensorBoard, PushToHub, and metric evaluation callbacks). Evaluate the model using the seqeval metric and log metrics.
  5. **Inference and Visualization:**  
     Run inference on document images by processing the input image with a LayoutLM processor, predicting token labels, and drawing bounding boxes around detected elements.

- **Results:**  
  The fine-tuning process achieved an overall F1 score of approximately 0.75, demonstrating the power of transfer learning on a limited dataset.

---

## Requirements

- **Python:** 3.9 (recommended)
- **TensorFlow:** 2.10.0 or later
- **Hugging Face Libraries:** transformers, datasets, evaluate, seqeval, tensorboard, push-to-hub utilities
- **Additional Packages:** pytesseract (if OCR is required), Pillow, and others as listed in `requirements.txt`

---
