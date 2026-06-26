# Smart Classroom Analytics System

A computer vision–based system for monitoring and analyzing student engagement in a smart classroom environment. The system processes live classroom video streams and performs multi-student face detection, tracking, recognition, emotion analysis, and engagement prediction through an efficient real-time pipeline.

---

## Features

* Multi-face detection
* Face tracking with persistent Track IDs
* Face recognition using **FaceNet + SVM + Cosine Similarity Verification**
* Emotion recognition using a **ResNet50 (PyTorch)** model
* Student engagement prediction using a **CNN + BiLSTM** sequence model
* Quality filtering to discard low-quality frames
* REST API built with **FastAPI**
* Interactive API documentation (Swagger UI)

---

## System Pipeline

```text
Camera
    │
    ▼
Quality Filter
    │
    ▼
Face Detection
    │
    ▼
Face Tracking
    │
    ▼
Face Recognition
    │
    ▼
Emotion Recognition
    │
    ▼
Engagement Recognition
    │
    ▼
Dashboard / API Response
```

---

## Technologies Used

* Python
* FastAPI
* PyTorch
* TorchVision
* FaceNet (facenet-pytorch)
* ResNet50
* RetinaFace
* OpenCV
* Scikit-learn (SVM)
* NumPy
* Pandas

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd student-engagement-system
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Run the API

```bash
uvicorn main:app --reload
```

The server will start at:

```text
http://127.0.0.1:8000
```

---

## API Documentation

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

---

## Project Structure

```text
student-engagement-system/
│
├── assets/
├── models/
├── routes/
├── services/
├── utils/
├── main.py
├── requirements.txt
└── README.md
```

---

## Project Overview

The system is designed for classroom analytics where a single camera monitors multiple students simultaneously. Each frame passes through a quality filtering stage before face detection and tracking. Every tracked face is recognized, analyzed for emotion, and accumulated into a fixed-length sequence for engagement prediction. The architecture follows a modular service-based design to improve maintainability, scalability, and real-time performance.

---

## License

This project was developed for educational and graduation project purposes.
