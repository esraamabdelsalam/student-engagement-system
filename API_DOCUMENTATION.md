# Student Engagement System API Documentation

## Overview

The Student Engagement System provides a REST API for analyzing classroom images as part of a **Smart Classroom Analytics** system.

The API is designed to process classroom images that may contain multiple students. For each detected student, the system performs face recognition, emotion recognition, and engagement analysis.

The API returns one result object for every detected face in the uploaded image.

---

# API Summary

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | /detect | Analyze a classroom image and return recognition, emotion, and engagement information for all detected students. |

---

# Base URL

```
http://127.0.0.1:8000
```

---

# Endpoint

## Analyze Classroom Image

### HTTP Method

```
POST
```

### URL

```
/detect
```

---

# Description

This endpoint receives a classroom image and analyzes every visible student.

For each detected face, the system performs:

- Face Detection
- Face Recognition
- Emotion Recognition
- Engagement Buffer Update

The endpoint supports multiple students in a single image.

---

# Processing Pipeline

The uploaded image is processed using the following pipeline:

```
Input Image
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
Engagement Buffer Update
      │
      ▼
API Response
```

---

# Request

## Content-Type

```
multipart/form-data
```

## Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image | File | Yes | Classroom image containing one or more students. |

---

# Success Response

HTTP Status Code

```
200 OK
```

Returned when the image is successfully processed.

---

# Response Structure

```json
{
  "faces_detected": "<integer>",
  "results": [
    {
      "face_id": "<integer>",
      "student_name": "<string>",
      "svc_confidence": "<float>",
      "cosine_similarity": "<float>",
      "recognition_confidence": "<float>",
      "emotion": "<string>",
      "emotion_buffer_size": "<integer>",
      "engagement": "<string>",
      "engagement_confidence": "<float>"
    }
  ]
}
```

---

# Response Fields

## faces_detected

**Type**

```
Integer
```

**Description**

Total number of faces detected in the uploaded classroom image.

This value depends entirely on the input image.

---

## results

**Type**

```
Array
```

**Description**

Contains one object for every detected face.

The number of objects returned depends on the number of students detected in the uploaded image.

---

## face_id

**Type**

```
Integer
```

**Description**

Temporary identifier assigned to each detected face within the current request.

> **Note**
>
> Internally, the system uses face tracking to associate students across consecutive frames.
> The current API response returns a temporary `face_id` for each detected face.

---

## student_name

**Type**

```
String
```

**Description**

Name of the recognized student.

If the student cannot be recognized, the value may be:

```
Unknown
```

---

## svc_confidence

**Type**

```
Float
```

**Description**

Confidence score produced by the SVM classifier.

This value is returned for reference only.

It is **not** used for the final recognition decision.

---

## cosine_similarity

**Type**

```
Float
```

**Description**

Cosine similarity between the detected face embedding and the enrolled face embedding.

The final recognition decision is based on this score.

---

## recognition_confidence

**Type**

```
Float
```

**Description**

Final recognition confidence.

Currently,

```
recognition_confidence = cosine_similarity
```

---

## emotion

**Type**

```
String
```

**Description**

Predicted facial emotion for the detected student.

Possible values include:

- Happy
- Sad
- Angry
- Fear
- Surprise

The returned value depends on the current prediction.

---

## emotion_buffer_size

**Type**

```
Integer
```

**Description**

Number of emotion predictions currently collected for the detected student.

The emotion buffer is managed internally and cleared after each completed processing cycle.

---

## engagement

**Type**

```
String
```

**Description**

Current engagement prediction.

Possible values include:

```
Not Ready
```

or one of the engagement classes produced by the engagement model after enough frames have been collected.

---

## engagement_confidence

**Type**

```
Float
```

**Description**

Confidence score of the engagement prediction.

Returns

```
0
```

while the engagement model is waiting for a complete frame sequence.

---

# Recognition Pipeline

The face recognition module performs the following steps:

1. FaceNet Embedding Extraction
2. Feature Normalization
3. SVM Classification
4. Cosine Similarity Verification

The final recognition decision is based on the cosine similarity score.

The SVM confidence score is returned for reference only.

---

# Emotion Recognition

Emotion prediction is performed independently for every detected face.

Each prediction is temporarily stored in an internal emotion buffer.

After the processing cycle is completed, the system selects the most frequent predicted emotion (Mode Emotion) as the final emotion for that student.

The emotion buffer is then cleared automatically.

---

# Engagement Recognition

The engagement model is based on a sequence of frames.

Unlike emotion recognition, engagement **cannot be predicted from a single image**.

For every tracked student, the system continuously collects consecutive frames.

Once **16 frames** have been collected for the same tracked student:

- The engagement model performs one prediction.
- The engagement confidence is calculated.
- The engagement buffer is cleared.
- A new sequence begins.

Before collecting the required sequence, the API returns:

```
engagement = "Not Ready"
engagement_confidence = 0
```

---

# Error Responses

## 400 Bad Request

Returned when the uploaded image is missing or invalid.

Example

```json
{
    "detail": "Invalid image."
}
```

---

## 500 Internal Server Error

Returned when an unexpected server-side error occurs.

Example

```json
{
    "detail": "Internal server error."
}
```

---

# Notes

- The API supports multiple students in a single classroom image.
- The number of detected faces is dynamic.
- Student names are generated according to the face recognition result.
- Confidence scores are produced by the machine learning models and vary for every prediction.
- Emotion predictions are generated independently for every detected student.
- Engagement prediction is sequence-based and requires 16 consecutive frames for the same tracked student.
- Until the required sequence is completed, the engagement result is returned as `"Not Ready"`.
- The response structure remains constant, while the returned values depend on the uploaded image and the model predictions.

---

# Technologies

- PyTorch
- FaceNet (512-D Embeddings)
- Support Vector Machine (SVM)
- Cosine Similarity Verification
- ResNet50 (Emotion Recognition)
- CNN + BiLSTM (Engagement Recognition)
- FastAPI
