# AI-Based Human Posture Detection & Correction System

**SURF Research Project (`SURF-2026-0420`)** | *Supervisor: Dr. Gomesh Nair*

A non-invasive, real-time computer vision pipeline for monitoring human posture from side-view frames, evaluating joint alignment angles, and providing instant ergonomic feedback without wearable sensors.

---

## Key Features & Methodology
* **Privacy-Safe Synthetic Dataset (`posture_data/`):** Curated an AI-generated side-view image dataset (`good/` and `bad/` classes) to evaluate posture alignment without exposing real human biometric data.
* **Pose Estimation Framework Comparison:** Benchmarked **MediaPipe**, **YOLOv8** (baseline v1, ear fine-tune, full-body v2), and an abstention-based **Multi-Model Ensemble** across accuracy, F1-score, recall, and inference stability.
* **Biomechanical Angle Extraction:** Computes 4 key segment angles relative to a fixed vertical reference axis from right-side keypoints:
  * **ES (Ear-to-Shoulder):** Forward head tilt
  * **SH (Shoulder-to-Hip):** Torso lean
  * **HK (Hip-to-Knee):** Hip flexion
  * **KA (Knee-to-Ankle):** Knee bend
* **Real-Time Classification & Feedback:** Classifies posture using geometric thresholds ($20^\circ$ / $25^\circ$) and ML classifiers, overlaying live segment diagnostics (`Good Posture` vs. `Bad Posture`) and actionable correction advice.

---

## Key Research Findings
* **MediaPipe** outperformed YOLO-based variants on the held-out test set at the $25^\circ$ E-S threshold (**F1 = 0.73**, **Balanced Accuracy = 0.70**, **Recall = 0.80** vs. YOLO F1 = 0.29–0.40).
* A lightweight **$20^\circ$ geometric threshold** achieved **86.4% accuracy** and **86.2% F1-score**, confirming the viability of interpretable real-time vision pipelines for healthcare and workplace ergonomics.
* **Abstention Ensemble Design:** Implemented a confidence-based abstention mechanism (`Adaptive YOLO-pose` and `Multi-model ensemble`) that achieved **zero misclassifications** on answered samples.

**Research Poster:** [View Full SURF Poster (PDF)](docs/SURF_2026_0420_Poster.pdf)

---

## Repository Structure
```text
posture-detection-cv/
├── docs/                        # Research poster (SURF_2026_0420_Poster.pdf)
├── posture_data/                # AI-generated side-view posture dataset (good/ & bad/)
├── extract_features.py          # Keypoint extraction & joint angle calculation
├── save_reference.py            # Reference vertical axis calibration
├── train_classifier.py          # ML posture classifier training
├── evaluate.py                  # Cross-model evaluation & metrics (F1, Recall, Accuracy)
├── main_posture.py              # Real-time webcam/video posture detection & feedback
├── generate.py                  # Data generation & augmentation utilities
├── generate_report.py           # Automated Excel performance reporting
├── summary.py                   # Experiment summary aggregation
├── posture_features.csv         # Extracted biomechanical angle dataset
├── posture_classifier.pkl       # Trained posture classification model
├── reference_pose.npy           # Calibrated vertical reference coordinates
├── model_evaluation.xlsx        # Framework comparison metrics
├── posture_report.xlsx          # Detailed sample-level predictions
├── summary_report.xlsx          # Aggregated experiment statistics
└── requirements.txt             # Python dependencies