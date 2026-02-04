# Image-Based Animal Type Classification for Cattle and Buffaloes

An AI-powered system that automates **Animal Type Classification (ATC)** by identifying whether an animal is a **Cattle** or **Buffalo** from an image using **Computer Vision** and **Machine Learning**.

This project is inspired by the **Rashtriya Gokul Mission (RGM)**, where physical traits of bovine animals are manually evaluated. Our system replaces subjective human evaluation with a **standardized, image-driven, automated classification pipeline**.

---

## Project Vision

Manual ATC scoring faces major challenges:

- Subjective interpretation of physical traits
- Variability between evaluators
- Time-consuming field inspection
- Difficulty in maintaining consistent records

This system aims to:

✅ Automate classification using AI  
✅ Remove human bias from scoring  
✅ Provide real-time results for field officers  
✅ Store records in a centralized cloud database  
✅ Build a scalable system that can integrate with government platforms like **Bharat Pashudhan App**

---

## What This System Does

1. User uploads an image of the animal
2. Image goes through preprocessing (resize, normalization, noise removal)
3. CNN model classifies: **Cattle** or **Buffalo**
4. Backend stores result with confidence score in database
5. Frontend displays result and maintains history

---

## Full System Architecture

This project follows a **Modular Multi-Layer Architecture**:

| Layer | Technology | Responsibility |
|------|------------|----------------|
| Frontend | React.js | Image upload UI, dashboard, result display |
| Backend API | Flask / FastAPI | Connect UI with model and database |
| AI & CV | TensorFlow, OpenCV | Image preprocessing, CNN classification |
| Database | MongoDB | Store image records and results |
| Deployment | Docker, AWS, Nginx, CI/CD | Cloud hosting and scalability |

---

## 📁 Planned Repository Structure

```
animal-type-classification/
├── frontend/              # React frontend
├── backend/               # Flask / FastAPI APIs
├── model/                 # CNN model training and inference code
├── preprocessing/         # Image preprocessing scripts
├── database/              # MongoDB schemas and configurations
├── docker/                # Dockerfiles and deployment configurations
├── docs/                  # Diagrams, ER, DFD, architecture images
└── README.md
```



---

## Tech Stack

- **Machine Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, NumPy
- **Backend**: Flask / FastAPI
- **Frontend**: React.js, Axios
- **Database**: MongoDB
- **Deployment**: Docker, AWS EC2, Nginx, GitHub Actions

---

## Project Phases (Practical Execution)

| Phase | Work |
|------|------|
| Phase 1 | Dataset collection & preprocessing from Roboflow |
| Phase 2 | CNN model training for cattle vs buffalo |
| Phase 3 | Backend API integration with model |
| Phase 4 | Frontend UI for image upload and results |
| Phase 5 | Database integration (MongoDB) |
| Phase 6 | Dockerization and AWS deployment |

---

## Initial Goal (Current Stage)

At this starting stage, the focus is to:

- Prepare dataset
- Build preprocessing pipeline
- Train first CNN model
- Create basic API for prediction

---

## Functional Features (Target)

- Upload image (JPG/PNG)
- Image validation and preprocessing
- AI-based classification with confidence score
- Store results with timestamp
- Retrieve past classification records
- Dashboard for monitoring

---

## Non-Functional Goals

- Classification within 2–3 seconds
- 85–90% minimum model accuracy
- Mobile-friendly UI for field officers
- Secure HTTPS APIs
- Scalable cloud deployment

---

## Future Scope

This system lays the foundation for:

- Feature-level ATC scoring (rump angle, chest width, BCS)
- Integration with government livestock applications
- Breed quality analysis using stored data
- Large-scale rural deployment

---

## How to Start (Coming Soon)

Setup instructions for each module will be added as development progresses.

---

## License

This project is developed as part of a Capstone Project for academic purposes.
