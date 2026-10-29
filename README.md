# Skin Disease Detection & Classification Using CNN

## Project Overview

Skin Disease Detection & Classification Using CNN is a machine learning-based web application designed to classify skin lesion images using a Convolutional Neural Network (CNN).

The application provides a web interface where users can register, log in, upload skin images, and view prediction results. It combines deep learning with web development to demonstrate how AI can support skin disease classification.

**Note:** This project is intended for educational and research purposes only. Its predictions are not a substitute for professional medical diagnosis.

## 🎯 Objectives

* Develop a deep learning model for skin lesion image classification.
* Integrate the trained model into a Django web application.
* Provide a simple interface for submitting images and viewing predictions.
* Connect the application to a MySQL database for user registration and related data.
* Explore the practical application of CNNs in medical image analysis.

## Features

* User registration and login.
* Image upload for skin disease classification.
* CNN-based image prediction.
* Web interface built with Django templates and CSS.
* MySQL database integration.
* Prediction result display.
* Sample images for testing.

##  Technologies Used

| Technology         | Purpose                                           |
| ------------------ | ------------------------------------------------- |
| Python             | Core programming language                         |
| Django             | Web application framework                         |
| TensorFlow / Keras | Deep learning model development and inference     |
| CNN                | Image classification                              |
| OpenCV             | Image processing                                  |
| NumPy              | Numerical operations                              |
| Scikit-learn       | Machine learning utilities and evaluation metrics |
| Matplotlib         | Data visualization                                |
| Seaborn            | Visualization and analysis                        |
| MySQL              | Database management                               |
| HTML and CSS       | Front-end interface                               |
| Git and GitHub     | Version control and project hosting               |

##  Disease Classification

The project model includes the following nine classes:

1. Actinic Keratosis
2. Basal Cell Carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented Benign Keratosis
7. Seborrheic Keratosis
8. Squamous Cell Carcinoma
9. Vascular Lesion

Classification results depend on the trained model, the input image, and the quality and limitations of the dataset.

##  Project Structure

```text
SkinDisease/
│
├── .env.example
├── .gitignore
├── README.md
├── requirements-current.txt
│
├── SkinDisease/
│   ├── manage.py
│   ├── DB.txt
│   ├── requirements.txt
│   ├── run.bat
│   │
│   ├── SkinDisease/
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   │
│   ├── SkinDiseaseApp/
│   │   ├── views.py
│   │   ├── models.py
│   │   ├── urls.py
│   │   ├── templates/
│   │   ├── static/
│   │   └── model/
│   │
│   ├── model/
│   └── testImages/
│
└── skdd doc/
```

##  Installation and Setup

### Prerequisites

Install the following before running the project:

* Python
* MySQL Server
* Git

### 1. Clone the Repository

```bash
git clone https://github.com/govardhanv-ai/skin-disease-detection-cnn.git
```

### 2. Navigate to the Django Project

```bash
cd skin-disease-detection-cnn/SkinDisease
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

From the outer repository folder, install the dependencies listed in the current environment's requirements file:

```bash
pip install -r ../requirements-current.txt
```

If a package is missing when you run the application, install or update the corresponding dependency as needed.

### 5. Configure the MySQL Database

Ensure MySQL Server is running.

Open `DB.txt` to view the SQL statements for creating the database and registration table.

Execute those SQL statements in your MySQL client.

The database connection settings must match your local MySQL configuration. Review the connection function in `SkinDiseaseApp/views.py` and configure the credentials securely before running the application.

### 6. Run the Django Application

From the Django project directory containing `manage.py`, execute:

```bash
python manage.py runserver
```

Open the local development address in your browser:

```text
http://127.0.0.1:8000/
```

Use the route configured in the Django URL files if the root address does not open the expected page.

##  Testing

The repository includes sample images in the `testImages/` directory.

You can use these images to explore the application's image-upload and prediction workflow.

The actual classification output depends on the model and the supplied image.

##  Future Enhancements

* Improve model performance using a larger and more diverse dataset.
* Add model evaluation results, including precision, recall, F1-score, and a confusion matrix.
* Improve image preprocessing and input validation.
* Enhance the user interface and overall user experience.
* Add stronger authentication and secure password storage.
* Deploy the application to a cloud hosting platform.
* Add automated tests for prediction and database functionality.

##  Author

**Govardhan Vallapu**

GitHub: [govardhanv-ai](https://github.com/govardhanv-ai)

##  Disclaimer

This application is an academic project developed for learning and demonstration purposes. It is not a certified medical diagnostic system. Predictions should not be used to make medical decisions without consultation with a qualified healthcare professional.
