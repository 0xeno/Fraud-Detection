# 🛡️ FraudShield AI - Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**FraudShield AI** is an end-to-end machine learning project designed to detect fraudulent credit card transactions. It consists of two main components: a comprehensive **Jupyter Notebook** for data research (EDA, SMOTE, Model Evaluation) and an interactive **Streamlit Web App** for real-time inference and monitoring.

🔗 **[Live Demo App](https://fraud-detection-mgiztcxyg7asq5fkzmhyj2.streamlit.app/)** 
---

## 🚀 Key Features

### 1. 📊 Interactive Dashboard
* **Monitoring View:** Visualize fraud patterns based on categories, transaction amounts, and time distribution using interactive Plotly charts.

### 2. 🔍 Prediction Modes
* **Single Prediction:** Manual input form for simulating individual transactions. Includes dynamic data cleaning and feature engineering under the hood.
* **Batch Prediction:** Upload a CSV file to process hundreds of transactions at once. 
    * 🎛️ **Dynamic Filtering:** Toggle to view only fraud cases or all transactions without reloading the model (using Session State).
    * 📥 **Export Results:** Download the analyzed data (All or Fraud-only) directly as a CSV.

### 3. 🧠 Smart Backend Logic
* **Imbalance Handling:** The underlying XGBoost model is optimized with `scale_pos_weight` to aggressively detect minority fraud classes.
* **Auto-Sanitization:** The `model_service.py` automatically aligns user input columns with the strict formatting expected by XGBoost.

---
*(Note: For details on how the Machine Learning model was built, data cleaning, and evaluation, please visit the [notebooks directory](notebooks/).)*

---
## ⚙️ Installation & Local Setup

Follow these steps to set up and run the application locally on your machine:

**1. Clone the repository** Download a copy of this project to your local machine and navigate into the project directory using your terminal:
```bash
git clone [https://github.com/0xeno/fraud-detection.git](https://github.com/0xeno/fraud-detection.git)
cd fraud-detection
```
**2. Create a Virtual Environment** It is highly recommended to use a virtual environment to keep the project dependencies isolated from your main system:
```bash
python -m venv venv
```
After creating it, activate the virtual environment:

  **Windows**:
  ```bash 
  venv\Scripts\activate
  ```
  
  **Mac/Linux**:
  ```bash 
  source venv/bin/activate
```
**3. Install Dependencies** Once the virtual environment is active, install all the required Python libraries listed in the configuration file:
```bash
pip install -r requirements.txt
```
**4. Run the Streamlit App** Start the local web server to launch the interactive dashboard in your browser:
```bash
streamlit run app.py
```
---
## 📝 Usage Guide (Batch Prediction)

The Batch Prediction feature expects a CSV file with the following columns. You can generate a test file directly from the Colab notebook.

| Column Name | Description | Example |
| :--- | :--- | :--- |
| `amt` | Transaction Amount | `150.50` |
| `category` | Merchant Category | `food_dining` |
| `gender` | Gender (M/F) | `M` |
| `age` | Customer Age | `35` |
| `city_pop` | City Population | `20000` |
| `distance_KM` | Distance to Merchant | `5.2` |
| `state_encoded`| State Code (0-50) | `1` |
| `trans_date` | Transaction Date | `2024-01-31` |
| `trans_time` | Transaction Time | `14:30:00` |


## 📂 Project Structure

```text
├── 📂 notebooks/               # 🔬 ML Research & Experiments
│   ├── Fraud_Detection.ipynb   # Complete ML Pipeline Notebook
│   └── README.md               # Details about the notebook and data
│
├── App.py                      # Main Streamlit application
├── model_service.py            # Business Logic: Preprocessing & Model Inference
├── ui_pages.py                 # UI Components: Dashboard & Prediction Pages
├── requirements.txt            # Project dependencies
├── fraud_detection_XGBoost_Original.pkl # Trained Model Package
└── README.md                   # Project documentation
