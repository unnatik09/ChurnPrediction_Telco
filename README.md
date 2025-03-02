# 📊 Churn Prediction for Telco Customers

This project predicts customer churn using **Neural Networks (NN) and XGBoost (XGB)**. The dataset is sourced from **Kaggle**, and preprocessing includes **Label Encoding** and **Standard Scaler**.

---

## 📂 Project Structure  

```
ChurnPrediction_Telco/
│-- data/                 # Raw & processed datasets
│-- models/               # Saved ML models
│-- notebooks/            # Jupyter notebooks for EDA & modeling
│-- src/                  # Python scripts for preprocessing & training
│-- final_code.ipynb      # Final Jupyter Notebook
│-- README.md             # Project documentation
│-- requirements.txt      # Required Python libraries
```

---

## 🚀 Features  

- **Data Preprocessing**
  - Label Encoding for categorical variables  
  - Standard Scaler for numerical features
  - Handling missing values  

- **Exploratory Data Analysis (EDA)**
  - Feature distribution and correlations  

- **Machine Learning Models**
  - XGBoost (XGB)  
  - Neural Networks (NN)  

- **Model Evaluation**
  - Accuracy, Precision, Recall, F1-score, AUC-ROC  

---

## 🛠️ Installation & Usage  

### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/your-username/ChurnPrediction_Telco.git
cd ChurnPrediction_Telco
```

### **2️⃣ Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **3️⃣ Run Jupyter Notebook**  
```sh
jupyter notebook
```
Open `final_code.ipynb` and execute the cells to train and evaluate the models.

---

## 📊 Dataset  

- **Source**: [Kaggle]([https://www.kaggle.com/](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))  
- **Description**: Contains customer details, account information, and churn status.  
- **Key Features**:  
  - `customerID` – Unique ID  
  - `gender` – Male or Female  
  - `tenure` – Duration of customer retention  
  - `Contract` – Subscription type  
  - `Churn` – Target variable (Yes/No)  

---

## 🤖 Machine Learning Models  

| Model  | Accuracy | Precision | Recall | F1-score |
|--------|----------|-----------|--------|----------|
| ENSEMBLE | 0.77 | 0.55 | 0.80 | 0.65 |

---

## 📝 License  

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments  

- [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- [XGBoost Docs](https://xgboost.readthedocs.io/)  
- [TensorFlow/PyTorch](https://www.tensorflow.org/)  

---

🔗 **GitHub Repository**: [ChurnPrediction_Telco](https://github.com/unnatik09/ChurnPrediction_Telco)  
📧 **Contact**: codeunnati09@gmail.com  

