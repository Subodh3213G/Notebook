# Data Science & Machine Learning Portfolio

A comprehensive collection of projects covering **Government Data Pipelines**, **Environmental Sustainability**, and **Advanced Computer Vision**. This repository demonstrates the practical application of machine learning to solve real-world problems.

---

## 🚀 Projects Overview

### 1. 🆔 UIDAI Aadhar Data Pipeline
* **File:** `UIDAI.ipynb`
* **Problem:** Government records often contain messy and non-standardized district/state names.
* **Solution:** Built a data cleaning pipeline using **Levenshtein fuzzy matching** to standardize over 900 name variations.
* **Impact:** Used **K-Means Clustering** to profile districts based on biometric update needs, leading to a strategy for deploying "Mobile Update Vans".

### 2. ♻️ Garbage Classification (Deep Learning)
* **Files:** `garbage_class.ipynb`, `train-model-garbage-classification.ipynb`, `regconize-anything-model-finetuned.ipynb`
* **Summary:** A deep learning project to automate waste sorting for recycling.
* **Technical Details:**
    * **Classes:** Cardboard, glass, metal, paper, plastic, and trash.
    * **Models:** Utilizes **Convolutional Neural Networks (CNNs)** and fine-tuned the **RAM (Recognize Anything Model)** for multi-tagging and image captioning.
    * **Hardware:** Trained using **NVIDIA Tesla T4 GPUs**.

### 3. 🌿 Green Tech & Environmental Analysis
* **Files:** `1. k-means clustering.ipynb`, `3. Logistic regression_.ipynb`, `4. Decision Trees.ipynb`, `7. k-nearest neighbors (KNN).ipynb`
* **Summary:** Predictive modeling to encourage the adoption of renewable energy and sustainable practices.
* **Key Algorithms:**
    * **K-Means:** Segmenting regions based on pollution levels and carbon emissions.
    * **Logistic Regression:** Predicting project sustainability based on energy output and cost efficiency.
    * **Decision Trees & KNN:** Modeling the adoption rates of renewable energy and effective emission reduction strategies.

---

## 🛠️ Technical Toolkit

| Category | Tools & Libraries |
| :--- | :--- |
| **Languages** | Python (3.x) |
| **Data Science** | Pandas, NumPy, Scikit-Learn |
| **Deep Learning** | PyTorch, TensorFlow, TIMM (Torch Image Models) |
| **Visualization** | Matplotlib, Seaborn |
| **Concepts** | Clustering, Regression, Decision Trees, CNNs, Fuzzy Matching |

---

## 💻 How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    ```
2.  **Install Requirements:**
    ```bash
    pip install pandas scikit-learn matplotlib seaborn torch torchvision tensorflow
    ```
3.  **Execute:** Open the `.ipynb` files in **Google Colab** (recommended for GPU support) or **Jupyter Lab** to view the results and training logs.

---

## 💡 Real-World Applications

* **Smart Cities:** Automated waste management systems using the Garbage Classification model.
* **Public Policy:** Data-driven resource allocation (like Mobile Aadhar Vans) based on demographic clustering.
* **Corporate ESG:** Helping companies decide on renewable energy investments by predicting sustainability outcomes.

---

*Note: This portfolio is a reflection of data-driven decision-making. By using facts and statistical models, we can replace intuition with high-confidence strategic planning.*
