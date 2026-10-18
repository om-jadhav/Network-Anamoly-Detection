# AI-Based Network Anomaly Detection

An AI/ML-based cybersecurity project that aims to analyze network traffic and identify potentially malicious activity using machine learning techniques.

The project uses the **CIC-IDS2017 dataset** to explore network traffic patterns, distinguish benign traffic from attacks, and develop a machine learning-based anomaly detection system.

## 🎯 Project Objectives

- Analyze real-world network traffic data.
- Understand the characteristics of benign and malicious traffic.
- Perform data cleaning and preprocessing.
- Develop machine learning models for detecting suspicious network activity.
- Evaluate model performance using appropriate classification metrics.
- Explore how anomaly detection can support network security monitoring.

## 🛠️ Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Cybersecurity Tools:** Wireshark (for network traffic analysis and learning)
- **Development Environment:** Visual Studio Code

## 📊 Dataset

This project uses the **CIC-IDS2017 dataset**, published by the Canadian Institute for Cybersecurity.

The dataset contains labeled network traffic flows, including benign activity and various types of attacks.

**Dataset source:** [CIC-IDS2017 — Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)

For the initial analysis, the project uses:

`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

This file contains benign traffic and DDoS attack records.

### Initial Dataset Analysis

| Metric | Result |
|---|---:|
| Total records | 225,745 |
| Total columns, including label | 79 |
| Benign traffic | 97,718 (43.29%) |
| DDoS traffic | 128,027 (56.71%) |
| Missing values in `Flow Bytes/s` | 4 |
| Infinite values in `Flow Bytes/s` | 30 |
| Infinite values in `Flow Packets/s` | 34 |
| Duplicate rows found within individual chunks | 1,870 |

*These results are from the initial inspection of the selected CSV file. Duplicate counts were checked within individual chunks, so duplicates spanning different chunks may not be included.*

## 📁 Project Structure

```text
Network-Anomaly-Detection/
│
├── data/
│   └── MachineLearningCSV/
│       └── (Dataset CSV files — downloaded separately)
│
├── src/
│   ├── inspect_data.py
│   └── analyze_data.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

The `data/` directory is excluded from Git because the dataset files are large. Download the dataset separately and place the required CSV files in the directory shown above.

## ⚙️ Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/om-jadhav/Network-Anamoly-Detection.git
cd Network Anamaloy Detection
```

Replace the placeholder URL with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download CIC-IDS2017 from the official dataset page:

https://www.unb.ca/cic/datasets/ids-2017.html

Extract the `MachineLearningCSV.zip` archive into:

```text
data/MachineLearningCSV/
```

### 5. Run the analysis

From the project's `src` directory, run:

```bash
python inspect_data.py
```

To analyze the dataset's label distribution, missing values, infinite values, and duplicate rows:

```bash
python analyze_data.py
```

## 🚧 Current Progress

- [x] Project setup and Python virtual environment
- [x] Dataset download and extraction
- [x] Initial dataset inspection
- [x] Label distribution analysis
- [x] Identification of missing and infinite values
- [x] Data preprocessing and cleaning
- [x] Exploratory data visualization
- [x] Machine learning model development
- [ ] Model evaluation and performance comparison
- [ ] Anomaly detection results and reporting

## 🧠 Planned Machine Learning Approach

The next development stages will focus on:

1. **Data preprocessing:** Handle missing values, infinite values, duplicates, and unsuitable features.
2. **Exploratory data analysis:** Understand traffic distributions and relationships between network features.
3. **Model development:** Explore Isolation Forest for unsupervised anomaly detection and Random Forest for supervised classification.
4. **Model evaluation:** Assess precision, recall, F1-score, confusion matrix, and other relevant metrics.
5. **Security analysis:** Examine the model's ability to identify suspicious traffic while minimizing false positives.

The final approach will be selected based on the dataset, experiments, and evaluation results.

## 🔐 Cybersecurity Relevance

Network anomaly detection can help identify unusual traffic patterns that may indicate denial-of-service attacks, scanning activity, or other suspicious behavior.

This project is intended for educational purposes and machine learning experimentation. Model predictions are not a substitute for a complete security monitoring system.

## 👨‍💻 Author

**Om Jadhav**

Computer Engineering | Development & Cybersecurity

GitHub: [Om-jadhav](https://github.com/Om-jadhav)

---

*This project is under active development. More preprocessing, machine learning, and evaluation features will be added as development progresses.*
