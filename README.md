# 🧾 **Resume Screening using NLP, Machine Learning & Deep Learning**

## 🔍 **Overview**
This project automates the process of **screening resumes** using **Natural Language Processing (NLP)**, **Machine Learning**, and **Deep Learning** techniques.  
It classifies resumes into relevant job categories such as **Data Science**, **HR**, **Quality Assurance**, etc., based on their text content.

By leveraging **TF-IDF**, **Naive Bayes**, and a **Deep Learning Neural Network**, the project provides a **data-driven solution** for HR departments to streamline and accelerate candidate shortlisting.

---

## 📊 **Project Accuracy**

### 🧠 **Machine Learning Model (Multinomial Naive Bayes)**
- **Training Accuracy:** ~98%  
- **Testing Accuracy:** ~96%  
- **Precision, Recall, F1-Score:** ~95–97%

### 🤖 **Deep Learning Model (Sequential Neural Network)**
- **Validation Accuracy:** ~94%  
- **Training Accuracy:** ~97%  
- **Model Configuration:**
  - Trained for **15 epochs**
  - **32 hidden neurons** with **ReLU activation**

---

## ⚙️ **Technologies Used**

| **Category**        | **Libraries / Tools** |
|----------------------|-----------------------|
| **Data Handling**    | pandas, numpy |
| **Visualization**    | matplotlib, seaborn |
| **NLP Processing**   | nltk *(stopwords, tokenization, lemmatization)* |
| **Machine Learning** | scikit-learn *(TfidfVectorizer, MultinomialNB, LabelEncoder, Metrics)* |
| **Deep Learning**    | keras *(Sequential, Dense, Flatten)* |
| **Language**         | Python 3.x |
| **Dataset**          | ResumeDataset.csv |

---

## 📚 **Steps Performed**

### 🪜 **Step 1: Importing Required Libraries**
- Imported all necessary Python, NLP, ML, and Deep Learning libraries.  
- Handled warnings for cleaner console output.

### 🪜 **Step 2: Viewing the Dataset**
- Displayed dataset head/tail, info, null values, duplicates, and category distributions.  
- Found that around **82% of records were duplicate**, but retained them to maintain data balance.

### 🪜 **Step 3: Merging Similar Categories**
- Combined similar categories for uniformity:
  - **Testing, Automation Testing → Quality Assurance**  
  - **Health and Fitness, Arts → Human Resources (HR)**
- Visualized category distribution using **count plots** and **pie charts**.

### 🪜 **Step 4: Data Cleaning**

#### 🔹 Part 1: Basic Cleaning
- Removed URLs, punctuation, and unwanted characters using **regex** and string operations.  
- Decoded text to remove unidentified letters.

#### 🔹 Part 2: NLP Cleaning
- Converted all text to **lowercase**.  
- Tokenized resumes using `word_tokenize`.  
- Removed **stopwords**.  
- Applied **WordNet Lemmatization** to reduce words to their root form.

### 🪜 **Step 5: Verifying Cleaned Data**
- Added **word count columns** before and after cleaning.  
- Verified cleaning through **pie chart comparison** (difference in word counts).

### 🪜 **Step 6: Encoding the Columns**
- Applied **TF-IDF Vectorization** to convert text into numerical vectors.  
- Encoded target categories using **LabelEncoder**.

### 🪜 **Step 7: Splitting the Dataset**
- Divided dataset into **training** and **testing** sets using `train_test_split`.

### 🪜 **Step 8: Machine Learning Model**
- Trained a **Multinomial Naive Bayes** model.  
- Evaluated performance using **accuracy**, **precision**, **recall**, and **F1-score**.

### 🪜 **Step 9: Deep Learning Model**
- Built a **Sequential Neural Network** using **Keras**.
- Visualized **accuracy** and **loss** trends using **line charts**.  
- Evaluated model performance on **validation data**.

---

## 🌍 **Scope & Real-World Usefulness**

### ✅ **Current Applications**
- **HR departments** can automatically categorize resumes, saving hours of manual review.  
- **Recruitment portals** can integrate it to instantly match candidates with suitable job roles.  
- **Companies** can use it to **pre-screen applicants** before human evaluation.

### 🚀 **Future Enhancements**
- Integrate **Word2Vec** or **BERT embeddings** for advanced semantic understanding.  
- Develop a **Django** or **Flask web interface** for real-time resume uploads and classification.  
- Connect with **Applicant Tracking Systems (ATS)** for large-scale hiring automation.  
- Extend to **multi-label classification** for resumes that fit multiple roles.

---


