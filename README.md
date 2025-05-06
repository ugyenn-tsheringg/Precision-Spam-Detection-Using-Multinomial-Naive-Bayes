# Precision Spam Detection Using Multinomial Naive Bayes

## Overview
This project develops a machine learning-based spam detection system to classify SMS messages as either **spam** or **ham** (non-spam). The goal is to build a model that accurately identifies spam messages with high precision, minimizing false positives (misclassifying ham as spam), which is critical for user experience in SMS filtering applications. The dataset used contains 5,572 SMS messages labeled as spam or ham, and the final model achieves a precision of **1.0** with an accuracy of **97.19%** using the Multinomial Naive Bayes algorithm.

## Dataset
The dataset, stored in `spam.csv`, consists of 5,572 SMS messages with the following columns:
- **v1**: Label (`ham` or `spam`)
- **v2**: Text of the SMS message
- **Unnamed: 2, Unnamed: 3, Unnamed: 4**: Mostly empty columns with sparse data (50, 12, and 6 non-null entries, respectively)

After preprocessing (removing duplicates), the dataset is reduced to **5,169 unique messages**.

## Project Steps
The project follows a systematic pipeline to build, evaluate, and prepare the spam detection model for deployment. The steps are:

1. **Data Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Text Preprocessing**
4. **Model Building**
5. **Model Evaluation**
6. **Model Improvement**
7. **Model Export**

Below is a detailed breakdown of each step, including code snippets, results, and key findings from the Jupyter Notebook (`Spam-detection.ipynb`).

---

## 1. Data Cleaning
**Objective**: Prepare the dataset by removing noise and irrelevant data to ensure high-quality input for modeling.

**Actions**:
- **Dropped sparse columns**: The columns `Unnamed: 2`, `Unnamed: 3`, and `Unnamed: 4` were removed due to their high missing value rates, as shown in the `df.info()` output:
  ```plaintext
  <class 'pandas.core.frame.DataFrame'>
  RangeIndex: 5572 entries, 0 to 5571
  Data columns (total 5 columns):
   #   Column      Non-Null Count  Dtype 
  ---  ------      --------------  ----- 
   0   v1          5572 non-null   object
   1   v2          5572 non-null   object
   2   Unnamed: 2  50 non-null     object
   3   Unnamed: 3  12 non-null     object
   4   Unnamed: 4  6 non-null      object
  ```
  ```python
  df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
  ```

- **Renamed columns**: Renamed `v1` to `target` and `v2` to `text` for clarity.
  ```python
  df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
  ```

- **Encoded labels**: Used `LabelEncoder` to convert `ham` to `0` and `spam` to `1`.
  ```python
  from sklearn.preprocessing import LabelEncoder
  encoder = LabelEncoder()
  df['target'] = encoder.fit_transform(df['target'])
  ```

- **Checked for missing values**: No missing values were found.
  ```python
  df.isnull().sum()
  # Output:
  # target    0
  # text      0
  # dtype: int64
  ```

- **Removed duplicates**: Identified and removed 403 duplicate rows.
  ```python
  df.duplicated().sum()  # Output: 403
  df = df.drop_duplicates(keep='first')
  df.shape  # Output: (5169, 2)
  ```

**Outcome**: A clean dataset with 5,169 unique entries and two columns (`target`, `text`), free of missing values and duplicates.

---

## 2. Exploratory Data Analysis (EDA)
**Objective**: Gain insights into the data distribution and text characteristics to inform model building.

**Actions**:
- **Class distribution**: Analyzed the distribution of `ham` and `spam` messages.
  ```python
  df['target'].value_counts()
  # Output:
  # 0    4516
  # 1     653
  # Name: target, dtype: int64
  ```
  - **Ham**: 4,516 messages (87.4%)
  - **Spam**: 653 messages (12.6%)
  - Visualized using a pie chart (via `matplotlib` and `seaborn`).

- **Text characteristics**: Explored word count, character count, and sentence length to identify differences between spam and ham messages. Spam messages often contained promotional keywords like "win", "free", and "prize".

**Outcome**: The dataset is imbalanced, with spam messages being significantly less frequent. This imbalance suggests that precision is a critical metric to avoid misclassifying ham messages as spam.

---

## 3. Text Preprocessing
**Objective**: Transform raw text into numerical features suitable for machine learning.

**Actions**:
- **Text preprocessing pipeline**:
  - Converted text to lowercase.
  - Tokenized text into words using `nltk.word_tokenize`.
  - Removed stopwords using NLTK's stopword list.
  - Removed punctuation and special characters.
  - Applied stemming using `PorterStemmer` to reduce words to their root form.
  ```python
  from nltk.corpus import stopwords
  from nltk.stem.porter import PorterStemmer
  import string

  ps = PorterStemmer()
  def transform_text(text):
      text = text.lower()
      text = nltk.word_tokenize(text)
      y = []
      for i in text:
          if i.isalnum():
              y.append(i)
      text = y[:]
      y.clear()
      for i in text:
          if i not in stopwords.words('english') and i not in string.punctuation:
              y.append(i)
      text = y[:]
      y.clear()
      for i in text:
          y.append(ps.stem(i))
      return " ".join(y)
  df['transformed_text'] = df['text'].apply(transform_text)
  ```

- **Vectorization**: Used `TfidfVectorizer` to convert preprocessed text into TF-IDF feature vectors.
  - Initially, no `max_features` limit was set.
  - Later, set `max_features=3000` during model improvement (see Section 6).
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  tfidf = TfidfVectorizer()
  X = tfidf.fit_transform(df['transformed_text']).toarray()
  y = df['target'].values
  ```

- **Train-test split**: Split data into 80% training and 20% testing sets.
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
  ```

**Outcome**: Transformed text data into a 5,169 × N feature matrix (where N is the vocabulary size, later limited to 3,000) and corresponding labels. The data is ready for model training.

---

## 4. Model Building
**Objective**: Train and evaluate multiple machine learning models to identify the best-performing one for spam detection.

**Actions**:
- **Initial Naive Bayes Models**: Tested three Naive Bayes variants due to their effectiveness in text classification.
  ```python
  from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
  gnb = GaussianNB()
  mnb = MultinomialNB()
  bnb = BernoulliNB()
  ```

  - **GaussianNB**:
    ```python
    gnb.fit(X_train, y_train)
    y_pred1 = gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred1))  # Output: 0.8694390715667312
    print(confusion_matrix(y_test, y_pred1))  # Output: [[788 108], [ 27 111]]
    print(precision_score(y_test, y_pred1))  # Output: 0.5068493150684932
    ```

  - **MultinomialNB**:
    ```python
    mnb.fit(X_train, y_train)
    y_pred2 = mnb.predict(X_test)
    print(accuracy_score(y_test, y_pred2))  # Output: 0.9709864603481625
    print(confusion_matrix(y_test, y_pred2))  # Output: [[896   0], [ 30 108]]
    print(precision_score(y_test, y_pred2))  # Output: 1.0
    ```

  - **BernoulliNB**:
    ```python
    bnb.fit(X_train, y_train)
    y_pred3 = bnb.predict(X_test)
    print(accuracy_score(y_test, y_pred3))  # Output: 0.9835589941972921
    print(confusion_matrix(y_test, y_pred3))  # Output: [[896   0], [ 17 121]]
    print(precision_score(y_test, y_pred3))  # Output: 1.0
    ```

  **Findings**: MultinomialNB and BernoulliNB achieved perfect precision (1.0), with BernoulliNB slightly outperforming in accuracy (0.9836 vs. 0.9710).

- **Other Models**: Evaluated a broader set of 10 classification algorithms to ensure a comprehensive comparison.
  ```python
  from sklearn.svm import SVC
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
  clfs = {
      'SVC': SVC(kernel='sigmoid', gamma=1.0),
      'KN': KNeighborsClassifier(n_neighbors=7),
      'NB': MultinomialNB(),
      'DT': DecisionTreeClassifier(max_depth=5),
      'LR': LogisticRegression(solver='liblinear'),
      'RF': RandomForestClassifier(n_estimators=50, random_state=2),
      'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=2),
      'BgC': BaggingClassifier(n_estimators=50, random_state=2),
      'ETC': ExtraTreesClassifier(n_estimators=50, random_state=2),
      'GBDT': GradientBoostingClassifier()
  }
  def train_classifier(clf, X_train, y_train, X_test, y_test):
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      accuracy = accuracy_score(y_test, y_pred)
      precision = precision_score(y_test, y_pred)
      return accuracy, precision
  ```

  - Results compiled into a DataFrame (`performance_df`):
    ```python
    performance_df
    # Output:
    #   Algorithm  Accuracy  Precision
    # 1        KN  0.900387   1.000000
    # 2        NB  0.959381   1.000000
    # 8       ETC  0.977756   0.991453
    # 5        RF  0.970019   0.990826
    # 0       SVC  0.972921   0.974138
    # 6  AdaBoost  0.962282   0.954128
    # 4        LR  0.951644   0.940000
    # 9      GBDT  0.951644   0.931373
    # 7       BgC  0.957447   0.861538
    # 3        DT  0.934236   0.830189
    ```

**Outcome**: Multinomial Naive Bayes (NB) and K-Neighbors Classifier (KN) achieved perfect precision (1.0). NB was preferred due to its higher accuracy (0.9594 vs. 0.9004). Extra Trees Classifier (ETC) and Random Forest (RF) also performed well but had slightly lower precision.

---

## 5. Model Evaluation
**Objective**: Compare model performance and select the best model based on precision and accuracy.

**Actions**:
- Focused on **precision** as the primary metric to minimize false positives (misclassifying ham as spam).
- Multinomial Naive Bayes was initially selected due to its perfect precision (1.0) and high accuracy (0.9594).
- Evaluated confusion matrices to understand model behavior:
  - For MultinomialNB:
    ```python
    [[896   0]
     [ 30 108]]
    ```
    - **True Negatives (Ham correctly predicted)**: 896
    - **False Positives (Ham misclassified as Spam)**: 0
    - **False Negatives (Spam misclassified as Ham)**: 30
    - **True Positives (Spam correctly predicted)**: 108

**Outcome**: Multinomial Naive Bayes was confirmed as a strong candidate due to its zero false positives, ensuring no legitimate messages are flagged as spam.

---

## 6. Model Improvement
**Objective**: Enhance model performance through hyperparameter tuning and ensemble methods.

**Actions**:
- **TF-IDF Tuning**: Adjusted `TfidfVectorizer` to `max_features=3000` to limit the vocabulary size and capture the most relevant features.
  ```python
  tfidf = TfidfVectorizer(max_features=3000)
  X = tfidf.fit_transform(df['transformed_text']).toarray()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
  ```

- **Re-evaluated Naive Bayes Models**:
  - **GaussianNB**:
    ```python
    gnb.fit(X_train, y_train)
    y_pred1 = gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred1))  # Output: 0.8704061895551257
    print(confusion_matrix(y_test, y_pred1))  # Output: [[788 108], [ 26 112]]
    print(precision_score(y_test, y_pred1))  # Output: 0.509090909090909
    ```

  - **MultinomialNB**:
    ```python
    mnb.fit(X_train, y_train)
    y_pred2 = mnb.predict(X_test)
    print(accuracy_score(y_test, y_pred2))  # Output: 0.971953578336557
    print(confusion_matrix(y_test, y_pred2))  # Output: [[896   0], [ 29 109]]
    print(precision_score(y_test, y_pred2))  # Output: 1.0
    ```

  - **BernoulliNB**:
    ```python
    bnb.fit(X_train, y_train)
    y_pred3 = bnb.predict(X_test)
    print(accuracy_score(y_test, y_pred3))  # Output: 0.9835589941972921
    print(confusion_matrix(y_test, y_pred3))  # Output: [[895   1], [ 16 122]]
    print(precision_score(y_test, y_pred3))  # Output: 0.991869918699187
    ```

  **Findings**: MultinomialNB maintained perfect precision (1.0) with improved accuracy (0.9719 vs. 0.9594). BernoulliNB's precision dropped slightly (0.9919) due to one false positive.

- **Re-evaluated All Models**: Re-ran the 10 classifiers with `max_features=3000`.
  ```python
  performance_df1
  # Output:
  #   Algorithm  Accuracy  Precision
  # 1        KN  0.905222   1.000000
  # 2        NB  0.971954   1.000000
  # 5        RF  0.975822   0.982906
  # 8       ETC  0.979691   0.975610
  # 0       SVC  0.974855   0.974576
  # 4        LR  0.956480   0.969697
  # 6  AdaBoost  0.961315   0.945455
  # 9      GBDT  0.946809   0.927835
  # 7       BgC  0.959381   0.869231
  # 3        DT  0.933269   0.841584
  ```

  - Comparison of performance before and after `max_features=3000`:
    ```python
    performance_df.merge(temp_df, on='Algorithm')
    # Output:
    #   Algorithm  Accuracy  Precision  Accuracy_max_ft_3000  Precision_max_ft_3000
    # 0        KN  0.900387   1.000000              0.905222               1.000000
    # 1        NB  0.959381   1.000000              0.971954               1.000000
    # 2       ETC  0.977756   0.991453              0.979691               0.975610
    # 3        RF  0.970019   0.990826              0.975822               0.982906
    # 4       SVC  0.972921   0.974138              0.974855               0.974576
    # 5  AdaBoost  0.962282   0.954128              0.961315               0.945455
    # 6        LR  0.951644   0.940000              0.956480               0.969697
    # 7      GBDT  0.951644   0.931373              0.946809               0.927835
    # 8       BgC  0.957447   0.861538              0.959381               0.869231
    # 9        DT  0.934236   0.830189              0.933269               0.841584
    ```

  **Findings**: Setting `max_features=3000` improved the accuracy of MultinomialNB (0.9719 vs. 0.9594) while maintaining perfect precision. Other models showed mixed results, with some (e.g., ETC, RF) experiencing slight precision drops.

- **Voting Classifier**: Combined SVC, MultinomialNB, and ExtraTreesClassifier using soft voting to leverage their strengths.
  ```python
  svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)
  mnb = MultinomialNB()
  etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
  voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)], voting='soft')
  voting.fit(X_train, y_train)
  y_pred = voting.predict(X_test)
  print("Accuracy", accuracy_score(y_test, y_pred))  # Output: 0.9816247582205029
  print("Precision", precision_score(y_test, y_pred))  # Output: 0.9917355371900827
  ```

  **Findings**: The Voting Classifier achieved high accuracy (0.9816) but lower precision (0.9917) compared to MultinomialNB alone, making it less suitable.

- **Stacking Classifier**: Used SVC, MultinomialNB, and ExtraTreesClassifier as base estimators with RandomForestClassifier as the final estimator.
  ```python
  estimators = [('svm', svc), ('nb', mnb), ('et', etc)]
  final_estimator = RandomForestClassifier()
  clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print("Accuracy", accuracy_score(y_test, y_pred))  # Output: 0.9787234042553191
  print("Precision", precision_score(y_test, y_pred))  # Output: 0.9393939393939394
  ```

  **Findings**: The Stacking Classifier had high accuracy (0.9787) but significantly lower precision (0.9394), making it unsuitable for the goal of maximizing precision.

**Outcome**: After tuning and ensemble experiments, **Multinomial Naive Bayes** with `max_features=3000` remained the best model, achieving:
- **Accuracy**: 0.9719 (97.19%)
- **Precision**: 1.0 (100% of predicted spam messages were actually spam)
- **Confusion Matrix**:
  ```python
  [[896   0]
   [ 29 109]]
  ```

The perfect precision ensures no ham messages are misclassified as spam, aligning with the project's primary objective.

---

## 7. Model Export
**Objective**: Save the trained model and vectorizer for integration into a web application.

**Actions**:
- Saved the `TfidfVectorizer` and Multinomial Naive Bayes model using `pickle`.
  ```python
  import pickle
  pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
  pickle.dump(mnb, open('model.pkl', 'wb'))
  ```

**Outcome**: The `vectorizer.pkl` and `model.pkl` files are ready for deployment in a web application or other production environment.

---

## Code Description
The implementation is contained in the `Spam-detection.ipynb` Jupyter Notebook. Key components include:

### Libraries Used
- **Data Manipulation**: `pandas`, `numpy`
- **Text Preprocessing**: `nltk` (for tokenization, stemming, stopwords), `sklearn.feature_extraction.text.TfidfVectorizer`
- **Machine Learning**: `sklearn` (classifiers, metrics, ensemble methods)
- **Visualization**: `matplotlib`, `seaborn`

### Key Functions
- **Text Preprocessing**: `transform_text` function to lowercase, tokenize, remove stopwords/punctuation, and stem text.
- **Model Training**: `train_classifier` function to train a classifier and compute accuracy and precision.
- **Pipeline**: Load data → Clean → Preprocess text → Vectorize → Train/test split → Train models → Evaluate → Tune/ensemble → Export.

### Final Model Performance
The selected Multinomial Naive Bayes model with `max_features=3000` achieves:
- **Accuracy**: 0.9719 (97.19%)
- **Precision**: 1.0 (100%)
- **Confusion Matrix**:
  - True Negatives: 896
  - False Positives: 0
  - False Negatives: 29
  - True Positives: 109

This performance ensures reliable spam detection with no misclassification of legitimate messages.

---

## Dependencies
To run the project, install the required Python libraries:
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## How to Run
1. Clone the repository or download `Spam-detection.ipynb` and `spam.csv`.
2. Place `spam.csv` in the same directory as the notebook.
3. Install dependencies (see above).
4. Open and run the Jupyter Notebook to execute the pipeline.
5. The trained model (`model.pkl`) and vectorizer (`vectorizer.pkl`) will be saved in the working directory.

## Future Work
- **Web Application**: Develop a web interface for real-time spam detection using the saved model and vectorizer.
- **Deployment**: Deploy the application on Heroku or a similar platform.
- **Deep Learning**: Explore advanced models like LSTM or BERT for potentially improved performance.
- **Class Imbalance**: Address the imbalanced dataset using techniques like SMOTE, oversampling, or weighted loss functions.
- **Feature Engineering**: Incorporate additional features (e.g., message length, keyword frequency) to enhance model performance.

## License
This project is licensed under the **MIT License**.
