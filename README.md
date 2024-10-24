# Spam_Detection_Project
Abstract
This paper presents a comparative study of two Natural Language Processing (NLP) approaches — Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) — 
to detect spam messages using machine learning algorithms. I applied Random Forest and Decision Tree classifiers to the preprocessed datasets derived from BoW and TF-IDF features, respectively. 
The models are evaluated based on accuracy and confusion matrices, with promising results indicating the efficiency of the proposed methods in classifying spam messages.

Keywords: Spam detection, Bag-of-Words, TF-IDF, Random Forest, Decision Tree, Natural Language Processing (NLP), Machine Learning.

1. Introduction
With the rapid growth of electronic communication, unsolicited spam messages have become an
increasing problem, leading to financial losses and privacy concerns.
Spam detection has emerged as a crucial task to ensure a safer online environment.
Various techniques have been developed to automatically classify messages as spam or ham (non-spam).
In this study, I explore two popular NLP techniques — Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) — in conjunction with machine learning algorithms to classify spam messages.

3. Related Work
Spam detection has been a widely researched area with various machine learning approaches.
Previous works have utilized Naive Bayes, Support Vector Machines (SVM), and deep learning methods.
However, traditional approaches such as Random Forest and Decision Tree classifiers remain effective for smaller datasets.

4. Dataset and Preprocessing
The dataset used is the “SMS Spam Collection” dataset from Kaggle. This dataset is commonly used for text classification tasks and contains
a collection of SMS messages labeled as either “ham” (not spam) or “spam.” The dataset was loaded into a Pandas DataFrame for preprocessing.
The columns were renamed, and unnecessary columns were removed. The label “ham” was replaced with “0” and “spam” with “1.”

3.1 Data Cleaning
For text preprocessing, I applied tokenization, removal of stopwords (excluding “not”), 
and stemming using the PorterStemmer from the Natural Language Toolkit (NLTK). The cleaned text was stored in a list named corpus.

corpus = []
for i in range(0, len(spam)):
    text = re.sub("[^a-zA-Z]", " ", spam["text"][i])
    text = text.lower().split()
    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if word not in set(all_stopwords)]
    corpus.append(" ".join(text))
4. Methodology
4.1 Bag-of-Words Model
Using the Bag-of-Words approach, I converted the cleaned text into a numerical representation by extracting the word frequencies. 
I utilized bi-grams and a maximum feature size of 2500 to ensure the model captures key patterns

cv = CountVectorizer(max_features=2500, binary=True, ngram_range=(2, 2))
X = cv.fit_transform(corpus).toarray()
4.2 Random Forest Classifier
The transformed dataset was split into training and testing sets (80–20 split). I then trained a Random Forest classifier on the training set.

rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
4.3 TF-IDF Model
In addition to BoW, I utilized TF-IDF to convert the text data into feature vectors. Similarly, I applied bi-grams and used 2500 features.

tf = TfidfVectorizer(max_features=2500, binary=True, ngram_range=(2, 2))
X = tf.fit_transform(corpus).toarray()
4.4 Decision Tree Classifier
For the TF-IDF approach, I employed a Decision Tree classifier with entropy as the criterion. The model was trained and evaluated similarly to the Random Forest classifier.

dt = DecisionTreeClassifier(random_state=42, criterion="entropy")
dt.fit(X_train, y_train)
5. Results
The performance of both classifiers was evaluated using a confusion matrix and accuracy score.

5.1 Random Forest Classifier (BoW)
The Random Forest classifier achieved an accuracy of 96.32%, with the following confusion matrix:

print(confusion_matrix(y_test, y_pred))
5.2 Decision Tree Classifier (TF-IDF)
The Decision Tree classifier using TF-IDF yielded an accuracy of 96.14%, with the confusion matrix shown below:

print(confusion_matrix(y_test, y_pred))
6. Discussion
The results demonstrate that both Bag-of-Words and TF-IDF approaches, combined with Random Forest and Decision Tree classifiers, 
effectively detect spam messages. Random Forest outperformed Decision Tree in terms of overall accuracy, 
likely due to its ability to handle a higher degree of model complexity. Future work could include more sophisticated models such as 
neural networks and deep learning architectures to improve performance further.

7. Conclusion
In this study, I compared two common text representation techniques — Bag-of-Words and TF-IDF — for spam detection using Random Forest and Decision Tree classifiers.
Both models showed promising results in terms of classification accuracy. However,
further improvements could be achieved by exploring additional feature engineering methods and more advanced classifiers.


References
Vapnik, V., The Nature of Statistical Learning Theory, Springer-Verlag, 1995.
McCallum, A., & Nigam, K., “A Comparison of Event Models for Naive Bayes Text Classification,” AAAI-98 Workshop on Learning for Text Categorization, 1998.
Zhang, Y., “Spam Detection Using Machine Learning Methods: A Survey,” IEEE Transactions on Knowledge and Data Engineering, 2021.
