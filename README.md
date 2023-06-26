# Machine-Learning-based-water-borne-disease-prediction-model-
In this project, machine learning techniques are used to predict the positive instances of
waterborne diseases. The base paper compares the performance of different machine learning
algorithms for predicting waterborne disease. The proposed architecture includes data preprocessing, feature selection, and machine learning-based classification. The dataset used in
the base paper contains 19 attributes. The Typhoid dataset has 68624 entries and malaria has
22,916 entries.

**1.1.Data pre-processing:**

The pre-processing procedures are applied to both the typhoid and malaria dataset. The preprocessing procedures include Data cleaning, Data balancing, Data transformation, and Data
normalization.
Data cleaning is done by the median imputation approach. The original dataset had many
missing values, and those null values are filled by median values. When the distribution is
skewed, median imputation is favored because the median is less likely to be affected by
outliers than the mean.
As the class distribution is highly imbalanced, data balancing is performed by random
oversampling and under-sampling methods. These sampling methods are used to balance the
minority and majority classes.
The dataset contains both categorical and numerical values. But most of the machine learning
models handle only numerical data. So, data transformation is performed by using one-hot
encoding and label encoding approaches to convert the categorical data into numerical data.
Data normalization is performed by using the z-score method to transform the features to be on 
a similar scale. It is the process of organizing data in a structured manner by applying certain 
techniques to ensure that data is consistent, accurate, and standardized.

**1.2.Feature Selection:**

The original dataset contains 19 features, out of which the top 7 features are selected by feature
selection. Feature selection is performed to select the most suitable features to train the model.
It also improves the accuracy of the models. By focusing on the most important factors and
removing the redundant and irrelevant ones, it also improves the algorithm’s ability to predict
results. In the base paper, it is performed using the random forest feature selection method. It
is a popular method for feature selection because it is effective at identifying the most relevant
features while also being relatively computationally efficient. RandomForest feature selection
is performed by training a random forest model on the entire dataset and then the importance of
every feature isranked based on the feature importance scores produced bythe model.

**1.3.Modelselection:**

The dataset is tested and trained by various machine learning models such as RF, DT, KNN,
logistic regression and SVM using ten cross-validation methods using Sklearn python library.
These algorithms are easily explainable, interpretable, implemented and used in many fields
with good performance, such as education and medicine. Furthermore, compared to artificial
neural networks and other DL methods, these algorithms use fewer data and demand less
computing power to train. The model is trained and tested using 10-fold cross-validation in
order to boost its capacity for generalization to the test data. Among this classifier, the random
forest classifier algorithm provides the best result for both datasets. In conclusion, the paper
presents a comparative study of various machine learning algorithms for waterborne disease
