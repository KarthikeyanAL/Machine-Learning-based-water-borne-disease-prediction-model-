Random forest classifier:
# Train and evaluate the model using KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracy_scores = []
y_preds=[]
y_tests=[]
for train_index, test_index in kf.split(X):
# Split data into train and test sets for this fold
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]
# Create and train the random forest classifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
# Make predictions on the test set and calculate accuracy
y_pred = rfc.predict(X_test)
y_preds.extend(y_pred)
y_tests.extend(y_test)
accuracy = accuracy_score(y_test, y_pred)
# Add accuracy score to list
accuracy_scores.append(accuracy)
# Compute and print the mean accuracy score and standard deviation
print("Accuracy- Random forest classifier: %0.2f (+/- %0.2f)" % (np.mean(accuracy_scores),
np.std(accuracy_scores) * 2))
Confusion matrix and ROC curve RF:
# Assuming the true and predicted labels are stored in y_true and y_pred respectively
cm= confusion_matrix(y_tests, y_preds)
# Create a heatmap of the confusion matrix using Seaborn
sp.heatmap(cm, annot=True, cmap="Blues", fmt = 'd')
# Add axis labels and a title
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix for Random forest")
plt.show()
# y_true: true labels, y_pred_prob: predicted probabilities
fpr, tpr, thresholds = roc_curve(y_tests, y_preds)
# plot ROC curve
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--') # plot random curve
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
10
plt.show()
KNN:
knn = KNeighborsClassifier(n_neighbors=5)
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
accuracy_scores = []
# Perform K-fold cross-validation and evaluate the model's performance
y_true = []
y_pred = []
for train_idx, test_idx in kfold.split(X):
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
knn.fit(X_train, y_train)
y_true.extend(y_test)
y_pred.extend(knn.predict(X_test))
accuracy = accuracy_score(y_true, y_pred)
# Add accuracy score to list
accuracy_scores.append(accuracy)
# Compute and print the mean accuracy score and standard deviation
print("Accuracy- KNN: %0.2f (+/- %0.2f)" % (np.mean(accuracy_scores),
np.std(accuracy_scores) * 2))
Decision tree:
# Initialize an empty list to store cross-validation scores
scores = []
y_pred=[]
y_preds=[]
y_tests=[]
# Iterate over the splits of the data and train/test the model
for train_index, test_index in kf.split(X):
# Split the data into training and testing sets for this fold
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]
# Train the decision tree classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
# Evaluate the model's performance on the test set for this fold and store the score
y_pred = dt.predict(X_test)
y_preds.extend(y_pred)
y_tests.extend(y_test)
11
score = accuracy_score(y_test, y_pred)
scores.append(score)
# Calculate the mean and standard deviation of the cross-validation scores
mean_score = sum(scores) / len(scores)
std_dev = np.std(scores)
# Print the results
print("Accuracy-Decision tree: %0.2f (+/- %0.2f)" % (mean_score, std_dev * 2))
Logistic Regression:
# Perform 10-fold cross validation using KFold method
kf = KFold(n_splits=10, shuffle=True, random_state=42)
lr = LogisticRegression(C=1)
scores = []
y_preds = []
y_true = []
for train_index, test_index in kf.split(X):
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_preds.extend(y_pred)
y_true.extend(y_test)
scores.append(lr.score(X_test, y_test))
# Calculate and print the cross-validation accuracy
print("Accuracy- Logistic Regression: %0.2f (+/- %0.2f)" % (np.mean(scores), np.std(scores)
* 2))
SVM:
y_pred=[]
y_preds=[]
y_true=[]
# Define the desired sample size for the reduced dataset
sample_size = 5000
# Initialize SVM classifier with default hyperparameters
svm = SVC()
# Use stratified k-fold cross-validation to evaluate classifier performance
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
scores = []
# Loop over each fold and perform stratified sampling on the training set
for train_index, test_index in skf.split(X,y):
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
12
y_train, y_test = y.iloc[train_index], y.iloc[test_index]
# Use stratified sampling to select the desired number ofsamples from each class
X_sampled, y_sampled = resample(X_train, y_train, n_samples=sample_size,
stratify=y_train, random_state=42)
# Fit SVM classifier on the reduced dataset and evaluate performance on the test set
svm.fit(X_sampled, y_sampled)
y_pred = svm.predict(X_test)
y_preds.extend(y_pred)
y_true.extend(y_test)
scores.append(svm.score(X_test,y_test))
# Print the average classification accuracy over all folds
print("Accuracy- SVM: {:.2f}".format(sum(scores)/len(scores)))
ENSEMBLE
Voting classifier Ensemble:
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# Set up KFold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
# Initialize empty lists to store scores and predictions
scores = []
y_preds = []
y_tests = []
# Train the models and ensemble them
rf = RandomForestClassifier(n_estimators=100)
dt = DecisionTreeClassifier(random_state=42)
lr=LogisticRegression(C=1)
ensemble = VotingClassifier(estimators=[('rf', rf), ('dt', dt), ('lr', lr)], voting='hard')
# Loop over the splits of the data and train/test the models
for train_index, test_index in kf.split(X):
# Split the data into training and testing sets for this fold
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]
# Fit the models to the training data for this fold
rf.fit(X_train, y_train)
dt.fit(X_train, y_train)
lr.fit(X_train, y_train)
# Fit the ensemble model to the training data for this fold
13
ensemble.fit(X_train, y_train)
# Evaluate the performance of the ensemble model on the test set for this fold
y_pred = ensemble.predict(X_test)
y_preds.extend(y_pred)
y_tests.extend(y_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)
# Calculate the mean and standard deviation of the cross-validation scores
mean_score = sum(scores) / len(scores)
std_dev = np.std(scores)
# Print the results
print("Ensemble accuracy: %0.4f (+/- %0.2f)" % (mean_score, std_dev * 2))
Adaboost ensemble:
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier
# Set up KFold cross-validation
kf = KFold(n_splits=4, shuffle=True, random_state=42)
# Initialize empty lists to store scores and predictions
scores = []
y_preds = []
y_tests = []
# Train the models and ensemble them
rf = RandomForestClassifier(n_estimators=100)
dt = DecisionTreeClassifier(random_state=42)
ada = AdaBoostClassifier(estimator=dt, n_estimators=100)
# Loop over the splits of the data and train/test the models
for train_index, test_index in kf.split(X):
# Split the data into training and testing sets for this fold
X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]
# Fit the models to the training data for this fold
rf.fit(X_train, y_train)
dt.fit(X_train, y_train)
ada.fit(X_train, y_train)
# Combine the models into a voting classifier
ensemble = VotingClassifier(estimators=[('rf', rf), ('ada', ada)], voting='hard')
# Fit the ensemble model to the training data for this fold
ensemble.fit(X_train, y_train)
# Evaluate the performance of the ensemble model on the test set for this fold
14
# Train the models and ensemble them
rf = RandomForestClassifier(n_estimators=100)dt =
DecisionTreeClassifier(random_state=42)
y_pred = ensemble.predict(X_test)
y_preds.extend(y_pred)
y_tests.extend(y_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)
# Calculate the mean and standard deviation of the cross-validation scores
mean_score = sum(scores) / len(scores)
std_dev = np.std(scores)
# Print the results
print("AdaBoost Ensemble accuracy: %0.4f (+/- %0.2f)" % (mean_score, std_dev * 2)
