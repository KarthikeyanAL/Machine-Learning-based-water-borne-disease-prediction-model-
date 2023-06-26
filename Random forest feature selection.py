# Drop rows with missing data
Normal_typhoid.dropna(inplace=True)
# Split the data into feature matrix X and target vector y
X = Normal_typhoid.drop('RESULT_TEXT', axis=1)
y = Normal_typhoid['RESULT_TEXT']
# Use random forest to select the most important features
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X, y)
importance = rfc.feature_importances_
# Create a list of (feature name, importance) tuples and sort by importance
features = list(zip(X.columns, importance))
features.sort(key=lambda x: x[1], reverse=True)
# Print the sorted list of feature importances
for f in features:
print(f)
# Select the top k features
k = 7
top_features = [f[0] for f in features[:k]]
for f in top_features:
print(f)
X = X[top_features]
plt.title("Histogram for Feature selection importance");
plt.barh([x[0] for x in features],[x[1] for x in features])
plt.show()
