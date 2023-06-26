le = LabelEncoder()
Balanced_typhoid["MRNO_encoded"] = le.fit_transform(Balanced_typhoid["MRNO"])
Balanced_typhoid["RESULT_VALUE_encoded"] =
le.fit_transform(Balanced_typhoid["RESULT_VALUE"])
Balanced_typhoid["GENDER_encoded"] = le.fit_transform(Balanced_typhoid["GENDER"])
Balanced_typhoid["REPORT_VERIFIED_encoded"] =
le.fit_transform(Balanced_typhoid["REPORT_VERIFIED"])
Balanced_typhoid["RESULT_TEXT_encoded"] =
le.fit_transform(Balanced_typhoid["RESULT_TEXT"])
# One-hot encode District and Tehsil
ohe = OneHotEncoder(sparse=False)
district_tehsil_encoded = ohe.fit_transform(Balanced_typhoid[["DISTRICT", "TEHSIL"]])
district_tehsil_encoded_df = pd.DataFrame(district_tehsil_encoded,
columns=ohe.get_feature_names_out(["DISTRICT", "TEHSIL"]))
# Combine the encoded columns with the original dataset
new_df = pd.concat([Balanced_typhoid["MRNO_encoded"], district_tehsil_encoded_df],
axis=1)
new_df["AGE"] = Balanced_typhoid["AGE"]
new_df["RESULT_TEXT"] = Balanced_typhoid["RESULT_TEXT_encoded"]
new_df["GENDER"] = Balanced_typhoid["GENDER_encoded"]
new_df["RESULT_VALUE"] = Balanced_typhoid["RESULT_VALUE_encoded"]
new_df["REPORT_VERIFIED"] = Balanced_typhoid["REPORT_VERIFIED_encoded"]
new_df["CPT_ID"] = Balanced_typhoid["CPT_ID"]
new_df["CPT_ID.1"] = Balanced_typhoid["CPT_ID.1"]
# Save the new dataframe to a new CSV file
new_df.to_csv('new_copy2/New_Typhoid.csv', index = False)
Data normalization:
value = Balanced_typhoid['AGE'].value_counts()
plt.bar(value.index, value.values)
plt.title('AGE')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
Transformed_typhoid = pd.read_csv('new_copy2/New_Typhoid.csv', low_memory = False)
# Column to be normalized
column = ['AGE']
Transformed_typhoid[column] = (Transformed_typhoid[column] -
Transformed_typhoid[column].mean()) / Transformed_typhoid[column].std()
# New .csv file with normalized data
8
Transformed_typhoid.to_csv('new_copy2/Normalized_Typhoid.csv', index = False)# Column
to be normalized
column = ['AGE']
Transformed_typhoid[column] = (Transformed_typhoid[column] -
Transformed_typhoid[column].mean()) / Transformed_typhoid[column].std()
# New .csv file with normalized data
Transformed_typhoid.to_csv('new_copy2/Normalized_Typhoid.csv', index = False)
Normal_typhoid = pd.read_csv('new_copy2/Normalized_Typhoid.csv', low_memory = False)
print(Normal_typhoid['AGE'].head())
value = Normal_typhoid['AGE'].value_counts()
plt.bar(value.index, value.values)
plt.title('AGE')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
