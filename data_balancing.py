class_counts_typ = typhoid['RESULT_TEXT'].value_counts()
class_distribution_typ =class_counts_typ / len(typhoid) *100
print(class_distribution_typ)
#plot a bar graph
value = typhoid['RESULT_TEXT'].value_counts()
plt.bar(value.index, value.values)
plt.title('RESULT_TEXT')
plt.xlabel('Result')
plt.ylabel('Count')
plt.show()
# Resample data to handle imbalance
x = typhoid.drop('RESULT_TEXT', axis = 1)
y = typhoid['RESULT_TEXT']
ros = RandomOverSampler(random_state=42)
x_resampled, y_resampled = ros.fit_resample(x, y)
#Concatenate the features and target into balanced datset
balanced_data = pd.concat([x_resampled, y_resampled], axis=1)
balanced_data.to_csv('new_copy2/Balanced_Typhoid.csv', index = False)
Balanced_typhoid = pd.read_csv('new_copy2/Balanced_Typhoid.csv')
#Class Distribution
class_counts_typ = Balanced_typhoid['RESULT_TEXT'].value_counts()
class_distribution_typ =class_counts_typ / len(Balanced_typhoid) *100
print(class_distribution_typ)
#plot a bar graph
value = Balanced_typhoid['RESULT_TEXT'].value_counts()
plt.bar(value.index, value.values)
plt.title('RESULT_TEXT')
7
plt.xlabel('Result')
plt.ylabel('Count')
plt.show()
