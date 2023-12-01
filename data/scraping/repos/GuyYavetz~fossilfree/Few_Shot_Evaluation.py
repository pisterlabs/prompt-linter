import gspread
from oauth2client.service_account import ServiceAccountCredentials as Credentials
import openai
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# Set up Google Sheets API credentials
scope = ['https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_json_keyfile_name('client_s.json', scope)
client = gspread.authorize(creds)
sheet = client.open('#USE YOUR SHEET NAME HERE')
worksheet = sheet.get_worksheet(0)


true_labels = worksheet.col_values(8)[1:]

predicted_labels = worksheet.col_values(11)[1:]

print("Unique true labels: ", np.unique(true_labels))
print("Unique predicted labels: ", np.unique(predicted_labels))


# Calculate the classification report
report = classification_report(true_labels, predicted_labels)
print(report)

# Calculate the confusion matrix
confusion = confusion_matrix(true_labels, predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
