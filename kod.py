import pandas as pd

true_df = pd.read_csv('true.csv')
fake_df = pd.read_csv('fake.csv')

true_df['label'] = 1   # 1 = pravdivá správa
fake_df['label'] = 0   # 0 = falošná správa

df = pd.concat([true_df, fake_df], ignore_index=True)
print(df.shape)
print(df.columns)  # mal by obsahovať napr. title, text, subject, date, label

df['text_all'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

df['text_all'] = df['text_all'].str.lower()
df['text_all'] = df['text_all'].str.replace('[^a-z ]', ' ', regex=True)

print(df['text_all'].head())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

X_train, X_test, y_train, y_test = train_test_split(
    df['text_all'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print("Train accuracy:", model.score(X_train_vec, y_train))
print("Test accuracy:", model.score(X_test_vec, y_test))

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['FAKE (0)', 'TRUE (1)']).plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()