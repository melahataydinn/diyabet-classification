import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükle
dataset = pd.read_csv('veriseti.csv')

# Bağımsız değişkenler (X) ve hedef değişken (y) ayırma
X = dataset.drop('HeartDisease', axis=1)
y = dataset['HeartDisease']

# Kategorik özelliklerin kodlanması (gerekirse)
X = pd.get_dummies(X)

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes modelini oluşturma ve eğitme
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Test verileri üzerinde modelin performansını değerlendirme
y_pred = nb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)

# Hassasiyet (Precision)
precision = precision_score(y_test, y_pred)
print("Hassasiyet:", precision)

# Geri çağırma (Recall)
recall = recall_score(y_test, y_pred)
print("Geri çağırma:", recall)

# F1-Skoru (F1-Score)
f1 = f1_score(y_test, y_pred)
print("F1-Skoru:", f1)

# Karmaşıklık matrisini hesapla
cm = confusion_matrix(y_test, y_pred)

# Karmaşıklık matrisini görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=['No Disease', 'Heart Disease'],
            yticklabels=['No Disease', 'Heart Disease'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Karmaşıklık Matrisi')
plt.show()
