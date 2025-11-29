from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# y_test: Gerçek etiketler
# y_pred: Modelin tahminleri

print("--- Detaylı Performans Raporu ---")
print(classification_report(y_test, y_pred))

# Karmaşıklık Matrisi (Confusion Matrix) Çizdir
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Durum')
plt.title('Confusion Matrix')
plt.show()