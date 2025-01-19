import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score


# Caricamento del dataset
file_path = 'data-istruzioni/Data.xlsx'
data = pd.read_excel(file_path)

# Rimozione colonne inutili
data_cleaned = data.drop(columns=['ID'])

#sns.heatmap(data_cleaned.isnull(), cbar=False, cmap='viridis')
#plt.title('Mappa dei Valori Mancanti')
#plt.show()

X = data_cleaned.drop('Recidiva/Non_Recidiva', axis = 1)
y = data_cleaned['Recidiva/Non_Recidiva']
# Divisione train/test con proporzione 70%-30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Gestione valori mancanti
numerical_imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(numerical_imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(numerical_imputer.transform(X_test), columns=X_test.columns)

#sns.heatmap(X_train.isnull(), cbar=False, cmap='viridis')
#plt.title('Mappa dei Valori Mancanti sul set di train dopo data imputation')
#plt.show()

svc_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling dei dati
    ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))  # Modello SVC
])

svc_pipeline.fit(X_train,y_train)

predictions = svc_pipeline.predict(X_test)

# Matrice di confusione e report
CM = confusion_matrix(y_test, predictions)
print("Matrice di Confusione:")
print(CM)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

#pred_classes = ['real_' + str(c) for c in map(str, y.unique())]
#real_classes = ['pred_' + str(c) for c in map(str, y.unique())]
#pd.DataFrame(confusion_matrix(y_test, predictions),
#             index=pred_classes, columns=real_classes)

#disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=['Non Recidiva', 'Recidiva'])
#disp.plot(cmap='Blues')
#plt.title('Matrice di Confusione')
#plt.show()

param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [1, 0.1, 0.01, 0.001, 'scale'],
    'svc__kernel': ['rbf', 'linear', 'poly']
}

grid_search = GridSearchCV(svc_pipeline, param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

grid_predictions = grid_search.best_estimator_.predict(X_test)

#pd.DataFrame(confusion_matrix(y_test, grid_predictions),
#             index=pred_classes, columns=real_classes)

CM2 = confusion_matrix(y_test, grid_predictions)
print("Matrice di Confusione:")
print(CM2)

#disp = ConfusionMatrixDisplay(confusion_matrix=CM2, display_labels=['Non Recidiva', 'Recidiva'])
#disp.plot(cmap='Blues')
#plt.title('Matrice di Confusione')
#plt.show()

print("\nClassification Report:")
print(classification_report(y_test,grid_predictions))

# Valutazione globale del modello
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\nMetriche globali sul Test Set:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Valutazione globale del modello con iperparametri
accuracy = accuracy_score(y_test, grid_predictions)
precision = precision_score(y_test, grid_predictions)
recall = recall_score(y_test, grid_predictions)
f1 = f1_score(y_test, grid_predictions)

print("\nMetriche globali sul Test Set:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

#if hasattr(grid_search, "decision_function"):
#    y_prob = grid_search.decision_function(X_test)
#
## Calcolo dei valori ROC
#fpr, tpr, thresholds = roc_curve(y_test, y_prob)
#roc_auc = roc_auc_score(y_test, y_prob)
#
## Plottaggio della curva ROC
#plt.figure(figsize=(8, 6))
#plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC)')
#plt.legend(loc="lower right")
#plt.grid(alpha=0.3)
#plt.show()

def test_full_test_set():
    """
    Testa il modello SVC sui dati del test set e calcola l'accuratezza.
    
    """
    # Predizioni sul test set
    predictions = grid_search.predict(X_test)

    # Calcolo dell'accuratezza
    correct_predictions = (y_test == predictions).sum()
    total_predictions = len(y_test)
    accuracy_on_test = correct_predictions / total_predictions

    print("\nRisultati sui dati del Test Set:")
    for i, pred in enumerate(predictions):
        print(f"Esempio {i+1}: {'Recidiva' if pred == 1 else 'Non Recidiva'} "
              f"(Reale: {'Recidiva' if y_test.iloc[i] == 1 else 'Non Recidiva'})")

    print(f"\nNumero di predizioni corrette: {correct_predictions}")
    print(f"Totale predizioni: {total_predictions}")
    print(f"Accuracy sul Test Set: {accuracy_on_test:.2f}")



if __name__ == '__main__':
    test_full_test_set()