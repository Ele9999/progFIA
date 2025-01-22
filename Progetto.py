import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
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

# Rimozione colonne inutili ai fini computazionali
data_cleaned = data.drop(columns=['ID'])

# Divisione del dataset in features e label
X = data_cleaned.drop('Recidiva/Non_Recidiva', axis = 1)
y = data_cleaned['Recidiva/Non_Recidiva']
# Divisione del dataset in set di training e set di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Gestione valori mancanti sostituendo i valori NaN con la media
numerical_imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(numerical_imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(numerical_imputer.transform(X_test), columns=X_test.columns)

# Creazione di una pipeline che fa lo scaling dei dati e applica un modello SVM con kernel RBF
svc_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling dei dati
    ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))  # Modello SVC
])

# Addestramento sul set di training
svc_pipeline.fit(X_train,y_train)
# Predizioni sul set di test
predictions = svc_pipeline.predict(X_test)

# Calcolo della matrice di confusione 
CM = confusion_matrix(y_test, predictions)
print("Matrice di Confusione:")
print(CM)

print("\nClassification Report:")
# Calcolo del report di classificazione
print(classification_report(y_test, predictions))

# Griglia di parametri per ottimizzare il modello
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [1, 0.1, 0.01, 0.001, 'scale'],
    'svc__kernel': ['rbf', 'linear', 'poly']
}

# Cerca, tra tutte le possibili combinazioni dei parametri, quelli migliori
grid_search = GridSearchCV(svc_pipeline, param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Predizioni sul set di test con il miglior modello trovato
grid_predictions = grid_search.best_estimator_.predict(X_test)

# Calcolo della matrice di confusione per il modello ottimizzato
CM2 = confusion_matrix(y_test, grid_predictions)
print("Matrice di Confusione:")
print(CM2)


print("\nClassification Report:")
print(classification_report(y_test,grid_predictions))

# Calcolo delle metriche di validazione per il modello con parametri standard
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
print("\nMetriche sul Test Set:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Calcolo delle metriche di validazione per il modello ottimizzato
accuracy = accuracy_score(y_test, grid_predictions)
precision = precision_score(y_test, grid_predictions)
recall = recall_score(y_test, grid_predictions)
f1 = f1_score(y_test, grid_predictions)
print("\nMetriche sul Test Set:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Funzione per testare il modello ottimizzato sul set di test
def test_full_test_set():

    # Previsioni sul set di test
    predictions = grid_search.predict(X_test)

    # Calcola il numero di previsioni corrette confrontando i valori reali con quelli predetti
    correct_predictions = (y_test == predictions).sum()
    # Determina il totale delle previsioni
    total_predictions = len(y_test)
    # Calcola l'accuratezza
    accuracy_on_test = correct_predictions / total_predictions

    # Stampa i risultati delle previsioni per ogni esempio nel dataset di test
    print("\nRisultati sui dati del Test Set:")
    # Itera attraverso le previsioni
    for i, pred in enumerate(predictions):
        # Per ogni esempio mostra la classe predetta
        print(f"Esempio {i+1}: {'Recidiva' if pred == 1 else 'Non Recidiva'} "
            # e il valore reale corrispondente per confrontare il modello con i dati reali
              f"(Reale: {'Recidiva' if y_test.iloc[i] == 1 else 'Non Recidiva'})")

    print(f"\nNumero di predizioni corrette: {correct_predictions}")
    print(f"Totale predizioni: {total_predictions}")
    print(f"Accuracy sul Test Set: {accuracy_on_test:.2f}")


if __name__ == '__main__':
    test_full_test_set()