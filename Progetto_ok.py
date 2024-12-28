import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from sklearn.ensemble import RandomForestClassifier


# Caricamento del dataset
file_path = 'data-istruzioni/Data.xlsx'  # Cambia con il percorso reale
data = pd.read_excel(file_path)

# Rimuovi colonne inutili
data_cleaned = data.drop(columns=['ID'])

# Gestione valori mancanti: sostituisce i NaN con la media per colonne numeriche
numerical_imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(numerical_imputer.fit_transform(data_cleaned.iloc[:, :-1]), columns=data_cleaned.columns[:-1])
y = data_cleaned.iloc[:, -1]  # Colonna target (Recidiva/Non_Recidiva)


# Selezione delle migliori 10 feature
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Salviamo il selettore per future predizioni
joblib.dump(selector, 'feature_selector.pkl')


# Addestramento del modello Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
model.fit(X_selected, y)

# Salva il modello
joblib.dump(model, 'final_model.pkl')

# Mostra le feature selezionate
selected_features = X.columns[selector.get_support()]
print("Feature selezionate:", selected_features.tolist())


# Carica il modello e il selettore salvati
model = joblib.load('final_model.pkl')
selector = joblib.load('feature_selector.pkl')

# Colonne originali del dataset usate per il training (249 feature)
all_columns = [
    'Epoca_TC', 'Area_grasso_periviscerale', 'Area_grasso_sottocutaneo', 'Istologia', 'GS_alla_diagnosi', 'TNM_alla_diagnosi', 'Eta_alla_RP', 'HIST_mean', 'HIST_std', 'HIST_skewness', 'HIST_kurtosis', 'HIST_energy', 'HIST_entropy', 'HIST_maxAssValue', 'HIST_maxAssPos', 'HIST_energy_around_maxAss', 'HIST_range', 'HIST_numMaxRel', 'HIST_energy_around_maxRel', 'GLCM3_autocorrelation_-1_-1_-1', 'GLCM3_covariance_-1_-1_-1', 'GLCM3_inertia_-1_-1_-1', 'GLCM3_absolute_-1_-1_-1', 'GLCM3_inverse_-1_-1_-1', 'GLCM3_energy_-1_-1_-1', 'GLCM3_entropy_-1_-1_-1', 'GLCM3_autocorrelation_-1_-1_0', 'GLCM3_covariance_-1_-1_0', 'GLCM3_inertia_-1_-1_0', 'GLCM3_absolute_-1_-1_0', 'GLCM3_inverse_-1_-1_0', 'GLCM3_energy_-1_-1_0', 'GLCM3_entropy_-1_-1_0', 'GLCM3_autocorrelation_-1_-1_1', 'GLCM3_covariance_-1_-1_1', 'GLCM3_inertia_-1_-1_1', 'GLCM3_absolute_-1_-1_1', 'GLCM3_inverse_-1_-1_1', 'GLCM3_energy_-1_-1_1', 'GLCM3_entropy_-1_-1_1', 'GLCM3_autocorrelation_-1_0_-1', 'GLCM3_covariance_-1_0_-1', 'GLCM3_inertia_-1_0_-1', 'GLCM3_absolute_-1_0_-1', 'GLCM3_inverse_-1_0_-1', 'GLCM3_energy_-1_0_-1', 'GLCM3_entropy_-1_0_-1', 'GLCM3_autocorrelation_-1_0_0', 'GLCM3_covariance_-1_0_0', 'GLCM3_inertia_-1_0_0', 'GLCM3_absolute_-1_0_0', 'GLCM3_inverse_-1_0_0', 'GLCM3_energy_-1_0_0', 'GLCM3_entropy_-1_0_0', 'GLCM3_autocorrelation_-1_0_1', 'GLCM3_covariance_-1_0_1', 'GLCM3_inertia_-1_0_1', 'GLCM3_absolute_-1_0_1', 'GLCM3_inverse_-1_0_1', 'GLCM3_energy_-1_0_1', 'GLCM3_entropy_-1_0_1', 'GLCM3_autocorrelation_-1_1_-1', 'GLCM3_covariance_-1_1_-1', 'GLCM3_inertia_-1_1_-1', 'GLCM3_absolute_-1_1_-1', 'GLCM3_inverse_-1_1_-1', 'GLCM3_energy_-1_1_-1', 'GLCM3_entropy_-1_1_-1', 'GLCM3_autocorrelation_-1_1_0', 'GLCM3_covariance_-1_1_0', 'GLCM3_inertia_-1_1_0', 'GLCM3_absolute_-1_1_0', 'GLCM3_inverse_-1_1_0', 'GLCM3_energy_-1_1_0', 'GLCM3_entropy_-1_1_0', 'GLCM3_autocorrelation_-1_1_1', 'GLCM3_covariance_-1_1_1', 'GLCM3_inertia_-1_1_1', 'GLCM3_absolute_-1_1_1', 'GLCM3_inverse_-1_1_1', 'GLCM3_energy_-1_1_1', 'GLCM3_entropy_-1_1_1', 'GLCM3_autocorrelation_0_-1_-1', 'GLCM3_covariance_0_-1_-1', 'GLCM3_inertia_0_-1_-1', 'GLCM3_absolute_0_-1_-1', 'GLCM3_inverse_0_-1_-1', 'GLCM3_energy_0_-1_-1', 'GLCM3_entropy_0_-1_-1', 'GLCM3_autocorrelation_0_-1_0', 'GLCM3_covariance_0_-1_0', 'GLCM3_inertia_0_-1_0', 'GLCM3_absolute_0_-1_0', 'GLCM3_inverse_0_-1_0', 'GLCM3_energy_0_-1_0', 'GLCM3_entropy_0_-1_0', 'GLCM3_autocorrelation_0_-1_1', 'GLCM3_covariance_0_-1_1', 'GLCM3_inertia_0_-1_1', 'GLCM3_absolute_0_-1_1', 'GLCM3_inverse_0_-1_1', 'GLCM3_energy_0_-1_1', 'GLCM3_entropy_0_-1_1', 'GLCM3_autocorrelation_0_0_-1', 'GLCM3_covariance_0_0_-1', 'GLCM3_inertia_0_0_-1', 'GLCM3_absolute_0_0_-1', 'GLCM3_inverse_0_0_-1', 'GLCM3_energy_0_0_-1', 'GLCM3_entropy_0_0_-1', 'GLCM3_autocorrelation_0_0_1', 'GLCM3_covariance_0_0_1', 'GLCM3_inertia_0_0_1', 'GLCM3_absolute_0_0_1', 'GLCM3_inverse_0_0_1', 'GLCM3_energy_0_0_1', 'GLCM3_entropy_0_0_1', 'GLCM3_autocorrelation_0_1_-1', 'GLCM3_covariance_0_1_-1', 'GLCM3_inertia_0_1_-1', 'GLCM3_absolute_0_1_-1', 'GLCM3_inverse_0_1_-1', 'GLCM3_energy_0_1_-1', 'GLCM3_entropy_0_1_-1', 'GLCM3_autocorrelation_0_1_0', 'GLCM3_covariance_0_1_0', 'GLCM3_inertia_0_1_0', 'GLCM3_absolute_0_1_0', 'GLCM3_inverse_0_1_0', 'GLCM3_energy_0_1_0', 'GLCM3_entropy_0_1_0', 'GLCM3_autocorrelation_0_1_1', 'GLCM3_covariance_0_1_1', 'GLCM3_inertia_0_1_1', 'GLCM3_absolute_0_1_1', 'GLCM3_inverse_0_1_1', 'GLCM3_energy_0_1_1', 'GLCM3_entropy_0_1_1', 'GLCM3_autocorrelation_1_-1_-1', 'GLCM3_covariance_1_-1_-1', 'GLCM3_inertia_1_-1_-1', 'GLCM3_absolute_1_-1_-1', 'GLCM3_inverse_1_-1_-1', 'GLCM3_energy_1_-1_-1', 'GLCM3_entropy_1_-1_-1', 'GLCM3_autocorrelation_1_-1_0', 'GLCM3_covariance_1_-1_0', 'GLCM3_inertia_1_-1_0', 'GLCM3_absolute_1_-1_0', 'GLCM3_inverse_1_-1_0', 'GLCM3_energy_1_-1_0', 'GLCM3_entropy_1_-1_0', 'GLCM3_autocorrelation_1_-1_1', 'GLCM3_covariance_1_-1_1', 'GLCM3_inertia_1_-1_1', 'GLCM3_absolute_1_-1_1', 'GLCM3_inverse_1_-1_1', 'GLCM3_energy_1_-1_1', 'GLCM3_entropy_1_-1_1', 'GLCM3_autocorrelation_1_0_-1', 'GLCM3_covariance_1_0_-1', 'GLCM3_inertia_1_0_-1', 'GLCM3_absolute_1_0_-1', 'GLCM3_inverse_1_0_-1', 'GLCM3_energy_1_0_-1', 'GLCM3_entropy_1_0_-1', 'GLCM3_autocorrelation_1_0_0', 'GLCM3_covariance_1_0_0', 'GLCM3_inertia_1_0_0', 'GLCM3_absolute_1_0_0', 'GLCM3_inverse_1_0_0', 'GLCM3_energy_1_0_0', 'GLCM3_entropy_1_0_0', 'GLCM3_autocorrelation_1_0_1', 'GLCM3_covariance_1_0_1', 'GLCM3_inertia_1_0_1', 'GLCM3_absolute_1_0_1', 'GLCM3_inverse_1_0_1', 'GLCM3_energy_1_0_1', 'GLCM3_entropy_1_0_1', 'GLCM3_autocorrelation_1_1_-1', 'GLCM3_covariance_1_1_-1', 'GLCM3_inertia_1_1_-1', 'GLCM3_absolute_1_1_-1', 'GLCM3_inverse_1_1_-1', 'GLCM3_energy_1_1_-1', 'GLCM3_entropy_1_1_-1', 'GLCM3_autocorrelation_1_1_0', 'GLCM3_covariance_1_1_0', 'GLCM3_inertia_1_1_0', 'GLCM3_absolute_1_1_0', 'GLCM3_inverse_1_1_0', 'GLCM3_energy_1_1_0', 'GLCM3_entropy_1_1_0', 'GLCM3_autocorrelation_1_1_1', 'GLCM3_covariance_1_1_1', 'GLCM3_inertia_1_1_1', 'GLCM3_absolute_1_1_1', 'GLCM3_inverse_1_1_1', 'GLCM3_energy_1_1_1', 'GLCM3_entropy_1_1_1', 'LBP_TOP_mean', 'LBP_TOP_std', 'LBP_TOP_skewness', 'LBP_TOP_kurtosis', 'LBP_TOP_energy', 'LBP_TOP_entropy', 'LBP_TOP_maxAssValue', 'LBP_TOP_maxAssPos', 'LBP_TOP_energy_around_maxAss', 'LBP_TOP_range', 'LBP_TOP_numMaxRel', 'LBP_TOP_energy_around_maxRel', 'LBP_TOP_ri_mean', 'LBP_TOP_ri_std', 'LBP_TOP_ri_skewness', 'LBP_TOP_ri_kurtosis', 'LBP_TOP_ri_energy', 'LBP_TOP_ri_entropy', 'LBP_TOP_ri_maxAssValue', 'LBP_TOP_ri_maxAssPos', 'LBP_TOP_ri_energy_around_maxAss', 'LBP_TOP_ri_range', 'LBP_TOP_ri_numMaxRel', 'LBP_TOP_ri_energy_around_maxRel', 'LBP_TOP_u_mean', 'LBP_TOP_u_std', 'LBP_TOP_u_skewness', 'LBP_TOP_u_kurtosis', 'LBP_TOP_u_energy', 'LBP_TOP_u_entropy', 'LBP_TOP_u_maxAssValue', 'LBP_TOP_u_maxAssPos', 'LBP_TOP_u_energy_around_maxAss', 'LBP_TOP_u_range', 'LBP_TOP_u_numMaxRel', 'LBP_TOP_u_energy_around_maxRel', 'LBP_TOP_ri_u_mean', 'LBP_TOP_ri_u_std', 'LBP_TOP_ri_u_skewness', 'LBP_TOP_ri_u_kurtosis', 'LBP_TOP_ri_u_energy', 'LBP_TOP_ri_u_entropy', 'LBP_TOP_ri_u_maxAssValue', 'LBP_TOP_ri_u_maxAssPos', 'LBP_TOP_ri_u_energy_around_maxAss', 'LBP_TOP_ri_u_range', 'LBP_TOP_ri_u_numMaxRel', 'LBP_TOP_ri_u_energy_around_maxRel'
]

data_to_test=[
    [1, 0, 2, 59.9835616438356, 90.0291533081153, 169.942334444832, 123.77885515153, 1487.40686699116, 5665.51898500577, -0.00795899055858431]
    ]
# Nuovo dato di input
new_data = pd.DataFrame(data_to_test, columns=selected_features)

# Trova le colonne mancanti
missing_cols = [col for col in all_columns if col not in new_data.columns]

# Crea un DataFrame con tutte le colonne mancanti riempite con 0.0
missing_df = pd.DataFrame(0.0, index=new_data.index, columns=missing_cols)

# Concatena il DataFrame originale con quello delle colonne mancanti
new_data = pd.concat([new_data, missing_df], axis=1)

# Riordina le colonne per corrispondere a quelle originali
new_data = new_data[all_columns]


# Converte il DataFrame in array NumPy per il selettore
new_data_array = new_data.to_numpy()

# Trasforma il nuovo dato con il selettore di feature
new_data_selected = selector.transform(new_data_array)

# Predizione con il modello
prediction = model.predict(new_data_selected)
print("Predizione:", "Recidiva" if prediction[0] == 1 else "Non Recidiva")