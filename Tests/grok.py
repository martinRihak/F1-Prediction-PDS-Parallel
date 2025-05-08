import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# 1. Načtení a prozkoumání datasetu
data = pd.read_csv('../scr/Dataset/output/results2018.csv')

# Přidání binární proměnné winner
data['winner'] = (data['positionOrder'] == 1).astype(int)

# 2. Identifikace korelace
numeric_features = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_features.corr()
print("Korelace s 'winner':")
print(correlation_matrix['winner'].sort_values(ascending=False))

# 3. Výběr důležitých příznaků pomocí Random Forest
X = numeric_features.drop(['winner', 'positionOrder'], axis=1)
y = data['winner']
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
selected_features = feature_importance.nlargest(5).index.tolist()
print("Vybrané příznaky:", selected_features)

# 4. Příprava dat pro predikční model
X_selected = data[selected_features + ['driverId', 'circuitId']]  # Přidáme kategoriální příznaky
cat_features = ['driverId', 'circuitId']
num_features = [col for col in selected_features if col not in cat_features]

# Nahrazení chybějících hodnot
X_selected = X_selected.replace('\\N', np.nan).fillna(X_selected.mean(numeric_only=True))

# Předzpracování
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(sparse_output=False), cat_features)
    ])
X_processed = preprocessor.fit_transform(X_selected)

