import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Wczytanie danych z pliku CSV
data = pd.read_csv("games.csv")

# Słownik: kod ECO -> nazwa otwarcia
eco_to_name = data.drop_duplicates('opening_eco').set_index('opening_eco')['opening_name'].to_dict()

# Tworzenie cech binarnych: otwarcie i zwycięzca
openings = pd.get_dummies(data['opening_eco'], prefix='eco')
winners = pd.get_dummies(data['winner'], prefix='winner')
df = pd.concat([openings, winners], axis=1)

# Szukanie częstych zbiorów
frequent = apriori(df, min_support=0.015, use_colnames=True)

# Generowanie reguł asocjacyjnych
rules = association_rules(frequent, metric="confidence", min_threshold=0.5)

# Wybór najciekawszych reguł (największy lift)
top_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False).head(20)

# Wyświetlenie reguły z nazwą otwarcia
for idx, row in top_rules.iterrows():
    eco_code = list(row['antecedents'])[0].replace('eco_', '')
    opening_name = eco_to_name.get(eco_code, eco_code)
    winner = list(row['consequents'])[0].replace('winner_', '')
    print(f"Reguła: Jeśli otwarcie to '{opening_name}' ({eco_code}), to zwycięzca to '{winner}'.")
    print(f"  - support: {row['support']:.3f} (czyli {row['support']*100:.1f}% partii)")
    print(f"  - confidence: {row['confidence']:.2f} (czyli w {row['confidence']*100:.1f}% przypadków)")
    print(f"  - lift: {row['lift']:.2f} (im wyżej ponad 1, tym silniejsza zależność)\n")

# Zapisanie wyników do pliku CSV
top_rules.to_csv("association_rules_results.csv", index=False)