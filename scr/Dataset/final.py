import pandas as pd  
import numpy as np

# Načtení datasetů
result = pd.read_csv('../Dataset/archive/results.csv')
drivers = pd.read_csv('../Dataset/archive/drivers.csv')
races = pd.read_csv('../Dataset/archive/races.csv')
constructors_standing = pd.read_csv('../Dataset/archive/constructor_standings.csv')
drivers_standings = pd.read_csv('../Dataset/archive/driver_standings.csv')
qualifying = pd.read_csv('../Dataset/archive/qualifying.csv')
pit_stops = pd.read_csv('../Dataset/archive/pit_stops.csv')
lap_times = pd.read_csv('../Dataset/archive/lap_times.csv')

# Filtrace závodů od roku 2007
races2018 = races[races['year'] >= 2007]
races2018 = races2018[['raceId', 'year', 'circuitId', 'name', 'date']]

# Filtrace výsledků podle závodů od 2007
result = result[result['raceId'].isin(races2018['raceId'])]

# Výběr závodů pro rok 2024 a aktivních jezdců
races2024 = races2018[races2018['year'] == 2024]
results2024 = result[result['raceId'].isin(races2024['raceId'])]
active_drivers_id = results2024['driverId'].unique()
drivers2024 = drivers[drivers['driverId'].isin(active_drivers_id)]
drivers2024.drop(columns=['url', 'number', 'driverRef', 'forename', 'surname', 'dob', 'nationality', 'code'], inplace=True)

# Filtrace výsledků a přejmenování sloupců
result = result[result['driverId'].isin(drivers2024['driverId'])]
result.drop(columns=['positionText', 'time', 'milliseconds'], inplace=True)
result.rename(columns={'position': 'final_position', 'points': 'points_per_race'}, inplace=True)

# Filtrace a přejmenování constructors_standing
constructors_standing = constructors_standing[constructors_standing['raceId'].isin(result['raceId'])]
constructors_standing.drop(columns=['constructorStandingsId', 'positionText'], inplace=True)
constructors_standing.rename(columns={'points': 'constructor_points', 'position': 'constructor_position', 'wins': 'constructor_wins'}, inplace=True)

# Filtrace drivers_standings
drivers_standings = drivers_standings[drivers_standings['raceId'].isin(result['raceId'])]
drivers_standings.drop(columns=['driverStandingsId', 'positionText'], inplace=True)

# Filtrace a přejmenování qualifying
qualifying = qualifying[qualifying['raceId'].isin(result['raceId'])]
qualifying.drop(columns=['qualifyId', 'number'], inplace=True)
qualifying = qualifying.rename(columns={'position': 'qualifying_position'})

# Spojení datasetů
data = pd.merge(qualifying, drivers2024, on=['driverId'], how='inner')
data = pd.merge(data, drivers_standings, on=['raceId', 'driverId'], how='inner')
data = pd.merge(data, result, on=['raceId', 'driverId', 'constructorId'], how='left')
data = pd.merge(data, constructors_standing, on=['raceId', 'constructorId'], how='left')

# Převod časů na sekundy
def time_to_seconds(time_str):
    if pd.isna(time_str) or time_str == '\\N':
        return np.nan
    minutes, seconds = time_str.split(':')
    return int(minutes) * 60 + float(seconds)

data['q1'] = data['q1'].apply(time_to_seconds)
data['q2'] = data['q2'].apply(time_to_seconds)
data['q3'] = data['q3'].apply(time_to_seconds)
data['fastestLapTime'] = data['fastestLapTime'].apply(time_to_seconds)

# Úprava sloupců a výpočet existujících průměrných hodnot
data['final_position'] = data['final_position'].replace({'\\N': 0}).astype(int)
data['fastestLap'] = data['fastestLap'].replace({'\\N': 0}).astype(int)
data['fastestLapSpeed'] = data['fastestLapSpeed'].replace({'\\N': 0}).astype(float)
data = data.sort_values(['driverId', 'raceId'])
data['avg_position_last_5'] = data.groupby('driverId')['positionOrder'].shift(1).rolling(window=5, min_periods=1).mean()
data['points_before_race'] = data.groupby('driverId')['points'].shift(1).fillna(0).cumsum()

# Přidání informací o okruhu a výpočet průměrné pozice na okruhu
races = pd.read_csv('../Dataset/archive/races.csv')[['raceId', 'circuitId', 'year']]
data = pd.merge(data, races, on='raceId', how='left')
circuit_avg = data.groupby(['driverId', 'circuitId'])['positionOrder'].mean().reset_index()
circuit_avg = circuit_avg.rename(columns={'positionOrder': 'avg_position_circuit'})
data = pd.merge(data, circuit_avg, on=['driverId', 'circuitId'], how='left')

# Výpočet rozdílu mezi grid a kvalifikační pozicí
data['grid_qual_diff'] = data['grid'] - data['qualifying_position']
data = pd.merge(data, races, on=['raceId', 'circuitId'], how='left')

# Úprava sloupců
data.drop(columns=['year_y', 'number', 'rank'], inplace=True)
data.rename(columns={'year_x': 'year'}, inplace=True)

# Nové průměrné hodnoty
# 1. Průměrná pozice jezdce v minulých sezónách
avg_position_season = data.groupby(['driverId', 'year'])['positionOrder'].mean().reset_index()
avg_position_season = avg_position_season.rename(columns={'positionOrder': 'avg_position_per_season'})
data = pd.merge(data, avg_position_season, on=['driverId', 'year'], how='left')

# 2. Průměrné body za závod v minulých sezónách
avg_points_season = data.groupby(['driverId', 'year'])['points_per_race'].mean().reset_index()
avg_points_season = avg_points_season.rename(columns={'points_per_race': 'avg_points_per_race_season'})
data = pd.merge(data, avg_points_season, on=['driverId', 'year'], how='left')

# 3. Průměrná pozice týmu v minulých sezónách
avg_constructor_position = data.groupby(['constructorId', 'year'])['constructor_position'].mean().reset_index()
avg_constructor_position = avg_constructor_position.rename(columns={'constructor_position': 'avg_constructor_position_season'})
data = pd.merge(data, avg_constructor_position, on=['constructorId', 'year'], how='left')

# 4. Průměrná rychlost nejrychlejšího kola jezdce
avg_fastest_lap_speed = data.groupby('driverId')['fastestLapSpeed'].mean().reset_index()
avg_fastest_lap_speed = avg_fastest_lap_speed.rename(columns={'fastestLapSpeed': 'avg_fastest_lap_speed'})
data = pd.merge(data, avg_fastest_lap_speed, on='driverId', how='left')

# 5. Průměrná doba zastávky v boxech pro jezdce
pit_stops['duration_seconds'] = pit_stops['milliseconds'] / 1000.0
avg_pit_stop_duration = pit_stops.groupby(['driverId', 'raceId'])['duration_seconds'].mean().reset_index()
avg_pit_stop_duration = avg_pit_stop_duration.groupby('driverId')['duration_seconds'].mean().reset_index()
avg_pit_stop_duration = avg_pit_stop_duration.rename(columns={'duration_seconds': 'avg_pit_stop_duration'})
data = pd.merge(data, avg_pit_stop_duration, on='driverId', how='left')

# 6. Průměrný čas kola na okruhu pro jezdce
lap_times['lap_time_seconds'] = lap_times['milliseconds'] / 1000.0
lap_times = lap_times.merge(races[['raceId', 'circuitId']], on='raceId', how='left')
avg_lap_time_circuit = lap_times.groupby(['driverId', 'circuitId'])['lap_time_seconds'].mean().reset_index()
avg_lap_time_circuit = avg_lap_time_circuit.rename(columns={'lap_time_seconds': 'avg_lap_time_circuit'})
data = pd.merge(data, avg_lap_time_circuit, on=['driverId', 'circuitId'], how='left')

# Uložení datasetu
data.to_csv('../Dataset/output/2024.csv', index=False)

# Načtení nového datasetu pro kontrolu
newData = pd.read_csv('../Dataset/output/2024.csv')
print(newData.head())