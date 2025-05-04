# Импортируем нужные библиотеки
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

# Функция для загрузки базы данных  
def load_skempi_data():
    # Формируем путь к файлу (в моем случае, БД в папке Downloads)
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    file_path = os.path.join(downloads_path, "skempi_v2.csv")
    
    # Проверяем, существует ли файл
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден в папке Downloads")
    
    # Читаем CSV, указывая разделитель, чтобы код работал корректно
    data = pd.read_csv(file_path, sep=';')
    return data

# Функция для извлечения признаков
def extract_features(data):
    # Таблица физико-химических свойств аминокислот (гидрофобность, заряд)
    amino_acid_properties = {
        'A': {'hydrophobicity': 1.8, 'charge': 0},  # Аланин
        'R': {'hydrophobicity': -4.5, 'charge': 1}, # Аргинин
        'N': {'hydrophobicity': -3.5, 'charge': 0}, # Аспарагин
        'D': {'hydrophobicity': -3.5, 'charge': -1},# Аспарагиновая кислота
        'C': {'hydrophobicity': 2.5, 'charge': 0},  # Цистеин
        'Q': {'hydrophobicity': -3.5, 'charge': 0}, # Глутамин
        'E': {'hydrophobicity': -3.5, 'charge': -1},# Глутаминовая кислота
        'G': {'hydrophobicity': -0.4, 'charge': 0}, # Глицин
        'H': {'hydrophobicity': -3.2, 'charge': 0}, # Гистидин
        'I': {'hydrophobicity': 4.5, 'charge': 0},  # Изолейцин
        'L': {'hydrophobicity': 3.8, 'charge': 0},  # Лейцин
        'K': {'hydrophobicity': -3.9, 'charge': 1}, # Лизин
        'M': {'hydrophobicity': 1.9, 'charge': 0},  # Метионин
        'F': {'hydrophobicity': 2.8, 'charge': 0},  # Фенилаланин
        'P': {'hydrophobicity': -1.6, 'charge': 0}, # Пролин
        'S': {'hydrophobicity': -0.8, 'charge': 0}, # Серин
        'T': {'hydrophobicity': -0.7, 'charge': 0}, # Треонин
        'W': {'hydrophobicity': -0.9, 'charge': 0}, # Триптофан
        'Y': {'hydrophobicity': -1.3, 'charge': 0}, # Тирозин
        'V': {'hydrophobicity': 4.2, 'charge': 0}   # Валин
    }
    
    # Извлекаем информацию о мутациях из столбца 'Mutation(s)_cleaned'
    features = []
    labels = []
    
    for idx, row in data.iterrows():
        try:
            # Преобразуем строки в числа
            affinity_mut = pd.to_numeric(row['Affinity_mut (M)'], errors='coerce')
            affinity_wt = pd.to_numeric(row['Affinity_wt (M)'], errors='coerce')
            
            # Проверяем, что значения не NaN
            if pd.isna(affinity_mut) or pd.isna(affinity_wt):
                continue
                
            ddG = affinity_mut - affinity_wt  # Вычисление ddG
            label = 1 if ddG > 0 else 0  # 1 - дестабилизирующая, 0 - стабилизирующая/нейтральная
        
            mutation = row['Mutation(s)_cleaned']
            if pd.isna(mutation):
                continue
        
            # Парсим мутацию (например, "RA123K" -> R на K в позиции 123)
            wild_aa = mutation[0]  # Дикая аминокислота
            mut_aa = mutation[-1]  # Мутантная аминокислота
            
            # Вычисляем разницу в свойствах
            hydrophobicity_diff = amino_acid_properties[mut_aa]['hydrophobicity'] - \
                                 amino_acid_properties[wild_aa]['hydrophobicity']
            charge_diff = amino_acid_properties[mut_aa]['charge'] - \
                          amino_acid_properties[wild_aa]['charge']
            
            features.append([hydrophobicity_diff, charge_diff])
            labels.append(label)
        except (ValueError, KeyError):
            continue
    
    return np.array(features), np.array(labels)

# Функция для обучения модели и визуализации
def train_model(X, y):
    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Инициализируем модель логистической регрессии
    model = LogisticRegression(max_iter=5000)
    
    # Обучаем модель
    model.fit(X_train, y_train)
    
    # Оцениваем точность
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Точность на обучающей выборке: {train_accuracy:.2f}")
    print(f"Точность на тестовой выборке: {test_accuracy:.2f}")
    
    # Визуализируем процесс обучения (лосс для обучающей выборки)
    train_losses = []
    model = LogisticRegression(max_iter=1, warm_start=True)  # warm_start для сохранения весов
    for i in range(50):
        model.fit(X_train, y_train)
        # Вычисляем логарифмическую потерю (log loss) для обучающей выборки
        y_pred_proba = model.predict_proba(X_train)
        train_loss = -np.mean(y_train * np.log(y_pred_proba[:,1]) + (1-y_train) * np.log(y_pred_proba[:,0]))
        train_losses.append(train_loss)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses)
    plt.title("Уменьшение функции потерь во время обучения (обучающая выборка)")
    plt.xlabel("Итерация")
    plt.ylabel("Log Loss")
    plt.grid(True)
    plt.show()
    
    # Визуализируем функции потерь для обучающей и тестовой выборок
    train_losses = []
    test_losses = []
    model = LogisticRegression(max_iter=1, warm_start=True)
    for i in range(50):
        model.fit(X_train, y_train)
        # Лосс для обучающей выборки
        y_train_pred_proba = model.predict_proba(X_train)
        train_loss = -np.mean(y_train * np.log(y_train_pred_proba[:,1]) + (1-y_train) * np.log(y_train_pred_proba[:,0]))
        train_losses.append(train_loss)
        # Лосс для тестовой выборки
        y_test_pred_proba = model.predict_proba(X_test)
        test_loss = -np.mean(y_test * np.log(y_test_pred_proba[:,1]) + (1-y_test) * np.log(y_test_pred_proba[:,0]))
        test_losses.append(test_loss)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Обучающая выборка', color='blue')
    plt.plot(test_losses, label='Тестовая выборка', color='orange')
    plt.title("Сравнение функций потерь (обучающая и тестовая выборки)")
    plt.xlabel("Итерация")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Запуск кода 
if __name__ == "__main__":
    # Загружаем данные
    data = load_skempi_data()
    
    # Извлекаем признаки и метки
    X, y = extract_features(data)
    
    # Проверяем, что данные не пустые
    if X.shape[0] == 0:
        print("Не удалось извлечь признаки. Проверьте данные.")
    else:
        print(f"Извлечено {X.shape[0]} примеров с {X.shape[1]} признаками")
        # Обучаем модель
        train_model(X, y)