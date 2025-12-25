# ML5

Я выбрал MLFlow <br>
 <br>
Этапы пайпа:  <br>
1.  <br>
```
# generate_data.py
import pandas as pd
import numpy as np
import mlflow
from datetime import datetime, timedelta
import os

print("=" * 60)
print("ШАГ 1: ГЕНЕРАЦИЯ ДАННЫХ")
print("=" * 60)

# Настройка MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("customer_churn_pipeline")

# Создаем run для этого шага
with mlflow.start_run(run_name="data_generation") as run:
    print("Настройка окружения...")
    
    # Параметры генерации
    np.random.seed(42)
    n_customers = 1000
    
    print(f"Создание данных для {n_customers} клиентов...")
    
    # 1. Основные данные клиентов
    customer_data = {
        'customer_id': range(n_customers),
        'age': np.random.randint(18, 70, n_customers),
        'income': np.random.normal(50000, 15000, n_customers),
        'balance': np.random.exponential(1000, n_customers),
        'tenure': np.random.randint(1, 60, n_customers),
        'support_calls': np.random.poisson(3, n_customers),
    }
    
    df_customers = pd.DataFrame(customer_data)
    
    # Целевая переменная (churn)
    df_customers['churn'] = ((df_customers['support_calls'] > 5) | 
                            (df_customers['tenure'] < 3)).astype(int)
    
    # 2. Данные для Feature Store (Feast)
    print("Создание данных для Feast Feature Store...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    dates = pd.date_range(start_date, end_date, freq='D')
    feast_records = []
    
    for customer_id in range(n_customers):
        # Для каждого клиента создаем записи за последние 30 дней
        for date in dates[-30:]:
            if np.random.random() > 0.7: 
                feast_records.append({
                    'customer': customer_id,
                    'avg_transaction': np.random.uniform(10, 500),
                    'transaction_count': np.random.poisson(15),
                    'credit_score': np.random.randint(300, 850),
                    'loyalty_score': np.random.uniform(0, 1),
                    'event_timestamp': date,
                    'created_timestamp': date
                })
    
    df_feast = pd.DataFrame(feast_records)
    
    # 3. Сохранение данных
    print("Сохранение данных")
    os.makedirs('data/raw', exist_ok=True)
    
    df_customers.to_csv('data/raw/customers.csv', index=False)
    df_feast.to_parquet('data/raw/customer_features.parquet')
    
    # 4. Логирование в MLflow
    mlflow.log_param("n_customers", n_customers)
    mlflow.log_param("n_feast_records", len(df_feast))
    mlflow.log_metric("churn_rate", df_customers['churn'].mean())
    mlflow.log_metric("avg_income", df_customers['income'].mean())
    mlflow.log_metric("avg_balance", df_customers['balance'].mean())
    
    # Сохраняем образцы данных как артефакты
    mlflow.log_artifact('data/raw/customers.csv')
    
    # 5. Вывод результатов
    print("\n РЕЗУЛЬТАТЫ ГЕНЕРАЦИИ:")
    print(f"   Клиентов: {n_customers}")
    print(f"   Churn rate: {df_customers['churn'].mean():.2%}")
    print(f"   Средний доход: ${df_customers['income'].mean():.0f}")
    print(f"   Средний баланс: ${df_customers['balance'].mean():.0f}")
    print(f"   Feast записей: {len(df_feast)}")
    print(f"   Сохранено: data/raw/customers.csv")
    print(f"   Сохранено: data/raw/customer_features.parquet")
    
    print("\n ДАННЫЕ УСПЕШНО СОЗДАНЫ!")

```

2. <br>
```
# validate_data.py
import pandas as pd
import numpy as np
import mlflow

print("=" * 60)
print("ШАГ 2: ВАЛИДАЦИЯ ДАННЫХ")
print("=" * 60)

# Настройка MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("customer_churn_pipeline")

with mlflow.start_run(run_name="data_validation") as run:
    print("Загрузка данных...")

    try:
        df = pd.read_csv('data/raw/customers.csv')
        print(f"Загружено {len(df)} записей")
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        raise

    print("\n ПРОВЕРКА ДАННЫХ:")

    # Список проверок
    checks = {
        "min_records": len(df) >= 100,
        "has_customer_id": 'customer_id' in df.columns,
        "has_target": 'churn' in df.columns,
        "no_nulls": df.isnull().sum().sum() == 0,
        "churn_rate_range": 0.05 <= df['churn'].mean() <= 0.95,
        "positive_balance": (df['balance'] >= 0).all(),
        "valid_age": ((df['age'] >= 18) & (df['age'] <= 100)).all(),
    }

    # Выполняем проверки
    all_passed = True
    for check_name, check_result in checks.items():
        status = "true" if check_result else "false"
        print(f"   {status} {check_name}: {check_result}")

        # Логируем результат проверки
        mlflow.log_param(f"check_{check_name}", check_result)

        if not check_result:
            all_passed = False

    # Дополнительные метрики
    mlflow.log_metric("total_records", len(df))
    mlflow.log_metric("churn_rate", df['churn'].mean())
    mlflow.log_metric("avg_age", df['age'].mean())
    mlflow.log_metric("avg_income", df['income'].mean())

    print("\n СТАТИСТИКА ДАННЫХ:")
    print(f"  Churn rate: {df['churn'].mean():.2%}")
    print(f"  Средний возраст: {df['age'].mean():.1f} лет")
    print(f"  Средний доход: ${df['income'].mean():.0f}")
    print(f"  Средний баланс: ${df['balance'].mean():.0f}")
    print(f"  Среднее кол-во обращений: {df['support_calls'].mean():.1f}")

    if all_passed:
        print("\n ВАЛИДАЦИЯ ПРОЙДЕНА УСПЕШНО!")
    else:
        print("\n ВАЛИДАЦИЯ НЕ ПРОЙДЕНА!")
        raise ValueError("Одна или несколько проверок не пройдены")

```


3. <br>
```
# feature_engineering.py
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

print("=" * 60)
print(" ШАГ 3: FEATURE ENGINEERING С FEATURE STORE")
print("=" * 60)

# Настройка MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("customer_churn_pipeline")

with mlflow.start_run(run_name="feature_engineering") as run:
    print(" Загрузка данных клиентов...")
    df_customers = pd.read_csv('data/raw/customers.csv')

    print(" Интеграция с Feature Store (Feast)...")

    try:
        # Имитация работы с Feast Feature Store
        print(" Подключение к Feast Feature Store...")

        # Загружаем фичи из файла (имитация Feast)
        df_features = pd.read_parquet('data/raw/customer_features.parquet')

        # Берем последние значения для каждого клиента
        df_latest_features = df_features.sort_values('event_timestamp') \
                                       .groupby('customer') \
                                       .last() \
                                       .reset_index()

        # Объединяем данные
        df_combined = pd.merge(
            df_customers,
            df_latest_features,
            left_on='customer_id',
            right_on='customer'
        )

        mlflow.log_param("feature_store", "feast_integrated")
        mlflow.log_param("external_features", 4)  # 4 фичи из Feast
        print(" Feast фичи успешно загружены")

    except Exception as e:
        print(f"  Ошибка при работе с Feast: {e}")
        print("  Используем только базовые фичи...")
        df_combined = df_customers.copy()
        mlflow.log_param("feature_store", "fallback")

    print("\n ПОДГОТОВКА ФИЧЕЙ:")

    # Список фичей для обучения
    base_features = ['age', 'income', 'balance', 'tenure', 'support_calls']
    external_features = ['avg_transaction', 'transaction_count', 'credit_score', 'loyalty_score']

    # Выбираем только доступные фичи
    available_features = []
    for feature in base_features + external_features:
        if feature in df_combined.columns:
            available_features.append(feature)

    X = df_combined[available_features]
    y = df_combined['churn']

    print(f"   Используется {len(available_features)} фичей:")
    for feature in available_features:
        print(f"   • {feature}")

    print("\n МАСШТАБИРОВАНИЕ И РАЗДЕЛЕНИЕ ДАННЫХ:")

    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")

    print("\n СОХРАНЕНИЕ ДАННЫХ:")

    # Создаем директории
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Сохраняем данные
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)

    # Сохраняем scaler
    joblib.dump(scaler, 'models/scaler.pkl')

    # Сохраняем список фичей
    with open('data/processed/feature_names.txt', 'w') as f:
        for feature in available_features:
            f.write(f"{feature}\n")

    # Логирование в MLflow
    mlflow.log_param("n_features", len(available_features))
    mlflow.log_param("train_samples", X_train.shape[0])
    mlflow.log_param("test_samples", X_test.shape[0])
    mlflow.log_artifact('models/scaler.pkl')
    mlflow.log_artifact('data/processed/feature_names.txt')

    print(f" Данные сохранены в data/processed/")
    print(f" Scaler сохранен в models/scaler.pkl")

    print("\n FEATURE ENGINEERING ЗАВЕРШЕН!")

```

4. <br>
```
#train.py
import argparse
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import json
import os
from datetime import datetime

print("=" * 60)
print(" ШАГ 4: ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

# Настройка MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("customer_churn_pipeline")

def train_model(model_type, params, run_suffix, X_train, y_train):
    """Обучение одной модели с заданными параметрами"""
    run_name = f"{model_type}_{run_suffix}"

    with mlflow.start_run(run_name=run_name, nested=True) as run:
        print(f"\n  Обучение {model_type} с параметрами: {params}")

        # Создаем модель
        if model_type == "random_forest":
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(**params, random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**params, random_state=42)
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

        # Кросс-валидация
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()

        # Обучение на всех тренировочных данных
        model.fit(X_train, y_train)

        # Логирование параметров
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_param("model_type", model_type)

        # Логирование метрик
        mlflow.log_metric("cv_accuracy_mean", mean_score)
        mlflow.log_metric("cv_accuracy_std", std_score)

        # Сохранение модели
        mlflow.sklearn.log_model(model, "model")

        print(f" {model_type} обучена:")
        print(f" Accuracy: {mean_score:.4f} ± {std_score:.4f}")
        print(f" Run ID: {run.info.run_id}")

        return {
            "model_type": model_type,
            "params": params,
            "cv_accuracy": mean_score,
            "cv_std": std_score,
            "run_id": run.info.run_id,
            "timestamp": datetime.now().isoformat()
        }

def main():
    parser = argparse.ArgumentParser(description="Обучение моделей машинного обучения")
    parser.add_argument("--model-type", type=str, default="random_forest",
                       choices=["random_forest", "gradient_boosting", "logistic_regression", "all"],
                       help="Тип модели для обучения")
    args = parser.parse_args()

    print(f"Модель для обучения: {args.model_type}")

    # Загружаем данные ДО определения функций
    print(" Загрузка тренировочных данных...")
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')

    print(f"   Данные: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Определяем какие модели обучать
    model_types_to_train = []
    if args.model_type == "all":
        model_types_to_train = ["random_forest", "gradient_boosting", "logistic_regression"]
    else:
        model_types_to_train = [args.model_type]

    # Параметры для каждой модели
    params_grid = {
        "random_forest": [
            {"n_estimators": 50, "max_depth": 5},
            {"n_estimators": 100, "max_depth": 10},
            {"n_estimators": 200, "max_depth": 20},
            {"n_estimators": 300, "max_depth": None}
        ],
        "gradient_boosting": [
            {"n_estimators": 50, "learning_rate": 0.01, "max_depth": 3},
            {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
            {"n_estimators": 200, "learning_rate": 0.3, "max_depth": 7}
        ],
        "logistic_regression": [
            {"C": 0.01, "max_iter": 1000},
            {"C": 0.1, "max_iter": 1000},
            {"C": 1.0, "max_iter": 1000},
            {"C": 10.0, "max_iter": 1000}
        ]
    }

    all_results = []

    with mlflow.start_run(run_name="training_experiments") as parent_run:
        print(f"\n НАЧАЛО ЭКСПЕРИМЕНТОВ")
        print(f"Всего экспериментов: {sum(len(params_grid[mt]) for mt in model_types_to_train)}")

        for model_type in model_types_to_train:
            print(f"\n ЭКСПЕРИМЕНТЫ С {model_type.upper()}:")

            model_results = []
            for i, params in enumerate(params_grid[model_type], 1):
                run_suffix = f"exp{i}"
                result = train_model(model_type, params, run_suffix, X_train, y_train)
                model_results.append(result)

            # Сохраняем результаты для этого типа модели
            os.makedirs('data/training', exist_ok=True)
            results_file = f'data/training/{model_type}_results.json'
            with open(results_file, 'w') as f:
                json.dump(model_results, f, indent=2)

            print(f" Результаты сохранены в {results_file}")
            all_results.extend(model_results)

        # Находим лучшую модель
        if all_results:
            best_result = max(all_results, key=lambda x: x['cv_accuracy'])

            print(f"\n ЛУЧШАЯ МОДЕЛЬ:")
            print(f"   Тип: {best_result['model_type']}")
            print(f"   Параметры: {best_result['params']}")
            print(f"   Accuracy: {best_result['cv_accuracy']:.4f}")
            print(f"   Run ID: {best_result['run_id']}")

            # Сохраняем информацию о лучшей модели
            with open('data/training/best_model.json', 'w') as f:
                json.dump(best_result, f, indent=2)

            mlflow.log_param("best_model_type", best_result['model_type'])
            mlflow.log_metric("best_cv_accuracy", best_result['cv_accuracy'])

            print(f"\n ОБУЧЕНИЕ ЗАВЕРШЕНО!")
            print(f" Всего обучено моделей: {len(all_results)}")
            print(f" Результаты в data/training/")
        else:
            print(" Нет результатов обучения")

if __name__ == "__main__":
    main()

```

5. <br>
```
#evaluate.py 
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from mlflow.tracking import MlflowClient
import json
import os

print("=" * 60)
print(" ШАГ 5: ОЦЕНКА МОДЕЛИ И MODEL REGISTRY")
print("=" * 60)

# Настройка MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("customer_churn_pipeline")

def load_best_model():
    """Загрузка лучшей модели из результатов обучения"""
    print(" Поиск лучшей модели...")

    # Читаем все результаты и выбираем лучшую модель
    model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
    all_results = []

    for model_type in model_types:
        results_file = f'data/training/{model_type}_results.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
                all_results.extend(results)

    if not all_results:
        print(" Нет результатов обучения")
        raise ValueError("Файлы с результатами обучения не найдены")

    # Выбираем лучшую модель по accuracy
    best_result = max(all_results, key=lambda x: x['cv_accuracy'])

    print(f"   Найдена лучшая модель: {best_result['model_type']}")
    print(f"   Accuracy: {best_result['cv_accuracy']:.4f}")
    print(f"   Run ID: {best_result['run_id']}")

    # Загружаем модель из MLflow
    model_uri = f"runs:/{best_result['run_id']}/model"
    model = mlflow.sklearn.load_model(model_uri)

    return model, best_result

def evaluate_model(model, X_test, y_test):
    """Оценка модели на тестовых данных"""
    print("\n ТЕСТИРОВАНИЕ МОДЕЛИ:")

    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    # Метрики
    accuracy = float(accuracy_score(y_test, y_pred))  # Преобразуем в обычный float
    roc_auc = float(roc_auc_score(y_test, y_pred_proba))  # Преобразуем в обычный float

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)

    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    ROC-AUC: {roc_auc:.4f}")
    print(f"    Матрица ошибок:")
    print(f"    True Negatives: {cm[0, 0]}")
    print(f"    False Positives: {cm[0, 1]}")
    print(f"    False Negatives: {cm[1, 0]}")
    print(f"    True Positives: {cm[1, 1]}")

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist()
    }

def check_production_metrics(client, model_name):
    """Проверка метрик production модели"""
    print("\n ПРОВЕРКА PRODUCTION МОДЕЛИ:")

    try:
        # Игнорируем deprecation warning
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)

        # Пытаемся найти production модель
        try:
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])

            if prod_versions:
                current_prod = prod_versions[0]
                current_run = client.get_run(current_prod.run_id)

                current_accuracy = float(current_run.data.metrics.get("test_accuracy", 0))
                current_roc_auc = float(current_run.data.metrics.get("test_roc_auc", 0))

                print(f"    Найдена production модель:")
                print(f"    Версия: {current_prod.version}")
                print(f"    Accuracy: {current_accuracy:.4f}")
                print(f"    ROC-AUC: {current_roc_auc:.4f}")

                return current_accuracy, current_roc_auc
        except Exception as e:
            # Если модель не найдена, это нормально для первого запуска
            pass

        print("    Production модель не найдена (первый запуск)")
        return 0.0, 0.0

    except Exception as e:
        print(f"    Ошибка при проверке production модели: {e}")
        return 0.0, 0.0

def main():
    # Загружаем тестовые данные
    print(" Загрузка тестовых данных...")
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')

    print(f"   Тестовые данные: {X_test.shape[0]} samples")

    # Загружаем лучшую модель
    model, best_model_info = load_best_model()

    # Оцениваем модель
    metrics = evaluate_model(model, X_test, y_test)

    # Инициализируем MLflow клиент
    client = MlflowClient()
    model_name = "customer_churn_classifier"

    with mlflow.start_run(run_name="model_evaluation") as run:
        print(f"\n УСЛОВНОЕ ВЫПОЛНЕНИЕ (CONDITIONAL EXECUTION):")

        # Проверяем текущую production модель
        current_accuracy, _ = check_production_metrics(client, model_name)

        # Определяем, нужно ли деплоить новую модель
        new_accuracy = metrics["accuracy"]
        accuracy_improvement = float(new_accuracy - current_accuracy)  # Преобразуем в float

        # Условие: улучшение accuracy минимум на 2%
        should_deploy = new_accuracy > current_accuracy + 0.02

        print(f"    Сравнение метрик:")
        print(f"    Текущая production: {current_accuracy:.4f}")
        print(f"    Новая модель: {new_accuracy:.4f}")
        print(f"    Улучшение: {accuracy_improvement:.4f}")
        print(f"    Порог улучшения: 2% (0.02)")

        if should_deploy:
            print(f"   Улучшение ≥ 2%: ДЕПЛОЙ РАЗРЕШЕН")
        else:
            print(f"   Улучшение < 2%: ДЕПЛОЙ ОТКЛОНЕН")

        # Логируем метрики
        mlflow.log_metric("test_accuracy", new_accuracy)
        mlflow.log_metric("test_roc_auc", metrics["roc_auc"])
        mlflow.log_metric("accuracy_improvement", accuracy_improvement)
        mlflow.log_param("should_deploy", bool(should_deploy))  # Преобразуем в bool
        mlflow.log_param("deployment_threshold", "2% improvement")

        #  MODEL REGISTRY ИНТЕГРАЦИЯ
        print(f"\n ИНТЕГРАЦИЯ С MODEL REGISTRY:")

        try:
            # Создаем или получаем зарегистрированную модель
            try:
                existing_model = client.get_registered_model(model_name)
                print(f"    Модель '{model_name}' уже в Registry")
            except:
                client.create_registered_model(model_name)
                print(f"   Создана модель '{model_name}' в Registry")

            # Создаем новую версию модели
            model_uri = f"runs:/{best_model_info['run_id']}/model"

            model_version = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=best_model_info['run_id']
            )

            print(f"   Создана версия {model_version.version}")

            # Добавляем описание
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=f"{best_model_info['model_type']}. Test Accuracy: {new_accuracy:.4f}"
            )

            # Добавляем теги (используем строки вместо bool)
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="feature_store",
                value="feast_integrated"
            )

            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="accuracy",
                value=f"{new_accuracy:.4f}"
            )

           
            registry_info = {
                "model_name": model_name,
                "version": int(model_version.version), 
                "metrics": {
                    "accuracy": float(new_accuracy),  
                    "roc_auc": float(metrics["roc_auc"])  
                },
                "model_type": best_model_info['model_type'],
                "should_deploy": bool(should_deploy),  
                "accuracy_improvement": float(accuracy_improvement),  
                "run_id": best_model_info['run_id']
            }

            with open('data/registry_info.json', 'w') as f:
                json.dump(registry_info, f, indent=2, default=str) 

            print(f" Информация сохранена в data/registry_info.json")

        except Exception as e:
            print(f" Ошибка при работе с Model Registry: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n ОЦЕНКА ЗАВЕРШЕНА!")

        # Возвращаем should_deploy для conditional execution
        with open('data/should_deploy.txt', 'w') as f:
            f.write(str(should_deploy))

if __name__ == "__main__":
    main()

```

6. <br>
```
# deploy.py
import json
from mlflow.tracking import MlflowClient
import os
import mlflow
import sqlite3
from datetime import datetime
import time

print("=" * 60)
print(" ШАГ 6: DEPLOYMENT С CONDITIONAL EXECUTION")
print("=" * 60)

def check_deployment_condition():
    """Проверка условия для деплоя"""
    print(" ПРОВЕРКА УСЛОВИЯ ДЕПЛОЯ:")

    try:
        with open('data/should_deploy.txt', 'r') as f:
            content = f.read().strip()
            print(f"   Содержимое файла: '{content}'")
            should_deploy = content.lower() == 'true'

        if should_deploy:
            print("    Условие выполнено: ДЕПЛОЙ РАЗРЕШЕН")
        else:
            print("     Условие не выполнено: ДЕПЛОЙ ОТКЛОНЕН")

        return should_deploy

    except FileNotFoundError:
        print("    Файл should_deploy.txt не найден")
        return False
    except Exception as e:
        print(f"    Ошибка при чтении файла: {e}")
        return False

def load_registry_info():
    """Загрузка информации о зарегистрированной модели"""
    try:
        with open('data/registry_info.json', 'r') as f:
            content = f.read()
            if not content.strip():
                print(" Файл registry_info.json пустой")
                return None
            return json.loads(content)
    except FileNotFoundError:
        print(" Файл registry_info.json не найден")
        return None
    except json.JSONDecodeError as e:
        print(f" Ошибка чтения JSON: {e}")
        return create_default_registry_info()
    except Exception as e:
        print(f" Ошибка при загрузке информации: {e}")
        return None

def create_default_registry_info():
    """Создание информации о регистрации по умолчанию"""
    return {
        "model_name": "customer_churn_classifier",
        "version": 1,
        "should_deploy": True
    }

def deploy_via_direct_db(model_name, version):
    """Деплой через прямое обновление базы данных SQLite"""
    print(f"  Используем прямой доступ к БД для деплоя...")

    try:
        # Подключаемся к базе данных MLflow
        db_path = os.path.join(os.getcwd(), "mlflow.db")

        if not os.path.exists(db_path):
            print(f"    Файл базы данных не найден: {db_path}")
            return False

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Получаем ID модели
        cursor.execute(
            "SELECT name, creation_timestamp FROM registered_models WHERE name = ?",
            (model_name,)
        )
        model_result = cursor.fetchone()

        if not model_result:
            print(f"    Модель '{model_name}' не найдена в БД")
            conn.close()
            return False

        print(f"    Модель найдена в БД: {model_result[0]}")

        # Получаем ID версии
        cursor.execute(
            "SELECT version, current_stage FROM model_versions WHERE name = ? AND version = ?",
            (model_name, version)
        )
        version_result = cursor.fetchone()

        if not version_result:
            print(f"    Версия {version} модели '{model_name}' не найдена")
            conn.close()
            return False

        print(f"    Версия найдена: {version_result[0]}, текущая стадия: {version_result[1]}")

        # Обновляем стадию в базе данных
        cursor.execute(
            "UPDATE model_versions SET current_stage = 'Production' WHERE name = ? AND version = ?",
            (model_name, version)
        )

        # Архивируем другие версии в Production
        cursor.execute(
            "UPDATE model_versions SET current_stage = 'Archived' WHERE name = ? AND version != ? AND current_stage = 'Production'",
            (model_name, version)
        )

        conn.commit()
        conn.close()

        print(f"    Версия {version} обновлена до Production в БД")
        return True

    except Exception as e:
        print(f"    Ошибка при работе с БД: {e}")
        return False

def deploy_to_production(client, model_name, version):
    """Деплой модели в Production"""
    print(f"\n ЗАПУСК ДЕПЛОЯ:")
    print(f"   Модель: {model_name}")
    print(f"   Версия: {version}")

    try:
        # Используем метод через БД
        db_success = deploy_via_direct_db(model_name, version)

        if db_success:
            # Пробуем установить теги через клиент (но не критично если не получится)
            try:
                client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="deployed",
                    value="true"
                )
                print("    Тег 'deployed' установлен")
            except:
                print("     Не удалось установить тег через клиент (не критично)")

            return True, datetime.now().isoformat()
        else:
            # Если не получилось через БД, пробуем через deprecated метод
            print("     Пробуем через старый метод...")
            try:
                # Перемещаем модель в Production (deprecated метод, но может сработать)
                client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage="Production",
                    archive_existing_versions=True
                )
                print(f"    Модель {version} перемещена в Production")
                return True, datetime.now().isoformat()
            except Exception as e:
                print(f"    Ошибка при деплое через старый метод: {e}")
                return False, None

    except Exception as e:
        print(f"    Ошибка при деплое: {e}")
        return False, None

def main():
    # Проверяем условие для деплоя
    should_deploy = check_deployment_condition()

    if not should_deploy:
        print("\n  DEPLOYMENT ПРОПУЩЕН (CONDITIONAL EXECUTION)")
        print("   Причина: метрики не улучшились на 2% или условие не выполнено")
        return

    # Загружаем информацию о модели
    registry_info = load_registry_info()
    if not registry_info:
        print("  Используем значения по умолчанию")
        registry_info = create_default_registry_info()

    model_name = registry_info.get("model_name", "customer_churn_classifier")
    version = registry_info.get("version", 1)

    print(f"\n ИНФОРМАЦИЯ О МОДЕЛИ:")
    print(f"   Название: {model_name}")
    print(f"   Версия: {version}")
    print(f"   Тип модели: {registry_info.get('model_type', 'N/A')}")

    if 'metrics' in registry_info:
        accuracy = registry_info['metrics'].get('accuracy', 'N/A')
        if isinstance(accuracy, (int, float)):
            print(f"   Accuracy: {accuracy:.4f}")
        else:
            print(f"   Accuracy: {accuracy}")

    # Инициализируем MLflow клиент
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()

    # Даем время на создание модели
    print(" Ожидание создания модели в registry...")
    time.sleep(3)

    # Выполняем деплой
    success, deploy_time = deploy_to_production(client, model_name, version)

    if success:
        # Сохраняем информацию о деплое
        deployment_info = {
            "model_name": model_name,
            "version": int(version),
            "stage": "production",
            "deployed": True,
            "deployment_timestamp": deploy_time,
            "metrics": registry_info.get("metrics", {}),
            "deployment_type": "conditional_approval",
            "condition": "accuracy_improvement_2%",
            "deployment_method": "direct_db_update"
        }

        os.makedirs('data', exist_ok=True)
        with open('data/deployment.json', 'w') as f:
            json.dump(deployment_info, f, indent=2, default=str)

        print(f"\n Информация о деплое сохранена в data/deployment.json")

        print(f"\n DEPLOYMENT УСПЕШНО ВЫПОЛНЕН!")
        print(f" Модель {model_name} v{version} теперь в Production!")

        print(f"\n МЕТРИКИ PRODUCTION МОДЕЛИ:")
        if 'metrics' in registry_info:
            for metric, value in registry_info['metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.4f}")
                else:
                    print(f"   {metric}: {value}")

        print(f"\n РУКОВОДСТВО ПО ИСПОЛЬЗОВАНИЮ:")
        print(f"   1. Откройте MLflow UI: http://localhost:5000")
        print(f"   2. Перейдите в Models -> {model_name}")
        print(f"   3. Найдите версию {version} со стадией 'Production'")
        print(f"   4. Используйте URI модели для предсказаний:")
        print(f"      mlflow.pyfunc.load_model('models:/{model_name}/{version}')")

        print(f"\n ДЛЯ ЗАПУСКА SCHEDULED RETRAINING:")
        print(f"   ./scheduled_retraining.sh")
    else:
        print(f"\n DEPLOYMENT НЕ УДАЛСЯ")

        # Создаем минимальный deployment.json для отчетности
        deployment_info = {
            "model_name": model_name,
            "version": version,
            "deployed": False,
            "error": "Ошибка при деплое, но модель зарегистрирована",
            "registry_info": registry_info
        }

        with open('data/deployment.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)

if __name__ == "__main__":
    main()

```

Файл airflow/dag <br>
```
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'ml_team',
    'start_date': datetime(2025, 12, 1),
    'retries': 1,
}

with DAG(
    'ml_retraining_pipeline',
    default_args=default_args,
    description='Scheduled ML Pipeline Retraining',
    schedule_interval='*/5 * * * *',
    catchup=False,
    tags=['mlflow', 'retraining'],
) as dag:

    # Запуск
    run_ml_pipeline = BashOperator(
        task_id='run_ml_pipeline',
        bash_command="""
        cd /home/va_zik/ml5 && \
        source venv_mlflow/bin/activate && \
        cd /home/va_zik/ml5/mlflow_ml_pipeline && \
        ./run_fixed.sh
        """,
    )

    # Логирование
    log_completion = BashOperator(
        task_id='log_completion',
        bash_command='echo "$(date): ML pipeline completed" >> /home/va_zik/ml5/mlflow_ml_pipeline/airflow/pipeline_runs.log',
    )

    run_ml_pipeline >> log_completion

```



ИТОГ: <br>

<img width="375" height="236" alt="image" src="https://github.com/user-attachments/assets/a7d9539f-7e27-4cdb-8026-853a92bcb691" /> <br>


<img width="594" height="288" alt="image" src="https://github.com/user-attachments/assets/ca237521-8bf6-4972-b722-7d475a80436f" /> <br>

<img width="496" height="117" alt="image" src="https://github.com/user-attachments/assets/0b962d58-acd3-43f4-8310-66200b3eb364" /> <br>

Вот запустился след пайп <br>

<img width="398" height="276" alt="image" src="https://github.com/user-attachments/assets/aeb49efd-37de-4dfc-a4db-87391f1e9629" /> <br>

<img width="539" height="160" alt="image" src="https://github.com/user-attachments/assets/1ef885b1-da63-4b03-959c-362d3d4dc4ff" /> <br>

И прошлая модель ушла в архив <br>

<img width="531" height="118" alt="image" src="https://github.com/user-attachments/assets/9208e8c7-4fdb-4e34-a4d5-42238df77467" /> <br>















