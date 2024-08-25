import warnings
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from xgboost import XGBRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def calculate_adjusted_r2(r2, n, k):
    """Calculate adjusted R² score."""
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def train_models():
    logging.info("Started model training")

    try:
        # Load transformed datasets
        X_train = np.load('artifacts/X_train.npy', allow_pickle=True)
        y_train = np.load('artifacts/y_train.npy', allow_pickle=True)
        X_test = np.load('artifacts/X_test.npy', allow_pickle=True)
        y_test = np.load('artifacts/y_test.npy', allow_pickle=True)
        logging.info("Loaded transformed datasets")

        models = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Lasso': {
                'model': Lasso(random_state=42),
                'params': {
                    'alpha': uniform(0.01, 100)
                }
            },
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': uniform(0.01, 100)
                }
            },
            'ElasticNet': {
                'model': ElasticNet(random_state=42),
                'params': {
                    'alpha': uniform(0.01, 100),
                    'l1_ratio': uniform(0.1, 0.9)
                }
            },
            'DecisionTreeRegressor': {
                'model': DecisionTreeRegressor(random_state=42),
                'params': {
                    'max_depth': [None, 3, 5, 7, 10],
                    'min_samples_split': randint(2, 11),
                    'min_samples_leaf': randint(1, 5)
                }
            },
            'RandomForestRegressor': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': randint(50, 201),
                    'max_depth': [None, 3, 5, 7, 10],
                    'min_samples_split': randint(2, 11),
                    'min_samples_leaf': randint(1, 5)
                }
            },
            'AdaBoostRegressor': {
                'model': AdaBoostRegressor(random_state=42),
                'params': {
                    'n_estimators': randint(50, 201),
                    'learning_rate': uniform(0.01, 1.0)
                }
            },
            'GradientBoostingRegressor': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': randint(50, 201),
                    'learning_rate': uniform(0.01, 0.2),
                    'max_depth': randint(3, 8)
                }
            },
            'XGBRegressor': {
                'model': XGBRegressor(random_state=42),
                'params': {
                    'n_estimators': randint(50, 201),
                    'learning_rate': uniform(0.01, 0.2),
                    'max_depth': randint(3, 8)
                }
            },
            'KNeighborsRegressor': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': randint(3, 10),
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
        }

        # Function to evaluate a regression model
        def evaluate_model(model, params, X_train, y_train, X_test, y_test):
            logging.info(f"Starting RandomizedSearchCV for model: {model.__class__.__name__}")
            # Perform randomized search
            grid_search = RandomizedSearchCV(model, params, cv=5, scoring='r2', n_jobs=-1, random_state=42)
            grid_search.fit(X_train, y_train)
            
            logging.info(f"Completed RandomizedSearchCV for model: {model.__class__.__name__}")
            
            # Get the best model and predictions
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            # Calculate R² and adjusted R²
            r2 = r2_score(y_test, y_pred)
            n = len(y_test)
            k = len(best_model.get_params()) - 1
            adj_r2 = calculate_adjusted_r2(r2, n, k)

            logging.info(f"Model: {best_model}")
            logging.info(f"Best parameters: {grid_search.best_params_}")
            logging.info(f"R² Score: {r2}")
            logging.info(f"Adjusted R² Score: {adj_r2}")
            logging.info("=======================================")

            return best_model, adj_r2

        adjusted_r2_scores = {}
        all_models = {}

        for model_name, model_info in models.items():
            logging.info(f"Evaluating and Training {model_name}...")
            best_model, adj_r2 = evaluate_model(model_info['model'], model_info['params'], X_train, y_train, X_test, y_test)
            adjusted_r2_scores[model_name] = adj_r2
            all_models[model_name] = best_model

            # Save the model
            with open(f"artifacts/{model_name}.pkl", 'wb') as file:
                pickle.dump(best_model, file)
            logging.info(f"Model {model_name} saved to artifacts/{model_name}.pkl")

        # Plotting the adjusted R² comparison
        model_names = list(adjusted_r2_scores.keys())
        adj_r2_values = list(adjusted_r2_scores.values())

        # Generate a list of colors for each bar
        colors = plt.cm.get_cmap('tab20', len(model_names))

        # Create a bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, adj_r2_values, color=[colors(i) for i in range(len(model_names))])

        # Add labels and title
        plt.xlabel('Model Names')
        plt.ylabel('Adjusted R² Score')
        plt.title('Model Adjusted R² Score Comparison')
        plt.xticks(rotation=45, ha='right')

        # Show adjusted R² values on top of the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom')

        # Display the chart
        plt.tight_layout()
        plt.savefig('artifacts/model_adjusted_r2_comparison.png')
        plt.show()

        logging.info("Model training and evaluation completed")

    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_models()
