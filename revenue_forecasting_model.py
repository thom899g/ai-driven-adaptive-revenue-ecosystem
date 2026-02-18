import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RevenueForecastingModel:
    def __init__(self):
        self.model: Optional[RandomForestRegressor] = None
        self.data_prepared: bool = False

    def _prepare_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for model training by handling missing values and encoding categorical variables."""
        logger.info("Preparing data for revenue forecasting model.")
        # Handle missing values with forward fill
        data = raw_data.fillna(method='ffill')
        # Convert categorical variables to one-hot encoding
        data = pd.get_dummies(data)
        return data

    def train_model(self, data: pd.DataFrame) -> None:
        """Train the revenue forecasting model using prepared data."""
        logger.info("Training revenue forecasting model.")
        if not self.data_prepared:
            self._prepare_data(data)
        
        # Splitting the data
        X = data.drop('revenue', axis=1)
        y = data['revenue']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Training the model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        logger.info(f"Model trained with {len(X_train)} samples.")

    def predict_revenue(self, new_data: pd.DataFrame) -> pd.Series:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        logger.info("Generating revenue forecast.")
        # Prepare new data
        prepared_new_data = self._prepare_data(new_data)
        # Make predictions
        predictions = pd.Series(self.model.predict(prepared_new_data), index=new_data.index, name='predicted_revenue')
        return predictions

    def evaluate_model(self) -> dict:
        """Evaluate the model's performance."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        logger.info("Evaluating revenue forecasting model.")
        X = pd.read_csv('test_data.csv')
        y_true = X['revenue']
        y_pred = self.predict_revenue(X.drop('revenue', axis=1))
        # Calculate metrics
        rmse = ((y_pred - y_true) ** 2).mean() ** 0.5
        r2 = (y_pred.corr(y_true))**2
        
        return {
            'RMSE': rmse,
            'R-squared': r2
        }