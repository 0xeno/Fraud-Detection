🔬 Model Development Process
Key Highlights from the Notebook:
1. Data Cleaning: Handled missing values and removed duplicate entries.
2. Feature Engineering:
   - Calculated distance between user and merchant using lat and long.
   - Extracted time-based features (Hour, Day, Month) from transaction timestamps.
   - Converted D.O.B to Age.
3. Handling Imbalance: Applied SMOTE strictly within the training fold of Cross-Validation to prevent data leakage.
4. Model Selection: XGBoost was chosen as the final model due to its superior F1-Score and Recall on the minority class.
