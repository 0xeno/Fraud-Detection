import pandas as pd
import numpy as np
import joblib

class ModelService:
    """
    This Class covers Business Logic and Prediction.
    """
    def __init__(self):
        # Load your original model (make sure the .pkl file is in the same folder)
        try:
            self.model = joblib.load('fraud_detection_XGBoost_Original.pkl') 
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        self.model_name = "Fraud Detection Model Real"
        
        self.columns = [
            'amt', 'gender', 'city_pop', 'year', 'hour', 'age', 'distance_KM', 'state_encoded',
            'category_entertainment', 'category_food_dining', 'category_gas_transport',
            'category_grocery_net', 'category_grocery_pos', 'category_health_fitness',
            'category_home', 'category_kids_pets', 'category_misc_net', 'category_misc_pos',
            'category_personal_care', 'category_shopping_net', 'category_shopping_pos', 'category_travel',
            'month_April', 'month_August', 'month_December', 'month_February', 'month_January',
            'month_July', 'month_June', 'month_March', 'month_May', 'month_November',
            'month_October', 'month_September',
            'day_Friday', 'day_Monday', 'day_Saturday', 'day_Sunday', 'day_Thursday', 'day_Tuesday', 'day_Wednesday',
            'time category_Afternoon', 'time category_Evening', 'time category_Morning', 'time category_Night',
            'day category_Weekday', 'day category_Weekend',
            'age_category_<25', 'age_category_26 - 40', 'age_category_41 - 60', 'age_category_>60'
        ]

    def preprocess_input(self, raw_input):
        """
        Convert user input that is 'easy to read' into the 51-column format
        required by the model.
        """
        # 1. Create an empty dictionary with all columns set to False/0
        data = {col: 0 for col in self.columns}

        # 2. Fill in the basic numeric fields
        data['amt'] = raw_input['amt']
        data['gender'] = raw_input['gender'] # Asumsi 0=F, 1=M atau sebaliknya
        data['city_pop'] = raw_input['city_pop']
        data['year'] = raw_input['date'].year
        data['hour'] = raw_input['time'].hour
        data['age'] = raw_input['age']
        data['distance_KM'] = raw_input['distance_KM']
        data['state_encoded'] = raw_input['state_encoded']

        # 3. Handling One-Hot Encoding: CATEGORY
        # Example: If the user selects ‘food_dining’, set ‘category_food_dining’ = 1
        cat_key = f"category_{raw_input['category']}"
        if cat_key in data:
            data[cat_key] = 1

        # 4. Handling One-Hot Encoding: MONTH
        # Take the month name from the date input (e.g., “January”)
        month_name = raw_input['date'].strftime('%B') 
        month_key = f"month_{month_name}"
        if month_key in data:
            data[month_key] = 1

        # 5. Handling One-Hot Encoding: DAY
        day_name = raw_input['date'].strftime('%A') # Example: "Monday"
        day_key = f"day_{day_name}"
        if day_key in data:
            data[day_key] = 1

        # 6. Handling TIME CATEGORY (Manual logic)
        hour = raw_input['time'].hour
        if 5 <= hour < 12:
            data['time category_Morning'] = 1
        elif 12 <= hour < 17:
            data['time category_Afternoon'] = 1
        elif 17 <= hour < 21:
            data['time category_Evening'] = 1
        else:
            data['time category_Night'] = 1

        # 7. Handling DAY CATEGORY
        if day_name in ['Saturday', 'Sunday']:
            data['day category_Weekend'] = 1
        else:
            data['day category_Weekday'] = 1

        # 8. Handling AGE CATEGORY
        age = raw_input['age']
        if age < 25:
            data['age_category_<25'] = 1
        elif 25 <= age <= 40:
            data['age_category_26 - 40'] = 1
        elif 41 <= age <= 60:
            data['age_category_41 - 60'] = 1
        else:
            data['age_category_>60'] = 1

        # 9. Return as DataFrame (1 line)
        # Convert to bool if your model is trained using bool data type.
        # But usually int (0/1) is safer. Change .astype(bool) if bool is required.
        df_final = pd.DataFrame([data])
        
        # Ensure the data type is bool for the one-hot column (according to your df.info information).
        bool_cols = [c for c in df_final.columns if 'category' in c or 'month' in c or 'day' in c]
        for c in bool_cols:
            df_final[c] = df_final[c].astype(bool)
            
        return df_final

    def get_dummy_data(self):
        """
        Generate dummy data that has a structure SIMILAR to the original data,
        but in raw format (before One-Hot Encoding) so that it is easy to visualize.
        """
        count = 100
        dates = pd.date_range(end=pd.Timestamp.now(), periods=count)
        
        # List of categories according to dataset
        categories = [
            'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 
            'grocery_pos', 'health_fitness', 'home', 'kids_pets', 'misc_net', 
            'misc_pos', 'personal_care', 'shopping_net', 'shopping_pos', 'travel'
        ]
        
        data = pd.DataFrame({
            'Date': dates,
            'amt': np.random.uniform(10, 1000, size=count),
            'category': np.random.choice(categories, size=count), # Consistent with the input form
            'gender': np.random.choice(['M', 'F'], size=count),
            'age': np.random.randint(18, 80, size=count),
            'Prediction': np.random.choice(['Normal', 'Fraud'], size=count, p=[0.9, 0.1])
        })
        return data

    def predict_single(self, raw_input, threshold=0.5):
        """Receive raw input, process it, then predict"""
        
        # Preprocess data
        input_df = self.preprocess_input(raw_input)
            
        if self.model is not None:
        
        # 1. Make probability/confidence predictions (how confident is the model?)
        # predict_proba returns an array, for example [0.1, 0.9] -> [prob_normal, prob_fraud]
        # take the probability belonging to class 1 (Fraud)
            probability = self.model.predict_proba(input_df)[0][1]
        
        # 2. Return results
            final_prediction = 1 if probability > threshold else 0
        
            return final_prediction, probability
        else:
            # Fallback if the model fails to load (so the app doesn't crash)
            return 0, 0.0

    def process_batch_prediction(self, df, threshold=0.5):
        """
        Make predictions on multiple data sets at once (Batch) using the Original Model.
        """
        # 1. Check whether the model has been loaded
        if self.model is None:
            # Return an empty dataframe or an error if the model is dead
            df['Error'] = "Model not loaded"
            return df

        # 2. Preprocessing Loop
        # convert each row of raw CSV data into a 51-column format
        # to match what the model has learned.
        
        ready_for_model = [] # List of processed rows
        valid_indices = [] # Save the row index that was successfully processed

        # iterate through the dataframe uploaded by the user.
        for index, row in df.iterrows():
            try:
                # A. Preparing Raw Data per Row
                # Ensure that the column names in the user CSV match the keys below.
                # We need to parse the Date & Time from the CSV string into Python objects.
                d_date = pd.to_datetime(row['trans_date']).date() 
                d_time = pd.to_datetime(row['trans_time']).time()
                
                raw_input = {
                    'amt': row['amt'],
                    'category': row['category'],
                    'gender': row['gender'],
                    'age': row['age'],
                    'city_pop': row['city_pop'],
                    'distance_KM': row['distance_KM'],
                    'state_encoded': row['state_encoded'],
                    'date': d_date,
                    'time': d_time
                }

                # B. Call the preprocess_input function that we created earlier.
                # This will convert raw_input -> a 1-row dataframe (51 columns).
                processed_row = self.preprocess_input(raw_input)
                
                
                # C. Add to list
                ready_for_model.append(processed_row)
                valid_indices.append(index)
                
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                # If there is an error line, we can skip it or fill in the default.
                # Here, we just continue.
                continue

        # 3. Combine all processed rows
        if len(ready_for_model) > 0:
            # Creating one large dataframe (e.g., 1000 rows x 51 columns)
            X_batch = pd.concat(ready_for_model, ignore_index=True)
            
            # predict_proba() -> Returns the probability
            probs = self.model.predict_proba(X_batch)[:, 1] # Take column index 1 (Fraud probability)

            # 4. Determine Label based on User THRESHOLD
            predictions = ['Fraud' if p > threshold else 'Normal' for p in probs]
            
            # 5. Return Results to Original DataFrame
            result_df = df.copy()
            # We only update valid rows (in case of row errors).
            result_df.loc[valid_indices, 'Prediction'] = predictions
            result_df.loc[valid_indices, 'Fraud_Probability'] = probs
            
            return result_df
        else:
            return df # Return the original if processing fails

    @staticmethod
    def highlight_fraud(row):
        """
        Styling helper for pandas.
        Highlight the Fraud row in red, with dark red text and bold formatting.
        """
        # We use .get() for security (in case the Prediction column is accidentally missing/does not exist).
        status = row.get('Prediction', 'Normal')
        
        if status == 'Fraud':
            # Pink background, Dark red text, Bold font
            css = 'background-color: #ffcccc; color: #8b0000; font-weight: bold'
            return [css] * len(row)
        else:
            return [''] * len(row)