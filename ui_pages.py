import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

# --- Class Dashboard ---
class DashboardPage:
    def __init__(self, model_service):
        self.service = model_service

    def show(self):
        st.title("📊 Model Performance Overview")
        
        df_dummy = self.service.get_dummy_data()
        total_trans = len(df_dummy)
        total_fraud = len(df_dummy[df_dummy['Prediction'] == 'Fraud'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", f"{total_trans:,}", "+120 today")
        col2.metric("Fraud Detected", f"{total_fraud}", "-2 from avg", delta_color="inverse")
        col3.metric("Model F1-Score", "0.92", "+0.01")
        
        st.markdown("---")
        
        c1, c2 = st.columns((2, 1))
        with c1:
            st.subheader("Transaction Trend")
            # Keep using the new ‘Date’ and 'amt'
            fig_trend = px.line(df_dummy, x='Date', y='amt', color='Prediction', 
                                color_discrete_map={'Normal': 'blue', 'Fraud': 'red'})
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with c2:
            st.subheader("Fraud by Category") 
            
            fraud_only = df_dummy[df_dummy['Prediction'] == 'Fraud']
            if not fraud_only.empty:
                fraud_by_cat = fraud_only['category'].value_counts().reset_index()
                fraud_by_cat.columns = ['Category', 'Count']
                fraud_by_cat['Category'] = fraud_by_cat['Category'].str.replace('_', ' ').str.title()
                fig_bar = px.bar(fraud_by_cat, x='Count', y='Category', orientation='h',
                                title="Top Fraud Categories")
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No fraud data generated in dummy.")

# --- Class Single Prediction ---
class SinglePredictionPage:
    def __init__(self, model_service):
        self.service = model_service

    def show(self, threshold=0.5):
        st.title("🔍 Single Transaction Analysis")
        st.markdown(f"Current Model Threshold: **{threshold}**") # Visual information to users
        st.info("Enter the transaction details below. The system will automatically convert the date and time data.")
        
        with st.form("prediction_form"):
            # --- GROUP 1: Transaction Details ---
            st.subheader("1. Transaction Details")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 1. NUMBER INPUT: Cannot use text placeholders, so use CAPTION below it.
                amt = st.number_input("Amount ($)", min_value=0.0, step=10.0)
                st.caption("Example: 150.5 (Use a period)") 
            
            with col2:
                category_options = [
                    'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 
                    'grocery_pos', 'health_fitness', 'home', 'kids_pets', 'misc_net', 
                    'misc_pos', 'personal_care', 'shopping_net', 'shopping_pos', 'travel'
                ]
                # 2. SELECTBOX: You can use Placeholder (index=None to make it empty first)
                category = st.selectbox(
                    "Category", 
                    category_options,
                    index=None,  # IMPORTANT: So that the default is empty
                    placeholder="Select category...", # Gray text inside
                    format_func=lambda x: x.replace('_', ' ').title()
                )
            
            with col3:
                gender_select = st.selectbox(
                    "Gender", 
                    ["Female", "Male"],
                    index=None,
                    placeholder="Select gender..."
                )
                gender = 1 if gender_select == "Male" else 0

            # --- GROUP 2: User Profile ---
            st.subheader("2. Profil & Location")
            col4, col5, col6 = st.columns(3)
            with col4:
                age = st.number_input("Age", min_value=10, max_value=100)
                st.caption("Range: 10 - 100 Years")
                
            with col5:
                city_pop = st.number_input("City Population", min_value=0, step=1000)
                st.caption("Example: 10000 (Residents)")
                
            with col6:
                distance = st.number_input("Distance (KM)", min_value=0.0)
                st.caption("Distance between user and merchant")
            
            # --- GROUP 3: Technical & Time ---
            st.subheader("3. Technical & Time")
            col7, col8, col9 = st.columns(3)
            with col7:
                state_encoded = st.number_input("State Code", min_value=0, max_value=50)
                st.caption("State Code (0-50)")
                
            with col8:
                date_input = st.date_input("Transaction Date", datetime.date.today())
                st.caption("Transaction Date") # Opsional
                
            with col9:
                time_input = st.time_input("Transaction Time", datetime.datetime.now().time())
                st.caption("Transaction Time") # Opsional
                
            st.markdown("---")
            submit_btn = st.form_submit_button("Analyze Transaction", type="primary")
        
        if submit_btn:
            # Validate empty input for selectbox (because we set index=None)
            if category is None or gender_select is None:
                st.error("❌ Please select Category and Gender first.")
            else:
                # Wrap the input into a dictionary
                raw_input = {
                    'amt': amt,
                    'category': category,
                    'gender': gender,
                    'age': age,
                    'city_pop': city_pop,
                    'distance_KM': distance,
                    'state_encoded': state_encoded,
                    'date': date_input,
                    'time': time_input
                }
                
                prediction, confidence = self.service.predict_single(raw_input, threshold=threshold)
                self._display_result(prediction, confidence, amt)

    def _display_result(self, prediction, confidence, amount):
        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ **WARNING: FRAUD DETECTED**")
            st.markdown(f"**Confidence Score:** {confidence*100:.1f}%")
            st.write(f"Transactions amounting to ${amount} this is suspicious based on the model pattern.")
        else:
            st.success("✅ **TRANSACTION SAFE**")
            st.markdown(f"**Confidence Score:** {confidence*100:.1f}%")

# --- Class Batch Prediction ---
class BatchPredictionPage:
    def __init__(self, model_service):
        self.service = model_service

    def show(self, threshold=0.5):
        st.title("📂 Batch Prediction (CSV)")
        
        # Info Visual Threshold
        st.markdown(f"Current Model Sensitivity: **{threshold}**")
        if threshold < 0.3:
            st.caption("⚠️ High Sensitivity Mode: More fraud detected, risk of false alarms increases.")
        # --- CSV FORMAT INFORMATION ---
        with st.expander("ℹ️  Click to view CSV File Format Requirements"):
            st.markdown("""
            For predictions to run smoothly, make sure your CSV file has the following **Column Names**:
            
            | Column Name | Data Type | Example |
            | :--- | :--- | :--- |
            | `amt` | Number | `150.50` |
            | `category` | Text | `food_dining`, `travel` |
            | `gender` | Number (0/1) | `1` (Male), `0` (Female) |
            | `age` | Number | `35` |
            | `city_pop` | Number | `20000` |
            | `distance_KM` | Number | `5.2` |
            | `state_encoded`| Number | `1` |
            | `trans_date` | Date (YYYY-MM-DD) | `2024-01-31` |
            | `trans_time` | Time (HH:MM:SS) | `14:30:00` |
            """)
            
            # Optional: Download empty template button
            # You can create an empty dictionary and then convert it to CSV
            sample_data = pd.DataFrame(columns=[
                'amt', 'category', 'gender', 'age', 'city_pop', 
                'distance_KM', 'state_encoded', 'trans_date', 'trans_time'
            ])
            csv_template = sample_data.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Blank CSV Template",
                data=csv_template,
                file_name='template_transaksi.csv',
                mime='text/csv',
            )

        st.markdown("---")

        # --- UPLOAD & PREDICTION ---
        uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview Data Upload:")
                st.dataframe(df.head())
                
                # Minimum Column Validation
                required_cols = ['amt', 'category', 'trans_date', 'trans_time']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ CSV file is missing a column: {', '.join(missing_cols)}")
                else:
                    if st.button("Predict All Data"):
                        with st.spinner('Analyzing transactions...'):
                            result_df = self.service.process_batch_prediction(df, threshold=threshold)
                            self._display_results(result_df)
                            
            except Exception as e:
                st.error(f"Error reading file: {e}")

    def _display_results(self, df):
        st.success("✅ Analysis Complete!")
        
        # Tab to separate Summary and Details
        tab1, tab2 = st.tabs(["📊 Summary", "📄 Data Details"])
        
        with tab1:
            total = len(df)
            fraud = len(df[df['Prediction'] == 'Fraud'])
            st.metric("Total Fraud Found", f"{fraud} / {total} Transactions")
            
        with tab2:
            show_fraud = st.checkbox("Show only Fraud", value=True)
            display_df = df[df['Prediction'] == 'Fraud'] if show_fraud else df
            
            st.dataframe(display_df.style.apply(self.service.highlight_fraud, axis=1))
            
            csv = display_df.to_csv(index=False).encode('utf-8')

            st.download_button("Download Analysis Results", csv, "fraud_report.csv", "text/csv")



