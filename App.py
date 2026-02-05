import streamlit as st
from model_service import ModelService
from ui_pages import DashboardPage, SinglePredictionPage, BatchPredictionPage

class FraudApp:
    def __init__(self):
        # 1. Setup Config
        st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")
        self._load_css()
        
        # 2. Initialize Logic Service
        self.model_service = ModelService()
        
        # 3. Initialize Pages (Inject Service into every page)
        self.pages = {
            "Dashboard": DashboardPage(self.model_service),
            "Single Prediction": SinglePredictionPage(self.model_service),
            "Batch Prediction": BatchPredictionPage(self.model_service)
        }

    def _load_css(self):
        st.markdown("""
        <style>
            .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 20px; }
            .stAlert { font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        with st.sidebar:
            st.title("🛡️ FraudShield AI")
            st.markdown("---")
            page_selection = st.radio("Navigation", list(self.pages.keys()))
            
            st.markdown("---")
            st.subheader("Model Settings")
            
            # SLIDER THRESHOLD
            # value=0.5 makes the default value 0.5 when first opened
            threshold = st.slider("Sensitivity Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            
            # Display visual information
            if threshold < 0.3:
                st.warning("⚠️ Low Sensitivity: A lot of fraud may slip through.")
            elif threshold > 0.8:
                st.warning("⚠️ High Sensitivity: Beware of many false alarms.")
            else:
                st.info(f"Active Threshold: **{threshold}**")
            
            # WE RETURN TWO VALUES: Page & Threshold
            return page_selection, threshold

    def run(self):
        # Capture both values from the sidebar
        selected_page_name, current_threshold = self.render_sidebar()
        page = self.pages[selected_page_name]
        
        if selected_page_name in ["Single Prediction", "Batch Prediction"]:
            page.show(current_threshold)
        else:
            page.show()

if __name__ == "__main__":
    app = FraudApp()

    app.run()

