import streamlit as st
import pandas as pd
import numpy as np
from interpreTS.core.feature_extractor import FeatureExtractor, Features
from interpreTS.core.time_series_data import TimeSeriesData

class InterpreTSApp:
    def __init__(self):
        self.data = None
        self.time_column = None
        self.value_column = None
        self.selected_features = []
        self.feature_options = {
            "Length": Features.LENGTH,
            "Mean": Features.MEAN,
            "Variance": Features.VARIANCE,
            #"SeasonalityStrength": Features.CALCULATE_SEASONALITY_STRENGTH,
            # "BinarizeMean": Features.BINARIZE_MEAN,
            # "Peak": Features.PEAK,
            # "Spikeness": Features.SPIKENESS,
            # "Entropy": Features.ENTROPY,
            # "Stability": Features.STABILITY,
            # "AbsoluteEnergy": Features.ABSOLUTE_ENERGY,
            # "FlatSpots": Features.FLAT_SPOTS,
            # "CrossingPoints": Features.CROSSING_POINTS,
            # "MissingPoints": Features.MISSING_POINTS,
            # "Trough": Features.TROUGH,
            # "Std1stDer": Features.STD_1ST_DER,
        }

    def configure_page(self):
        st.set_page_config(page_title="InterpreTS Feature Extraction", layout="wide")
        st.title("InterpreTS Feature Extraction GUI")
        st.write("This app allows you to upload a CSV file containing time series data and extract interpretable features using the InterpreTS library.")

    def sidebar_upload(self):
        st.sidebar.header("Step 1: Upload CSV File")
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file containing time series data", type=["csv"])
        if uploaded_file is not None:
            try:
                self.data = pd.read_csv(uploaded_file)
                return True
            except Exception as e:
                st.error(f"Error reading the file: {e}")
        return False

    def preprocess_data(self):
        if self.data is not None:
            st.write("Data Preview:")
            st.write(self.data.head())
            # Normalize columns
            self.data.columns = self.data.columns.str.strip().str.lower()
            st.write("Normalized Columns:", ", ".join(self.data.columns))
            return True
        return False

    def select_time_value_columns(self):
        if self.data is not None:
            st.sidebar.header("Step 2: Select Columns")
            self.time_column = st.sidebar.selectbox("Select Time Column", options=self.data.columns)
            self.value_column = st.sidebar.selectbox("Select Value Column", options=self.data.columns)

            # Validate columns
            if self.time_column not in self.data.columns or self.value_column not in self.data.columns:
                st.error(f"Invalid columns selected: {self.time_column}, {self.value_column}")
                return False

            # Convert time column to datetime
            try:
                self.data[self.time_column] = pd.to_datetime(self.data[self.time_column], errors='coerce')
                self.data = self.data.dropna(subset=[self.time_column])
                if self.data[self.time_column].dtype != 'datetime64[ns]':
                    st.error(f"The time column '{self.time_column}' could not be converted to datetime.")
                    return False
                return True
            except Exception as e:
                st.error(f"Error processing the time series data: {e}")
                return False
        return False

    def select_features(self):
        st.sidebar.header("Step 3: Select Features")
        self.selected_features = st.sidebar.multiselect(
            "Choose features to extract", 
            options=self.feature_options.keys(), 
            default=["Length", "Mean", "Variance"]
        )

    def extract_features(self):
        # Check if we have all prerequisites
        if self.data is not None and self.time_column and self.value_column:
            if st.sidebar.button("Extract Features"):
                if self.selected_features:
                    try:
                        # Initialize extractor
                        extractor = FeatureExtractor(features=[self.feature_options[feat] for feat in self.selected_features])

                        # We need a DataFrame with a single 'value' column
                        # Set the time column as index for clarity (TimeSeriesData expects this format)
                        ts_data = self.data.set_index(self.time_column)[self.value_column]
                        ts_data = ts_data.dropna()

                        # Extract features
                        # According to interpreTS, extractor expects a DataFrame with a single value column
                        feature_df = extractor.extract_features(pd.DataFrame({'value': ts_data.values}))

                        st.subheader("Extracted Features")
                        st.write("The following features were successfully extracted:")
                        st.dataframe(feature_df)
                    except Exception as e:
                        st.error(f"An error occurred while extracting features: {e}")
                else:
                    st.sidebar.error("Please select at least one feature to extract.")

    def run(self):
        self.configure_page()
        file_uploaded = self.sidebar_upload()

        if file_uploaded:
            if self.preprocess_data():
                if self.select_time_value_columns():
                    self.select_features()
                    self.extract_features()
        else:
            st.info("Please upload a CSV file to get started.")

        # Footer
        st.sidebar.markdown("---")
        st.sidebar.write("Developed with ❤️ using Streamlit and InterpreTS.")


# Entry point
if __name__ == "__main__":
    app = InterpreTSApp()
    app.run()
