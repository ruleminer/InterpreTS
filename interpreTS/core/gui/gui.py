import streamlit as st
import pandas as pd
import numpy as np
import time
from interpreTS.core.feature_extractor import FeatureExtractor, Features
from interpreTS.core.time_series_data import TimeSeriesData

class InterpreTSApp:
    def __init__(self):
        extractor = FeatureExtractor()
        self.data = None
        self.time_column = None
        self.value_column = None
        self.selected_features = []
        self.feature_options = extractor.generate_feature_options()


    def configure_page(self):
        st.set_page_config(page_title="InterpreTS Feature Extraction", layout="wide")
        st.title("InterpreTS Feature Extraction GUI")
        st.write("This app allows you to upload a CSV file containing time series data and extract interpretable features using the InterpreTS library.")
        st.write("""
                The CSV file should have two columns: one for time and one for values. (the order doesn't matter beacuse they can be switched). Example format:
                """)
        
        # Create a sample DataFrame for better visualisation
        sample_data = pd.DataFrame({
            'time': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'value': [100, 110, 105]
        })
        # Display the DataFrame as a table
        st.table(sample_data)


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


    def windows_slider(self):
        if self.data is not None:
            if len(self.data) > 1000:
                # Use text input for large datasets
                window_size = st.number_input(
                    'Enter a window size',
                    min_value=1,
                    max_value=len(self.data),
                    value=3,
                    step=1
                )
            else:
                # Use slider for smaller datasets
                window_size = st.slider(
                    'Select a window size',
                    min_value=1,
                    max_value=len(self.data),
                    value=3,
                    step=1
                )
            return window_size
        return None


    def stride_slider(self):
        if self.data is not None:
            if len(self.data) > 1000:
                # Use text input for large datasets
                stride_size = st.number_input(
                    'Enter a stride',
                    min_value=1,
                    max_value=len(self.data),
                    value=3,
                    step=1
                )
            else:
                # Use slider for smaller datasets
                stride_size = st.slider(
                    'Select a stride',
                    min_value=1,
                    max_value=len(self.data),
                    value=3,
                    step=1
                )
            return stride_size
        return None


    def select_time_value_columns(self):
        if self.data is not None:
            st.sidebar.header("Step 2: Select Columns")
            self.time_column = st.sidebar.selectbox("Select Time Column", options=self.data.columns, index=0)
            self.value_column = st.sidebar.selectbox("Select Value Column", options=self.data.columns, index=1)

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


    def extract_features(self, window_size=1, stride=1):
        if self.data is not None and self.time_column and self.value_column:
            if st.sidebar.button("Extract Features"):
                if self.selected_features:
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(progress):
                            progress_bar.progress(progress)
                            status_text.text(f"Progress: {progress}%")

                        extractor = FeatureExtractor(
                            features=[self.feature_options[feat] for feat in self.selected_features],
                            window_size=window_size,
                            stride=stride
                        )

                        status_text.text("Preparing data...")
                        ts_data = self.data.set_index(self.time_column)[self.value_column].dropna()

                        status_text.text("Calculating features...")
                        feature_df = extractor.extract_features(
                            pd.DataFrame({'value': ts_data.values}),
                            progress_callback=update_progress,
                            mode='sequential'  
                        )

                        progress_bar.progress(100)
                        st.success("Features extracted successfully!")
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
                window_size = self.windows_slider()
                stride = self.stride_slider()
                st.write(f"Selected window size: {window_size}")
                st.write(f"Selected stride size: {stride}")

                if self.select_time_value_columns():
                    self.select_features()
                    self.extract_features(window_size, stride)
        else:
            st.info("Please upload a CSV file to get started.")

        # Footer
        st.sidebar.markdown("---")
        st.sidebar.write("Developed with ❤️ using Streamlit and InterpreTS.")

#TODO
# def start_gui():
#     app = InterpreTSApp()
#     app.run()
#     return app

if __name__ == "__main__":
    app = InterpreTSApp()
    app.run()
