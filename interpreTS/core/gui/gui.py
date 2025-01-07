import streamlit as st
import pandas as pd
from ..feature_extractor import FeatureExtractor
from ...utils.feature_loader import FeatureLoader

class InterpreTSApp:
    def __init__(self):
        loader = FeatureLoader()
        self.data = None
        self.time_column = None
        self.value_columns = []  # Allow for multiple value columns
        self.selected_features = []
        self.feature_options = loader.generate_feature_options()
        self.sep = ','  # Default separator

    def configure_page(self):
        st.set_page_config(page_title="InterpreTS Feature Extraction", layout="wide")
        st.title("InterpreTS Feature Extraction GUI")
        st.write("This app allows you to upload a CSV file containing time series data and extract interpretable features using the InterpreTS library.")
        st.write("""
                The CSV file should have at least two columns: one for time and one (or more) for values. 
                The order doesn't matter because they can be switched. Example format:
                """)
        
        # Create a sample DataFrame for better visualization
        sample_data = pd.DataFrame({
            'time': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'value_1': [100, 110, 105],
            'value_2': [5, 10, 15],
            'value_<n>': [8, 16, 24]
        })
        # Display the DataFrame as a table
        st.table(sample_data)

    def sidebar_upload(self):
        # Step 1: Select CSV separator
        st.sidebar.header("Step 1: Upload CSV File")
        self.sep = st.sidebar.selectbox("Select CSV Separator", [",", ";", "tab", "space"], index=0)

        # File uploader
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file containing time series data", type=["csv"])
        if uploaded_file is not None:
            try:
                if self.sep == "tab":
                    self.sep = "\t"
                elif self.sep == "space":
                    self.sep = " "
                self.data = pd.read_csv(uploaded_file, sep=self.sep)
                return True
            except Exception as e:
                st.error(f"Error reading the file with the selected separator '{self.sep}': {e}")
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
            self.value_columns = st.sidebar.multiselect(
                "Select Value Columns",
                options=[col for col in self.data.columns if col != self.time_column],
                default=[self.data.columns[1]] if len(self.data.columns) > 1 else []
            )

            # Validate columns
            if self.time_column not in self.data.columns or not self.value_columns:
                st.error(f"Invalid columns selected: Time: {self.time_column}, Value(s): {self.value_columns}")
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
        if self.data is not None and self.time_column and self.value_columns:
            if st.sidebar.button("Extract Features"):
                if self.selected_features:
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(progress):
                            progress_bar.progress(progress)
                            status_text.text(f"Progress: {progress}%")

                        # Dictionary to store feature DataFrames for each value column
                        feature_dfs = []

                        for column in self.value_columns:
                            extractor = FeatureExtractor(
                                features=[self.feature_options[feat] for feat in self.selected_features],
                                window_size=window_size,
                                stride=stride
                            )

                            status_text.text(f"Preparing data for column: {column}...")
                            ts_data = self.data.set_index(self.time_column)[column].dropna()

                            status_text.text(f"Calculating features for column: {column}...")
                            feature_df = extractor.extract_features(
                                pd.DataFrame({'value': ts_data.values}),
                                progress_callback=update_progress,
                                mode='sequential'
                            )

                            # Optional: Prefix feature columns to distinguish them by column name
                            # e.g., 'Mean' becomes 'colname_Mean'
                            feature_df = feature_df.add_prefix(f"{column}_")

                            feature_dfs.append(feature_df)

                        progress_bar.progress(100)
                        st.success("Features extracted successfully!")

                        # Combine all feature DataFrames side-by-side
                        combined_features = pd.concat(feature_dfs, axis=1)

                        # Display only a preview of the combined table (e.g., first 50 rows)
                        st.subheader("Extracted Features (Preview)")
                        st.dataframe(combined_features.head(50))

                        # Allow user to download the entire combined table as a CSV
                        csv_data = combined_features.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Full Extracted Features as CSV",
                            data=csv_data,
                            file_name="extracted_features.csv",
                            mime="text/csv"
                        )

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

if __name__ == "__main__":
    app = InterpreTSApp()
    app.run()
