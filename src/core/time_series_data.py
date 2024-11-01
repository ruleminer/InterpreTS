import pandas as pd

class TimeSeriesData:
    """
    A class to manage and process time series data.
    """

    def __init__(self, data):
        """
        Initialize the TimeSeriesData with time series data.
        
        Parameters
        ----------
        data : pd.Series, pd.DataFrame, or np.ndarray
            The time series data to be managed. If provided as a numpy array,
            it will be converted to a pandas DataFrame for consistency.
            
        Examples
        --------
        >>> import pandas as pd
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> ts_data = TimeSeriesData(data)
        """
        
        
        if isinstance(data, (pd.Series, pd.DataFrame)):
            self.data = data
        else:
            raise ValueError("Data must be a pandas Series or DataFrame.")
    
    def resample(self, interval):
        """
        Resample the time series data to a specified interval.
        
        Parameters
        ----------
        interval : str
            The interval to resample the data, e.g., 'D' for daily, 'H' for hourly.
            
        Returns
        -------
        TimeSeriesData
            A new TimeSeriesData object with resampled data.
            
        Examples
        --------
        >>> data = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2023-01-01", periods=5, freq="D"))
        >>> ts_data = TimeSeriesData(data)
        >>> resampled_data = ts_data.resample("2D")
        """
        
        
        if isinstance(self.data, pd.Series) and self.data.index.is_all_dates:
            resampled_data = self.data.resample(interval).mean()
        else:
            raise ValueError("Data must have a DateTime index for resampling.")
        
        return TimeSeriesData(resampled_data)

    def split(self, train_size=0.7):
        """
        Split the time series data into training and test sets.
        
        Parameters
        ----------
        train_size : float, optional
            The proportion of the data to use for training, by default 0.7.
            
        Returns
        -------
        tuple of TimeSeriesData
            A tuple containing the training and test sets as TimeSeriesData objects.
            
        Examples
        --------
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> ts_data = TimeSeriesData(data)
        >>> train, test = ts_data.split(0.6)
        """
        
        
        split_index = int(len(self.data) * train_size)
        train_data = self.data[:split_index]
        test_data = self.data[split_index:]
        
        return TimeSeriesData(train_data), TimeSeriesData(test_data)
