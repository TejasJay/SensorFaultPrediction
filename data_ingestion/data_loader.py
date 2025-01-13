import pandas as pd

class Data_Getter:
    """
    This class shall  be used for obtaining the data from the source for training.

    Written By: Tejas Jay (TJ)
    Version: 1.0
    Revisions: None

    """
    def __init__(self, file_object, logger_object):
        self.training_file='Training_FileFromDB/InputFile.csv'
        self.file_object=file_object
        self.logger_object=logger_object

    def get_data(self):
        """
        Method Name: get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception with specific message.
    
        Written By: Tejas Jay (TJ)
        Version: 1.1
        Revisions: Added file validation and improved error handling.
    
        """
        self.logger_object.log(self.file_object, 'Entered the get_data method of the Data_Getter class')
        
        # Validate the file path
        if not self.training_file or not isinstance(self.training_file, str):
            error_message = "Invalid file path provided. Please check the training file path."
            self.logger_object.log(self.file_object, error_message)
            raise Exception(error_message)
    
        try:
            # Log the file being used
            self.logger_object.log(self.file_object, f'Reading data from file: {self.training_file}')
            
            # Reading the data file
            self.data = pd.read_csv(self.training_file)
            
            self.logger_object.log(self.file_object, 'Data Load Successful. Exited the get_data method of the Data_Getter class')
            return self.data
    
        except Exception as e:
            error_message = f"Exception occurred in get_data method of the Data_Getter class. Exception message: {str(e)}"
            self.logger_object.log(self.file_object, error_message)
            self.logger_object.log(self.file_object, 'Data Load Unsuccessful. Exited the get_data method of the Data_Getter class')
            
            # Raising the exception with a descriptive message
            raise Exception(f"Data load failed: {str(e)}")

