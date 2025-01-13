import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from file_operations import file_methods

class KMeansClustering:
    """
    This class shall be used to divide the data into clusters before training.

    Written By: Tejas Jay (TJ)
    Version: 1.0
    Revisions: None

    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object



    def elbow_plot(self, data):
        """
        Method Name: elbow_plot
        Description: This method saves the plot to decide the optimum number of clusters to the file.
        Output: A picture saved to the directory and returns number of clusters
        On Failure: Raise Exception with specific message.
    
        Written By: Tejas Jay (TJ)
        Version: 1.1
        Revisions: Added file path validation, enhanced logging, and improved exception handling.
        """
        self.logger_object.log(self.file_object, 'Entered the elbow_plot method of the KMeansClustering class')
    
        # Validate the input data
        if data is None or len(data) == 0:
            error_message = "Input data is empty or None. Unable to perform elbow plot."
            self.logger_object.log(self.file_object, error_message)
            raise ValueError(error_message)
    
        wcss = []  # initializing an empty list to store Within-Cluster Sum of Squares (WCSS)
    
        try:
            for i in range(1, 11):
                # Initializing the KMeans object
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(data)  # Fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
                self.logger_object.log(self.file_object, f"WCSS for {i} clusters: {kmeans.inertia_}")
    
            # Plotting the WCSS values
            plt.plot(range(1, 11), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
    
            # Check if the directory exists, else create it
            save_dir = 'preprocessing_data'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                self.logger_object.log(self.file_object, f"Created directory: {save_dir}")
    
            # Saving the elbow plot locally
            plot_path = os.path.join(save_dir, 'K-Means_Elbow.PNG')
            plt.savefig(plot_path)
            plt.close()  # Close the plot to free up memory
    
            self.logger_object.log(self.file_object, f'Elbow plot saved successfully at {plot_path}')
    
            # Programmatically find the optimal number of clusters using KneeLocator
            self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            optimum_clusters = self.kn.knee
            self.logger_object.log(self.file_object, f'The optimum number of clusters is: {optimum_clusters}.')
    
            self.logger_object.log(self.file_object, 'Exited the elbow_plot method of the KMeansClustering class')
            return optimum_clusters
    
        except Exception as e:
            error_message = f"Exception occurred in elbow_plot method of the KMeansClustering class. Exception message: {str(e)}"
            self.logger_object.log(self.file_object, error_message)
            self.logger_object.log(self.file_object, 'Finding the number of clusters failed. Exited the elbow_plot method of the KMeansClustering class')
            
            # Raising the exception with the specific error message
            raise Exception(f"Failed to generate elbow plot or determine optimal clusters: {str(e)}")


    def create_clusters(self,data,number_of_clusters):
        """
        Method Name: create_clusters
        Description: Create a new dataframe consisting of the cluster information.
        Output: A datframe with cluster column
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.logger_object.log(self.file_object, 'Entered the create_clusters method of the KMeansClustering class')
        self.data=data
        try:
            self.kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            #self.data = self.data[~self.data.isin([np.nan, np.inf, -np.inf]).any(1)]
            self.y_kmeans=self.kmeans.fit_predict(data) #  divide data into clusters

            self.file_op = file_methods.File_Operation(self.file_object,self.logger_object)
            self.save_model = self.file_op.save_model(self.kmeans, 'KMeans') # saving the KMeans model to directory
                                                                                    # passing 'Model' as the functions need three parameters

            self.data['Cluster']=self.y_kmeans  # create a new column in dataset for storing the cluster information
            self.logger_object.log(self.file_object, 'succesfully created '+str(self.kn.knee)+ 'clusters. Exited the create_clusters method of the KMeansClustering class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in create_clusters method of the KMeansClustering class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Fitting the data to clusters failed. Exited the create_clusters method of the KMeansClustering class')
            raise Exception()
