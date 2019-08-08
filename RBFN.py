import numpy as np 
from scipy.spatial.distance import cdist
import plotly.offline as pyo
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

def train_test_sets_builder(df, x, y, z, var, test_size=0.33):
    """Creates train test datasets
    
    Args:
        df (DataFrame): pandas dataframe
        x (str): x coordinates column name
        y (str): y coordinates column name
        z (str): z coordinates column name
        var (str): variable column name
        test_size (float, optional): test set size. Defaults to 0.33.
    
    Returns:
        arrays: X_train, X_test, y_train, y_test
    """

    X = df[[x,y,z]].values
    y = df[var].values
    return train_test_split(X, y, test_size=test_size, random_state=42)

def cluster_centers_evaluation(coordinates, max_num_clusters):
    """Plots different metrix against number of clusters. inertia, calinski harabasz, silhoute score and davies douldin.
    
    Args:
        coordinates (array): (npoints, 3) array
        max_num_clusters (int): maximum number of clusters
    
    Returns:
        iplot: metrics against number of clusters plot
    """
        
    inertia = []
    calisnki = []
    silhouete = []
    davies = []
    n_clus = np.arange(1,max_num_clusters,1)
    for n in n_clus:
        kmeans = KMeans(n_clusters=n).fit(coordinates)
        inertia.append(kmeans.inertia_)
        calisnki.append(calinski_harabasz_score(coordinates, kmeans.labels_)) if n != 1 else calisnki.append(float('nan'))
        silhouete.append(silhouette_score(coordinates, kmeans.labels_, metric='euclidean')) if n != 1 else silhouete.append(float('nan'))
        davies.append(davies_bouldin_score(coordinates, kmeans.labels_)) if n != 1 else davies.append(float('nan'))

    traces = []

    trace = {
    'type':'scatter',
    'mode':'lines',
    'x':n_clus,
    'y':inertia,
    'name':'inertia'
    }
    
    traces.append(trace)

    trace = {
    'type':'scatter',
    'mode':'lines',
    'x':n_clus,
    'y':calisnki,
    'name':'Calinski-Harabasz' 
    }
    
    traces.append(trace)

    trace = {
    'type':'scatter',
    'mode':'lines',
    'x':n_clus,
    'y':silhouete,
    'name':'Silhouette Coefficient' 
    }
    
    traces.append(trace)

    trace = {
    'type':'scatter',
    'mode':'lines',
    'x':n_clus,
    'y':davies,
    'name':'Davies-Bouldin' 
    }
    
    traces.append(trace)
    
    layout = {
    'title':'Indexes',
    'yaxis':{'title':''},
    'xaxis':{'title':'Number of clusters'}
    }

    fig = go.Figure(traces, layout)

    return pyo.iplot(fig)

def cluster_centers(n_clus, coordinates):
    """Generates RBF cluster centers.
    
    Args:
        n_clus (int): number of clusters
        coordinates (array): (npoints, 3) array
    
    Returns:
        array: (nclusters, 3) array
    """

    return KMeans(n_clusters=n_clus).fit(coordinates).cluster_centers_

class RBFN:

    def __init__(self, cluster_centers, sigma=None):
        """Constructs radial basis functions network with n cluster. If sigma is not passed the same sigma will be calculated for each RBF based on maximum distance between clusters.
        
        Args:
            cluster_centers (array): (nclusters, 3) array
            sigma (float, optional): sigma value. Defaults to None.
        """

        if sigma is None:

            clusters_dist_mat = cdist(cluster_centers, cluster_centers)
            max_dist = np.max(clusters_dist_mat)
            sigma = 1/(2*max_dist/(np.sqrt(2*len(cluster_centers))))**2
            self.sigma = np.array([sigma] * len(cluster_centers)).T
            #print('Sigma vector: {}'.format(self.sigma))

        self.cluster_centers = cluster_centers
        self.sigma = np.array([sigma] * len(cluster_centers))
        self.bias = 0
        self.interpolation_matrix = None

    def knn_sigma_definition(self, neighbors_number):
        """Calculates a different sigma value for each RBF based on k nearest neighbors.
        
        Args:
            neighbors_number (int): number of nearest neighbors
        """
        
        knn = NearestNeighbors(n_neighbors=neighbors_number+1)
        knn.fit(self.cluster_centers)
        distances = knn.kneighbors(self.cluster_centers)[0]
        self.sigma = 1/(2*np.max(distances,axis=1)/np.sqrt(2*neighbors_number))**2
        #print('Sigma vector: {}'.format(self.sigma))

    def random_bias(self):
        """Set a random bias value between 0 and 1
        """

        self.bias = np.random.uniform(0,1,1)[0]
        print('Random bias: {}'.format(self.bias))

    def _gaussian_kernel(self, dist):

        return np.exp(-1*self.sigma*dist**2)

    def _interpolation_matrix(self, data_points, cluster_centers):

        dist_mat = cdist(data_points, cluster_centers)
        return self._gaussian_kernel(dist_mat)

    def fit(self, X, y):
        """Train weights for each RBF by pseudo inverse method
        
        Args:
            X (array): (npoints, 3) coordinates array
            y (array): grades values vector
        """

        int_mat = self._interpolation_matrix(X, self.cluster_centers)
        weights = np.dot(np.linalg.pinv(int_mat), y)
        weights = np.insert(weights, 0, self.bias)
        self.weights = weights

    def predict(self, X):
        """Predicts grades based on trained weights
        
        Args:
            X (array): (npoints, 3) estimation coordinates array 
        
        Returns:
            array: estimated grades array
        """
        
        weights = self.weights
        int_mat = self._interpolation_matrix(X, self.cluster_centers)
        int_mat = np.insert(int_mat, 0, np.ones(int_mat.shape[0]), axis=1)
        self.interpolation_matrix = int_mat
        predictions = np.dot(int_mat, weights) 
        return predictions

    def loss(self, X_train, X_test, y_train, y_test):
        """Calculates loss function for a train test set
        
        Args:
            X_train (array): (npoints, 3) train coordinates array 
            X_test (array): (npoints, 3) test coordinates array 
            y_train (array): train grades array
            y_test (array): test grades array
        
        Returns:
            [type]: [description]
        """
        
        predictions = self.predict(X_test)
        loss = mean_squared_error(predictions, y_test)/2
        real_minus_predicted = predictions - y_test
        return loss, real_minus_predicted

    def train(self, epochs, X_train, X_test, y_train, y_test, learning_rate_w=0.001, learning_rate_c=0.001, learning_rate_sigma=0.001):
        """Trains weights, centers and sigmas by gradient descent
        
        Args:
            epochs (int): number of iterations
            X_train (array): (npoints, 3) train coordinates array 
            X_test (array): (npoints, 3) test coordinates array 
            y_train (array): train grades array
            y_test (array): test grades array
            learning_rate_w (float, optional): leraning rate for weights. Defaults to 0.001.
            learning_rate_c (float, optional): learning rate for centers. Defaults to 0.001.
            learning_rate_sigma (float, optional): learning rate for sigma. Defaults to 0.001.
        
        Returns:
            [type]: [description]
        """

        losses = []
        
        for epoch in range(epochs):
        
            loss, real_minus_predicted = self.loss(X_train, X_test, y_train, y_test, self.weights)
            losses.append(loss)

            delta_w = -1 * learning_rate_w * np.dot(real_minus_predicted, self.interpolation_matrix)
            self.weights = self.weights + delta_w

            '''delta_c = -1 * learning_rate_c * np.dot(real_minus_predicted, self.weights)
            self.cluster_centers = self.cluster_centers + delta_c

            delta_sigma = -1 * learning_rate_c * np.dot(real_minus_predicted, self.weights)
            self.sigma = self.sigma + delta_sigma'''

            #print('Epoch: {} \n Loss: {}'.format(epoch, loss))

        traces = []

        trace = {
        'type':'scatter',
        'mode':'lines',
        'x':[x for x in range(epochs)],
        'y':losses,
        'name':'weights'
        }
        
        traces.append(trace)

        layout = {
        'title':'Training',
        'yaxis':{'title':'Loss'},
        'xaxis':{'title':'epoch'}
        }

        fig = go.Figure(traces, layout)

        return pyo.iplot(fig)






