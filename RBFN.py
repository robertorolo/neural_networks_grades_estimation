import numpy as np 
from scipy.spatial.distance import cdist
import plotly.offline as pyo
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

def train_test_sets_builder(df, x, y, z, var, test_size=0.3):

    X = df[[x,y,z]].values
    y = df[var].values

    return train_test_split(X, y, test_size=0.33, random_state=42)

def cluster_centers_evaluation(coordinates, max_num_clusters):
        
    inertia = []
    n_clus = np.arange(1,max_num_clusters,1)
    for n in n_clus:
        kmeans = KMeans(n_clusters=n).fit(coordinates)
        inertia.append(kmeans.inertia_)

    traces = []

    trace = {
    'type':'scatter',
    'mode':'lines',
    'x':n_clus,
    'y':inertia,
    }
    
    traces.append(trace)
    
    layout = {
    'title':'inertia',
    }

    fig = go.Figure(traces, layout)

    return pyo.iplot(fig)

def cluster_centers(n_clus, coordinates):

    return KMeans(n_clusters=n_clus).fit(coordinates).cluster_centers_

class RBFN:

    def __init__(self, cluster_centers, sigma=None):

        if sigma is None:

            clusters_dist_mat = cdist(cluster_centers, cluster_centers)
            max_dist = np.max(clusters_dist_mat)
            sigma = 1/(2*max_dist/(np.sqrt(2*len(cluster_centers))))**2
            self.sigma = np.array([sigma] * len(cluster_centers)).T

        self.cluster_centers = cluster_centers
        self.sigma = np.array([sigma] * len(cluster_centers))

    def knn_sigma_definition(self, neighbors_number):
        
        knn = NearestNeighbors(n_neighbors=neighbors_number+1)
        knn.fit(self.cluster_centers)
        distances = knn.kneighbors(self.cluster_centers)[0]
        #self.sigma = np.sqrt((np.sum(distances**2,axis=1))/neighbors_number)
        self.sigma = 1/(2*np.max(distances,axis=1)/np.sqrt(2*neighbors_number))**2

    def gaussian_kernel(self, dist):

        return np.exp(-1*self.sigma*dist**2)

    def interpolation_matrix(self, data_points, cluster_centers):

        dist_mat = cdist(data_points, cluster_centers)
        return self.gaussian_kernel(dist_mat)

    def fit(self, X, Y):

        int_mat = self.interpolation_matrix(X, self.cluster_centers)
        self.weights = np.dot(np.linalg.pinv(int_mat), Y)

    def predict(self, X):

        int_mat = self.interpolation_matrix(X, self.cluster_centers)
        predictions = np.dot(int_mat, self.weights)
        return predictions


