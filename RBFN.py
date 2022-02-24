import numpy as np 
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

class RBFN:

    def __init__(self, cluster_centers, sigma=None, function='gaussian'):
        """Constructs radial basis functions network with n cluster. If sigma is not passed the same sigma will be calculated for each RBF based on maximum distance between clusters.
        
        Args:
            cluster_centers (array): (nclusters, 3) array
            sigma (float, optional): sigma value. Defaults to None.
            function (string): RBF kernel function
        """

        if sigma is None:

            clusters_dist_mat = cdist(cluster_centers, cluster_centers)
            max_dist = np.max(clusters_dist_mat)
            sigma = max_dist/(np.sqrt(2*len(cluster_centers)))
            self.sigma = np.array([sigma] * len(cluster_centers)).T
            print('Sigma value set')

        self.function = function
        self.cluster_centers = cluster_centers
        self.sigma = np.array([sigma] * len(cluster_centers))
        self.bias = 0
        self.interpolation_matrix = None
        self.dist_mat = None
        self.weights = np.random.randn(len(cluster_centers)+1)
        print('Random weights set')

    def random_bias(self):
        """Set a random bias value between 0 and 1
        """

        self.bias = np.random.uniform(0,1,1)[0]
        print('Random bias: {}'.format(self.bias))

    def _kernel(self, dist, function):
        if function == 'gaussian':
            return np.exp(-1*(dist**2)/(2*self.sigma**2))
        if function == 'multiquadratic':
            return np.sqrt((dist/2*self.sigma**2)**2 + 1)

    def _interpolation_matrix(self, data_points, cluster_centers):
        """Calculates interpolation matrix between data points and cluster centers.

        Args:
            data_points (array): data points coordinates
            cluster_centers (array): cluster centers coordinates

        Returns:
            array: interpolation matrix
        """        

        dist_mat = cdist(data_points, cluster_centers)
        self.dist_mat = dist_mat
        return self._kernel(dist_mat, self.function)

    def fit(self, X, y):
        """Train weights for each RBF by pseudo inverse solution
        
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
        self.int_mat_no_zero = int_mat
        int_mat = np.insert(int_mat, 0, np.ones(int_mat.shape[0]), axis=1)
        self.interpolation_matrix = int_mat
        predictions = np.dot(int_mat, weights) 
        return predictions

    def loss(self, X_train, y_train):
        """Calculates loss function for a train test set
        
        Args:
            X_train (array): (npoints, 3) train coordinates array 
            X_test (array): (npoints, 3) test coordinates array 
            y_train (array): train grades array
            y_test (array): test grades array
        
        Returns:
            float: loss and real minus predicted
        """
        
        predictions = self.predict(X_train)
        loss = sum((np.array(predictions) - np.array(y_train))**2)/2
        real_minus_predicted = np.array(y_train) - np.array(predictions)
        return loss, real_minus_predicted

    def train(self, epochs, X_train, y_train, learning_rate_w=0.001):
        """Trains weights, centers and sigmas by gradient descent
        
        Args:
            epochs (int): number of iterations
            X_train (array): (npoints, 3) train coordinates array 
            X_test (array): (npoints, 3) test coordinates array 
            y_train (array): train grades array
            y_test (array): test grades array
            learning_rate_w (float, optional): leraning rate for weights. Defaults to 0.001.
        """

        losses = []
        
        for epoch in range(epochs):

            loss, real_minus_predicted = self.loss(X_train, y_train)
            losses.append(loss)

            if epoch % 500 == 0:
                print('Epoch: {} \n loss: {}'.format(epoch, np.mean(loss)))


            #trainning weights
            delta_w = np.dot(real_minus_predicted, self.interpolation_matrix)
            self.weights = self.weights - learning_rate_w * np.array(delta_w)

            #training centers
            #later

            #training sigmas
            #later
          
        fig = plt.figure(figsize=(10,5))
        x = [x for x in range(epochs)]
        y = losses
        plt.plot(x, y)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()

        plt.show()

    







