import numpy as np
from scipy.spatial.distance import cdist
import math

def _coordinates_transform(coords, major_med, major_min, azimuth, dip, rake):

    if azimuth >= 0 and azimuth <=270:
        alpha = math.radians(90-azimuth)
    else:
        alpha = math.radians(450-azimuth)
    beta = -math.radians(dip)
    phi = math.radians(rake)
    
    rot_matrix = np.zeros((3,3))

    rot_matrix[0,0] = math.cos(beta)*math.cos(alpha)
    rot_matrix[0,1] = math.cos(beta)*math.sin(alpha)
    rot_matrix[0,2] = -math.sin(beta)
    rot_matrix[1,0] = major_med*(-math.cos(phi)*math.sin(alpha)+math.sin(phi)*math.sin(beta)*math.cos(alpha))
    rot_matrix[1,1] = major_med*(math.cos(phi)*math.cos(alpha)+math.sin(phi)*math.sin(beta)*math.sin(alpha))
    rot_matrix[1,2] = major_med*(math.sin(phi)*math.cos(beta))
    rot_matrix[2,0] = major_min*(math.sin(phi)*math.sin(alpha)+math.cos(phi)*math.sin(beta)*math.cos(alpha))
    rot_matrix[2,1] = major_min*(-math.sin(phi)*math.cos(alpha)+math.cos(phi)*math.sin(beta)*math.sin(alpha))
    rot_matrix[2,2] = major_min*(math.cos(phi)*math.cos(beta))

    return np.array([np.dot(rot_matrix, i) for i in coords])

class RBF:

    def __init__(self, support, major_med, major_min, azimuth, dip, rake):

        self.support = support
        self.major_med = major_med
        self.major_min = major_min
        self.azimuth = azimuth
        self.dip = dip
        self.rake = rake
        self.weights = None
        self.X = None

    def _gauss(self, sigma, dist):
        cte = 1/(sigma*np.sqrt(2*np.pi))
        return cte*np.exp(-(dist**2)/(2*sigma**2))

    def fit(self, X, y):
        X = _coordinates_transform(X, self.major_med, self.major_min, self.azimuth, self.dip, self.rake)
        self.X = X
        dist_mat = cdist(X, X)
        int_mat = self._gauss(self.support, dist_mat)
        self.weights = np.dot(np.linalg.inv(int_mat), y) 

    def predict(self, x):
        x = _coordinates_transform(x, self.major_med, self.major_min, self.azimuth, self.dip, self.rake)
        dist_mat = cdist(self.X, x)
        int_mat = self._gauss(self.support, dist_mat)
        return np.dot(int_mat.T, self.weights)


