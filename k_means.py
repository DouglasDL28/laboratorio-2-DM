# Universidad del Valle de Guatemala
# Data Mining
# Douglas de León Molina - 18037

from copy import deepcopy
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame

def distances(x, centroids):
    cluster = 0
    min_dist = np.inf
    for i in range(len(centroids)):
        cols = centroids[i].columns # columns w/o clusters
        c_vals = centroids[i][cols].values[0] # centroid values
        dist = np.linalg.norm(x[cols] - c_vals) # euclidean distance
        if dist < min_dist:
            min_dist = dist
            cluster = i
    
    return cluster
        

class KMeans():
    def __init__(self, k:int, df: DataFrame):
        self.k = k
        self.df = df
        self.norm_df = self.normalize()

    def normalize(self):
        return (self.df-self.df.mean())/self.df.std()
        
    def fit(self):
        # initialize centroids with random samples from df (?)
        centroids = [self.norm_df.sample(1) for i in range(self.k)]

        max_iters = 100
        n_inters = 0
        new_assignments = True
        while(new_assignments and (n_inters < max_iters)):
            # assign clusters based on eucledian distance
            n_inters += 1
            new_centroids = deepcopy(centroids)
            self.norm_df["cluster"] = self.norm_df.apply(lambda row: distances(row, centroids), axis=1)

            # align centroids to mean values
            for c in range(len(new_centroids)):
                c_cols = list(new_centroids[c].columns)

                new_centroids[c][c_cols] = self.norm_df[self.norm_df['cluster'] == c].mean()[c_cols]
            
            # check if there were new assignments
            for c in range(len(centroids)):
                new_assignments = new_assignments and not new_centroids[c].equals(centroids[c])
            
            centroids = deepcopy(new_centroids)

            print("ITERATION N=", n_inters)
            print("Centroids:\n", new_centroids)

        # copy cluster assignments to original df
        self.df['cluster'] = self.norm_df['cluster']

        return centroids, self.df
