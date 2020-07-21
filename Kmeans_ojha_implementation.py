# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:58:19 2020
Implementation of KMeans clustering with 3 clusters
@author: Aditya Ojha
"""
###Necessary Libraries
import matplotlib.pyplot as plt #for data presentation
from scipy.spatial import distance
import numpy as np

###Necessary Functions
#distance.euclidean gets us the distance between two points
#to make the function name shorter we'll use lambda
dist = lambda p1,p2: distance.euclidean(p1,p2)

#function points_average
#inputs: 'points' a list of tuples, each tuple a x,y coordinate
#output: returns a tuple x,y where x = avg(all x points); y = avg(all y points)
###
def points_average(points):
    #we assume that this is a list of tuples
    x_points = [point[0] for point in points]
    y_points = [point[1] for point in points]
    x = sum(x_points)/len(x_points)
    y = sum(y_points)/len(y_points)
    return (x,y)

#function: get distances
#inputs: data is a list of tuples, each tuple=data points. Cluster is tuple
#outputs: list of the distances between each point in data and cluster.
def get_distances(data,cluster) -> list:
    distance_list = []
    for point in data:
        distance_list.append(dist(point,cluster))
    return distance_list
#function get closest point
#inputs: data a list of tuples representing points, cluster is a tuple
#output: the point in data closest in distance to cluster
def get_closest_point(data,cluster) -> tuple:
    return data[np.argmin(get_distances(data, cluster))]

#function get furthest point
#inputs: data a list of tuples representing points, cluster is a tuple
#output: the point(tuple) in data furthest in distance to cluster
def get_furthest_point(data,cluster) -> tuple:
    return data[np.argmax(get_distances(data, cluster))]

###Start of Implementation:
#Define the data to be entered
dataset = [(0,0),(1,1),(2,2),(6,6),(7,7),(15,15),(16,16),(17,17),(18,18)]

clusters = [None]*3 #list of the three clusters
clusters[0] = dataset[0] #first cluster is the first point
clusters[1] = get_furthest_point(dataset[0:],clusters[0]) #2nd cluster is point furthest from 1st cluster
clusters[2] = get_furthest_point(dataset[1:], clusters[1])#etc(^)
old_groups = []#values set so that they purposely don't equal each other
new_groups = [1]
while(old_groups != new_groups):
    old_groups = new_groups.copy() #use a copy because we don't want both vars to reference same data
    classifi_dict = dict()
    for cluster in clusters:
        classifi_dict[cluster] = []
    #now assign point to a cluster
    for point in dataset:
        cluster_for_this_point = get_closest_point(list(classifi_dict.keys()),point) #get the cluster that is closest to this point
        classifi_dict[cluster_for_this_point].append(point)#append the point to that cluster's values in the dict
    #keep track of which cluster got which points
    new_groups = list(classifi_dict.values())
    #points_groups is a list of lists; assume indexs x,y
    #x = classification (0,1,2,etc)
    #y = list of points in the classification specified by x
    
    ###Redefine the clusters based on the points in each group
    for index,points_in_group in enumerate(new_groups):
        clusters[index] = points_average(points_in_group)
print(clusters)
for value in new_groups:
    x_points = [point[0] for point in value]
    y_points = [point[1] for point in value]
    plt.scatter(x_points,y_points)