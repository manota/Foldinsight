import numpy as np
from numba import jit,prange
import pandas as pd
import itertools
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial import Delaunay



@jit(parallel=True)
def get_inter_equinox(coord_and_value, v,base_distance):
    n_sample=coord_and_value.shape[0]
    inter_equinox_array=np.zeros((n_sample*n_sample,3))
    inter_equinox_array[:,:]=np.nan
    for i,i_coord in enumerate(coord_and_value):
        for j in prange(coord_and_value.shape[0]):
            j_coord= coord_and_value[j, :]
            distance = np.sqrt(np.sum((i_coord[:3] - j_coord[:3]) ** 2))
            if distance == 0 or distance >base_distance:
                continue
            inter_equinox = (v - i_coord[3]) / (j_coord[3] - i_coord[3])
            if inter_equinox < 0 or inter_equinox > 1:
                continue
            inter_equinox_array[n_sample*i+j,:]=inter_equinox* (j_coord[:3] - i_coord[:3])+i_coord[:3]
    #ans_array=ans_array[~np.isnan(ans_array).all(axis=1),:]
    return inter_equinox_array

def rapper_inter_equinox(coord_and_value,v,base_distance=4):
    inter_equinox_array=get_inter_equinox(coord_and_value,v,base_distance)
    inter_equinox_array=inter_equinox_array[~np.isnan(inter_equinox_array).all(axis=1),:]
    return inter_equinox_array

def make_bild_file(coord,file_path,color='green'):
    hull = ConvexHull(coord)
    bild_file=[]
    for simplex in hull.simplices:
        coord_0 = np.round(coord[simplex[0], :], decimals=2)
        coord_1 = np.round(coord[simplex[1], :], decimals=2)
        coord_2 = np.round(coord[simplex[2], :], decimals=2)
        strings = '.color '+ color+'\n'+'.polygon ' + " ".join(map(str, coord_0)) + " " + " ".join(map(str, coord_1)) + " " + " ".join(map(str, coord_2)) + '\n'
        bild_file.append(strings)
    with open(file_path, mode='w') as f:
        f.writelines(bild_file)


def make_bild_file_all(coord,file_path,color='green',base_distance=10):
    hull = ConvexHull(coord)
    bild_file=[]
    for simplex in hull.simplices:
        coord_0 = np.round(coord[simplex[0], :], decimals=2)
        coord_1 = np.round(coord[simplex[1], :], decimals=2)
        coord_2 = np.round(coord[simplex[2], :], decimals=2)

        coord_set=[coord_0,coord_1,coord_2]
        distance_set = []
        for i, j in itertools.combinations(coord_set, 2):
            distance = np.sqrt(np.sum((i - j) ** 2))
            distance_set.append(distance)
        ans_binary=[i>base_distance for i in distance_set]
        if any(ans_binary)>0:
            continue
        strings = '.color '+ color+'\n'+'.polygon ' + " ".join(map(str, coord_0)) + " " + " ".join(map(str, coord_1)) + " " + " ".join(map(str, coord_2)) + '\n'
        bild_file.append(strings)
    with open(file_path, mode='w') as f:
        f.writelines(bild_file)



def get_tree_cluster(coord,base_distance=6):
    distance_tree = []
    host_inter_equinox_coord = [i for i in range(coord.shape[0])]
    for i in host_inter_equinox_coord:
        coord_i = coord[i, :]
        cluster = []
        cluster.append(i)
        temp = [j for j in host_inter_equinox_coord if
                np.linalg.norm(coord_i - coord[j, :]) < base_distance and np.linalg.norm(
                    coord_i - coord[j, :]) > 0]
        temp = np.setdiff1d(temp, cluster).tolist()
        for l in temp:
            cluster.append(l)

        for j in cluster:
            coord_j = coord[j, :]
            temp = [k for k in host_inter_equinox_coord if
                    np.linalg.norm(coord_j - coord[k, :]) < base_distance and np.linalg.norm(
                        coord_j - coord[k, :]) > 0]
            temp = np.setdiff1d(temp, cluster).tolist()
            for l in temp:
                cluster.append(l)
        print(cluster)
        distance_tree.append(cluster)
        for l in cluster:
            try:
                pos = host_inter_equinox_coord.index(l)
            except:
                continue
            host_inter_equinox_coord.pop(pos)
        print(len(cluster))
    return distance_tree

def get_delaunay_pair(coord):
    points = coord[:, :3]
    tri = Delaunay(points)
    delaunay_pair = []
    for vertexes in tri.simplices:
        for i, j in itertools.combinations(vertexes, 2):
            delaunay_pair.append(np.sort([i, j]).tolist())

    delaunay_pair = np.array(delaunay_pair)
    delaunay_pair = np.unique(delaunay_pair, axis=0)
    return delaunay_pair


def get_inter_equinox_based_delauny(coord_and_value, v,base_distance=100):
    delaunay_pair=get_delaunay_pair(coord_and_value[:,:3])
    inter_equinox_array=np.zeros((delaunay_pair.shape[0],3))
    inter_equinox_array[:,:]=np.nan
    for i,vertexes in enumerate(delaunay_pair):
        vert_0 = coord_and_value[vertexes[0], :]
        vert_1 = coord_and_value[vertexes[1], :]
        distance=np.sqrt(np.sum((vert_1[:3]-vert_0[:3])**2))
        if distance==0 or distance>base_distance:
            continue
        inter_equinox = (v - vert_0[3]) / (vert_1[3] - vert_0[3])
        if inter_equinox < 0 or inter_equinox > 1:
            continue
        inter_equinox_points = inter_equinox * (vert_1[:3] - vert_0[:3]) + vert_0[:3]
        inter_equinox_array[i,:]=inter_equinox_points
    inter_equinox_array = inter_equinox_array[~np.isnan(inter_equinox_array).all(axis=1), :]
    return inter_equinox_array





# a=np.arange(5)
# coord=[pd.DataFrame(np.array([i,j,k])) for i,j,k in itertools.product(a,a,a)]
# coord=pd.concat(coord,axis=1).values
# coord=coord.reshape(-1,3)
# random_v=np.random.random((coord.shape[0],1))
#
# coord_and_value=np.concatenate([coord,random_v],axis=1)
#
#
#
#
# v=0.5
# xx=get_inter_equinox_based_delauny(coord_and_value,v)
#
# print(xx)
# print('sss')