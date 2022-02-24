# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:01:31 2015

@author: avanetten


"""

from utils import (latlon_to_wmp, wmp_to_latlon)
import graph_utils
import utils

import os
import sys
import time
import random
import numpy as np
import networkx as nx

#############
src_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(src_path)
sys.path.extend([src_path])

###############################################################################
def build_graph(graph_path, bbox_path, 
            min_YOLT_prob=0.05, 
            max_dist_m=5,
            # max_aug_dist=10,
            dist_buff=20):
    '''Build the road + vehicle graph
    # max_dist_m = 5   # max distance vehicle can be from road
    # dist_buff = 80  # radius to look for nodes when finding nearest edge
    # max_aug_dist    # distance to extrapolate control values
    '''


    print("Loading graph pickle:", graph_path, "...")
    # init_page_status_para.text = 'Loading data...'
    G = nx.read_gpickle(graph_path)
    
    # remove self edges?
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    
    # print some graph properties
    print("Num G_gt_init.nodes():", len(G.nodes()))
    print("Num G_gt_init.edges():", len(G.edges()))
    # print random node prop
    node_tmp = random.choice(list(G.nodes()))
    print(("G random node props:", node_tmp, ":", G.nodes[node_tmp]))
    # print random edge properties
    edge_tmp = random.choice(list(G.edges()))
    print("G random edge props:", edge_tmp, ":",
          G.edges[edge_tmp[0], edge_tmp[1], 0])

    # get graph extent
    xmin, ymin, xmax, ymax = graph_utils.get_G_extent(G)
    print("xmin, ymin, xmax, ymax:", xmin, ymin, xmax, ymax)
    
    # create bbox df and source (update G as well...)
    G, df_bbox, g_node_props_dic = graph_utils.load_YOLT(
        bbox_path,
        nearest_edge=True,
        G=G, categories=[], min_prob=min_YOLT_prob,
        scale_alpha_prob=True, max_dist_m=max_dist_m,
        dist_buff=dist_buff,
        verbose=False)
    print("df_bbox.iloc[0]:", df_bbox.iloc[0])

    # ensure node coords contain latlon
    xnode_tmp = [data['x'] for _, data in G.nodes(data=True)]
    ynode_tmp = [data['y'] for _, data in G.nodes(data=True)]
    lats, lons = utils.wmp_to_latlon(xnode_tmp, ynode_tmp)
    for i, (node, data) in enumerate(G.nodes(data=True)):
        data['lat'] = lats[i]
        data['lon'] = lons[i]

    # get kd tree
    kd_idx_dic, kdtree = graph_utils.G_to_kdtree(G)

    print("Data successfully loaded!")
    
    return G, df_bbox, g_node_props_dic, kd_idx_dic, kdtree # , auglist


###############################################################################
def update_G(G):
    """
    Update G weights based on density of cars.
    """
    
    # global G, edge_update_dict
    edge_update_dict = graph_utils.compute_traffic(G, verbose=False)
    G = graph_utils.update_Gweights(G, edge_update_dict,
                                    speed_key1='inferred_speed_mph',
                                    speed_key2='speed_traffic',
                                    edge_id_key='uv',
                                    travel_time_key='Travel Time (h)',
                                    travel_time_key2='Travel Time (h) Traffic',
                                    travel_time_key_default='Travel Time (h) default',
                                    verbose=False)
    return G, edge_update_dict


###############################################################################
def run(graph_path, bbox_path, 
            min_YOLT_prob=0.05, max_dist_m=50,
            dist_buff=80):
    '''
    '''

    # load data
    G, df_bbox, g_node_props_dic, kd_idx_dic, kdtree, auglist = build_graph(graph_path, bbox_path, 
            min_YOLT_prob=0.05, max_dist_m=50,
            dist_buff=80)
    G, edge_update_dict = update_G()

    return G

###############################################################################
if __name__ == "__main__":
    
    bbox_path = sys.argv[1]
    graph_path = sys.argv[2]

    run(graph_path, bbox_path, 
            min_YOLT_prob=0.05, max_dist_m=50,
            dist_buff=80)