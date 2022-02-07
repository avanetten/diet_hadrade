#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 23:17:26 2018

@author: Adam Van Etten
"""

from shapely.geometry import Point, LineString
import scipy.spatial
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import scipy.spatial
import time
import sys
import os
import copy
import importlib


import utils

###############################################################################
def define_colors_alphas():
    '''
    Define colors for plotting
    https://www.w3schools.com/colors/colors_names.asp
    '''
    # colors
    color_dic = \
    {
     
        # categories of objects
        'Bus':                      'blue',
        'Truck':                    'firebrick',
        'Small_Vehicle':            'mediumseagreen',
        'building':                 'blue',
        
        # computed colors
        'goodnode_color':           'blue',
        'goodnode_aug_color':       'dodgerblue',#'cornflowerblue',
        'badnode_color':            'firebrick',#'crimson',
        'badnode_aug_color':        'red',
        'diff_node_color':          'purple',
        'diff_node_color_sec':      'lightgreen',
        'missing_node_color':       'chartreuse',#'fuchsia',
        'compute_path_color_good':  'aqua', #'mediumseagreen'
        'compute_path_color_bad':   'magenta',#'darkmagenta',#'aqua',#'mediumseagreen'
        'compute_path_color':       'lawngreen',#'aqua',#'chartreuse',#'darkmagenta',#'aqua',#'mediumseagreen',#'mediumvioletred',
        'crit_node_color':          'orange',#'darkorchid','mediumseagreen',
        'source_color':             'mediumvioletred',
        'target_color':             'green',
        'spread_seen_color':        'maroon',
        'spread_new_color':         'mediumvioletred',
        'histo_bin_color':          'teal',#'purple',
        'histo_cum_color':          'slateblue',#'orange'
        'node_centrality_color':    'lime',#'teal'#'lightgreen'
        'overlap_node_color':       'darkorchid',
        'risk_color':               'springgreen',

        # osm edges
        'motorway':                 'darkred',
        'trunk':                    'tomato',
        'primary':                  'orange',
        'secondary':                'gold',
        'tertiary':                 'yellow',
        'bridge':                   'pink',

        # speed colors
        0:                          'lightyellow',
        5:                          'lightyellow',
        10:                         'lightyellow',
        15:                         'yellow',
        20:                         'yellow',
        25:                         'gold',
        30:                         'gold',
        35:                         'orange',
        40:                         'orange',
        45:                         'tomato',
        50:                         'tomato',
        55:                         'firebrick',
        60:                         'firebrick',
        65:                         'darkred',
        70:                         'darkred',

        # traffic colors (1 means no traffic, 0 means all the traffic)
        # https://www.rapidtables.com/web/color/purple-color.html
        '0.0':                      'darkslategray',
        '0.2':                      'indigo',
        '0.4':                      'purple',
        '0.6':                      'blueviolet',
        '0.8':                      'mediumorchid',
        '1.0':                      'lavender',

        # raw color
        'raw_edge':                 'darkgray',

        # osm nodes
        'intersection':             'gray',
        'endpoint':                 'gray',
        'midpoint':                 'gray',  # 'black'
        'start':                    'green',
        'end':                      'red'
    }

    # opacity
    alpha_dic = {
        'osm_edge':                 0.4,
        'osm_node':                 0.15,
        'gdelt':                    0.6,
        'aug':                      0.35,
        'end_node':                 0.6,
        'crit_node':                0.7,
        'missing_node':             0.6,
        'target':                   0.7,
        'compute_paths':            0.6,
        'compute_paths_sec':        0.3,
        'centrality':               0.5,
        'histo_bin':                0.5,
        'histo_cum':                0.6,
        'label_slight':             0.5,
        'label_general':            0.7,
        'label_bold':               0.9,
        'hull':                     0.025,  # #4.5, #0.025,
        'diff':                     0.5,
        'risk':                     0.65,
        'contours':                 0.65,
        'force_proj':               0.2
    }

    return color_dic, alpha_dic



###############################################################################
### Data ingest
###############################################################################
###############################################################################
def load_YOLT(test_predictions_gdf, G=None, categories=[],
                 min_prob=0.05, scale_alpha_prob=True, max_dist_m=10000,
                 dist_buff=5000, nearest_edge=True, search_radius_mult=5,
                 max_rows=10000, randomize_goodness=False,
                 verbose=False, super_verbose=False):
    '''Load data from output of YOLT
    Columns:
        Loc_Tmp	Prob	Xmin	Ymin	Xmax	Ymax	Category	
        Image_Root_Plus_XY	Image_Root	Slice_XY	Upper	Left
        Height	Width	Pad	Im_Width	Im_Height	Image_Path
        Xmin_Glob	Xmax_Glob	Ymin_Glob	Ymax_Glob
        Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp
    # buses are good (+1) trucks are bad (-1), cars are neutral (0)
    '''

    t0 = time.time()
    print("Executing load_YOLT()...")
    print("  max_dist_m:", max_dist_m)
    print("Test predictions gdf:", test_predictions_gdf)

    # colors for plotting
    color_dic, alpha_dic = define_colors_alphas()

    # read gdf
    df_raw = gpd.read_file(test_predictions_gdf)
    
    # rename cols
    df_raw = df_raw.rename(columns={"prob": "Prob", "category": "Category"})
    
    # filter length
    df_raw = df_raw[:max_rows]
    print("Len raw predictions:", len(df_raw))

    # make sure index starts at 1, not 0
    df_raw.index = np.arange(1, len(df_raw) + 1)
    print("Unique Categories:", np.unique(df_raw['Category'].values))

    # filter out probabilities
    df = df_raw[df_raw['Prob'] >= min_prob]

    # filter out categories
    if len(categories) > 0:
        df = df.loc[df['Category'].isin(categories)]
    
    # get bounds of geoms
    xmin_list, xmax_list, ymin_list, ymax_list = [], [], [], []
    for geom_tmp in df['geometry']:
        minx, miny, maxx, maxy = geom_tmp.bounds
        xmin_list.append(minx)
        xmax_list.append(maxx)
        ymin_list.append(miny)
        ymax_list.append(maxy)
    df['Xmin_wmp'] = xmin_list
    df['Xmax_wmp'] = xmax_list
    df['Ymin_wmp'] = ymin_list
    df['Ymax_wmp'] = ymax_list

    print("Len predictions csv:", len(df))

    # # rename wmp columns to what run_gui.py expects
    # df = df.rename(index=int, columns={
    #     'x0_wmp': 'Xmin_wmp',
    #     'x1_wmp': 'Xmax_wmp',
    #     'y0_wmp': 'Ymin_wmp',
    #     'y1_wmp': 'Ymax_wmp',
    # })

    df['xmid'] = 0.5*(df['Xmin_wmp'].values + df['Xmax_wmp'].values)
    df['ymid'] = 0.5*(df['Ymin_wmp'].values + df['Ymax_wmp'].values)

    # each box has a count (and num) of 1
    df['count'] = np.ones(len(df))
    df['num'] = np.ones(len(df))

    # set lat lons
    lats, lons = utils.wmp_to_latlon(df['xmid'].values, df['ymid'].values)
    df['lat'] = lats
    df['lon'] = lons

    # determine colors
    colors = [color_dic[cat] for cat in df['Category'].values]
    # V0
    # colors = [color_dic['goodnode_color'] if v > 0 else color_dic['badnode_color'] for v in df['Val'].values]
    df.insert(len(df.columns), "color", colors)
    # df['color'] = colors

    # set Val
    # buses are good (+1) trucks are bad (-1), cars are neutral (0)
    # vals = compute_goodness(df, randomize=randomize_goodness)
    df['Val'] = 0 # vals

    # vals = np.zeros(len(df))
    # pos_idxs = np.where(df['Category'].values == 'Bus')
    # neg_idxs = np.where(df['Category'].values == 'Truck')
    # vals[pos_idxs] = 1
    # vals[neg_idxs] = -1
    # df['Val'] = vals
    ## V0 (random)
    ## asssign a random value of 0 or 1
    ## df_raw['Val'] = np.random.randint(0,2,size=len(df_raw))

    # get kdtree
    # if G is not None:
    #    kd_idx_dic, kdtree = G_to_kdtree(G)
    #    # get graph bounds
    #    xmin0, ymin0, xmax0, ymax0 = get_G_extent(G)
    #    xmin, ymin = xmin0 - dist_buff, ymin0 - dist_buff
    #    xmax, ymax = xmax0 + dist_buff, ymax0 + dist_buff
    #    print ("xmin, xmax, ymin, ymax:", xmin, xmax, ymin, ymax)

    # set alpha
    if scale_alpha_prob:
        # scale between 0.3 and 0.8
        prob_rescale = df['Prob'] / np.max(df['Prob'].values)
        alphas = 0.3 + 0.5 * prob_rescale
    else:
        alphas = 0.75
    # set alpha column
    df.insert(len(df.columns), "line_alpha", alphas)
    df.insert(len(df.columns), "fill_alpha", alphas-0.1)
    #df['line_alpha'] = alphas
    #df['fill_alpha'] = alphas - 0.15

    # create dictionary of gdelt node properties
    df.insert(len(df.columns), "nearest_osm", '')
    df.insert(len(df.columns), "dist", 0.0)
    df.insert(len(df.columns), "status", [
              'good' if v > 0 else 'bad' for v in df['Val']])
    #df['nearest_osm'] = ''
    #df['dist'] = 0.0
    #df['status'] = ['good' if v > 0 else 'bad' for v in df['Val']]

    # create sets of nodes
    s0 = set([])
    idx_rem = []
    g_node_props_dic = {}
    xmids, ymids, dists, nearests = [], [], [], []

    # iterate through rows to determine nearest edge to each box
    # need to speed this up!!!
    # https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe
    # https://github.com/CosmiQ/solaris/blob/master/solaris/utils/geo.py
    if nearest_edge:
        print("Inserting new nodes for each YOLT data point...")
        X = df['xmid'].values
        Y = df['ymid'].values

        # get graph bounds
        xmin0, ymin0, xmax0, ymax0 = get_G_extent(G)
        xmin, ymin = xmin0 - dist_buff, ymin0 - dist_buff
        xmax, ymax = xmax0 + dist_buff, ymax0 + dist_buff
        print("xmin, xmax, ymin, ymax:", xmin, xmax, ymin, ymax)

        # construct a dictionary to track altered edges
        #dict_edge_altered = dict.fromkeys(list(G_.edges()), [])
        dict_edge_altered = {}
        for e in G.edges():
            dict_edge_altered[e] = []
        #print (dict_edge_altered)

        # get edges
        # time all at once
        tt0 = time.time()
        spacing_dist = 5  # spacing distance (meters)
        nearest_edges = ox.get_nearest_edges(
            G, X, Y, method='kdtree', dist=spacing_dist)
        print("Time to get {} nearest edges = {}".format(
            len(nearest_edges), time.time()-tt0))
        # print("nearest_edges:", nearest_edges)
        # print("df.index:", df.index)
        tt1 = time.time()
        for i, (index, row) in enumerate(df.iterrows()):

            # if index > 23:
            #    return

            insert_node_id = int(-1 * index)
            #insert_node_id = -1 + int(-1 * index)
            if (i % 500 ) == 0:
                print(i, "index:", index, "/", len(df))
            #print ("  row:", row)
            #cat, prob = row['Category'], row['Prob']
            Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp = \
                row['Xmin_wmp'], row['Xmax_wmp'], row['Ymin_wmp'], row['Ymax_wmp']
            #row['x0_wmp'], row['x1_wmp'], row['y0_wmp'], row['y1_wmp']
            xmid, ymid = (Xmin_wmp + Xmax_wmp)/2., (Ymin_wmp + Ymax_wmp)/2.
            #print ("Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp:", Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp)
            #print (" (Xmin_wmp < xmin):",  (Xmin_wmp < xmin))
            #print (" (Ymin_wmp < ymin):",  (Ymin_wmp < ymin))
            #print (" (Xmax_wmp > xmax):",  (Xmax_wmp > xmax))
            #print (" (Ymax_wmp > ymax):",  (Ymax_wmp > ymax))

            # skip if too far from max coords of G.nodes (much faster than full
            # distance calculation)
            if (Xmin_wmp < xmin) or (Ymin_wmp < ymin) \
                    or (Xmax_wmp > xmax) or (Ymax_wmp > ymax):
                idx_rem.append(index)
                print("Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp:",
                      Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp)
                print(" (Xmin_wmp < xmin):",  (Xmin_wmp < xmin))
                print(" (Ymin_wmp < ymin):",  (Ymin_wmp < ymin))
                print(" (Xmax_wmp > xmax):",  (Xmax_wmp > xmax))
                print(" (Ymax_wmp > ymax):",  (Ymax_wmp > ymax))
                return
                continue

            best_edge = nearest_edges[i]
            if verbose:
                print("best_edge:", best_edge)
            # we 1-indexed the df
            # best_edge = nearest_edges[index-1]
            [u, v] = best_edge[:2]
            if verbose:
                print("u,v:", u, v)
            point = (xmid, ymid)

            # Get best edge
            # check if edge even exists
            #   if so, insert into the edge
            if G.has_edge(u, v):
                best_edge = (u, v)
            elif G.has_edge(v, u):
                best_edge = (v, u)
            else:
                # we'll have to use the newly added edges that won't show up in
                #  ne (the list of closest edges)
                if verbose:
                    print("edge {} DNE!".format((u, v)))
                edges_new = dict_edge_altered[(u, v)]
                if verbose:
                    print("edges_new:", edges_new)
                # get nearest edge from short list
                best_edge, min_dist, best_geom = \
                    get_closest_edge_from_list(G, edges_new,
                                                    Point(xmid, ymid),
                                                    verbose=verbose)

            # insert point
            if verbose:
                print("best edge:", best_edge)
            G, node_props, min_dist, edge_list, edge_props, rem_edge \
                = insert_point_into_edge(G, best_edge, point,
                                              node_id=int(insert_node_id),
                                              max_distance_meters=max_dist_m,
                                              allow_renaming_once=False,
                                              verbose=verbose,
                                              super_verbose=super_verbose)          
            # assign newly inserted node pix coords as the mid of the bbox coords
            if int(insert_node_id) in list(G.nodes()):
                G.nodes[int(insert_node_id)]['x_pix'] = int(np.mean([row['x0_pix'], row['x1_pix']]))
                G.nodes[int(insert_node_id)]['y_pix'] = int(np.mean([row['y0_pix'], row['y1_pix']]))

            if verbose:
                print("min_dist:", min_dist)
            # if an edge has been updated, load new props into dict_edge_altered
            if len(edge_list) > 0:
                z = dict_edge_altered[(u, v)]
                # remove rem_edge item from dict
                if len(rem_edge) > 0 and rem_edge in z:
                    z.remove(rem_edge)
                #print ("z:", z)
                # z.append(edge_list)  # can't append or all values get updated!!!
                val_tmp = z + edge_list
                # update dict value
                dict_edge_altered[(u, v)] = val_tmp
                if super_verbose:
                    print("\n", u, v)
                    print("edge_list:", edge_list)
                    print("dict_edge_altered[(u,v)] :",
                          dict_edge_altered[(u, v)])
                    #print ("  dict", dict_edge_altered)

            # maybe if distance is too large still plot it, but don't include
            #   in analytics
            if min_dist > max_dist_m:
                if verbose:
                    print("dist > max_dist_m")
                node_name = 'Null'
                xmids.append(xmid)
                ymids.append(ymid)
                dists.append(min_dist)
                node_name_tmp = 'null'
                nearests.append(node_name_tmp)
                continue

            else:
                if verbose:
                    print("Updating df values")
                    print("node_props:", node_props)
                node_name = insert_node_id  # node_props['osmid']
                xmids.append(xmid)
                ymids.append(ymid)
                dists.append(min_dist)
                nearests.append(node_name)

                # update node properties if nearest node in s0, else create new
                # If node_name is in s0, there could be multiple reports for the same
                # location.  For now, just keep the report with the largest count
                if node_name not in s0:
                    g_node_props_dic[node_name] = \
                        {'index': [index], 'dist': [min_dist]}
                    s0.add(node_name)
                    # if node_name < 0:
                    #    insert_node_id += -1

                else:
                    if verbose:
                        print("node name", node_name, "already in s0:")
                    g_node_props_dic[node_name]['index'].append(index)
                    g_node_props_dic[node_name]['dist'].append(min_dist)

        # add travel time
        G = add_travel_time(G)

    # iterate through rows to determine nearest node to each box
    else:
        print("Assigning existing intersection/endpoint for each YOLT data point...")
        for index, row in df.iterrows():
            print("index:", index, "/", len(df))
            #cat, prob = row['Category'], row['Prob']
            Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp = \
                row['Xmin_wmp'], row['Xmax_wmp'], row['Ymin_wmp'], row['Ymax_wmp']
            xmid, ymid = (Xmin_wmp + Xmax_wmp)/2., (Ymin_wmp + Ymax_wmp)/2.
            #print ("Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp:", Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp)
            #print (" (Xmin_wmp < xmin):",  (Xmin_wmp < xmin))
            #print (" (Ymin_wmp < ymin):",  (Ymin_wmp < ymin))
            #print (" (Xmax_wmp > xmax):",  (Xmax_wmp > xmax))
            #print (" (Ymax_wmp > ymax):",  (Ymax_wmp > ymax))

            # skip if too far from max coords of G.nodes (much faster than full
            # distance calculation)
            if (Xmin_wmp < xmin) or (Ymin_wmp < ymin) \
                    or (Xmax_wmp > xmax) or (Ymax_wmp > ymax):
                idx_rem.append(index)
                print("Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp:",
                      Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp)
                print(" (Xmin_wmp < xmin):",  (Xmin_wmp < xmin))
                print(" (Ymin_wmp < ymin):",  (Ymin_wmp < ymin))
                print(" (Xmax_wmp > xmax):",  (Xmax_wmp > xmax))
                print(" (Ymax_wmp > ymax):",  (Ymax_wmp > ymax))
                return
                continue

            # find nearest osm node, update dataframe values
            node_name, dist, kd_idx = utils.query_kd(
                kdtree, kd_idx_dic, xmid, ymid)
            #print ("node_name, dist, kd_idx:", node_name, dist, kd_idx)

            # remove if distance is too large, continue loop
            if dist > max_dist_m:
                idx_rem.append(index)
                print("dist > max_dist_m")
                print("  dist:", dist)
                # return
                continue

            xmids.append(xmid)
            ymids.append(ymid)
            dists.append(dist)
            nearests.append(node_name)
            #df.loc[index, 'Xmid_wmp'] = xmid
            #df.loc[index, 'Ymid_wmp'] = ymid
            #df.loc[index, 'dist'] = dist
            #df.loc[index, 'nearest_node'] = node_name
            ##df.set_value(index, 'Xmid_wmp', xmid)
            ##df.set_value(index, 'Ymid_wmp', ymid)
            ##df.set_value(index, 'dist', dist)
            ##df.set_value(index, 'nearest_osm', node_name)

            # update osm node properties if nearest node in s0, else create new
            # If node_name is in s0, there could be multiple reports for the same
            # location.  For now, just keep the report with the largest count
            if node_name not in s0:
                g_node_props_dic[node_name] = \
                    {'index': [index], 'dist': [dist]}
                s0.add(node_name)

            else:
                g_node_props_dic[node_name]['index'].append(index)
                g_node_props_dic[node_name]['dist'].append(dist)

            #colortmp = color_dic['badnode_color']
            #df.set_value(index, 'color', colortmp)

    # remove unnneeded indexes of df
    # print("df.index:", df.index)
    print("idx_rem", idx_rem)
    print("len of original df", len(df))
    if len(idx_rem) > 0:
        df = df.drop(np.unique(idx_rem))
        #df = df.drop(df.index[np.unique(idx_rem)])
    print("len of refined df", len(df))

    df['index'] = df.index.values
    df['Xmid_wmp'] = xmids
    df['Ymid_wmp'] = ymids
    df['dist'] = dists
    df['nearest_node'] = nearests

    print("Time to load YOLT data:", time.time() - t0, "seconds")
    # return G, df, source_YOLT, g_node_props_dic
    return G, df, g_node_props_dic


###############################################################################
def density_speed_conversion(N, frac_per_car=0.025, min_val=0.2):
    """
    Fraction to multiply speed by if there are N nearby vehicles
    """
    
    z = 1.0 - (frac_per_car * N)
    # z = 1.0 - 0.04 * N
    return max(z, min_val)
    

###############################################################################
def compute_traffic(G, query_radius_m=50, min_nearby=2, max_edge_len=50,
                    verbose=False):
    """
    Compute updated edge speeds based on density of cars.
    Calls density_speed_conversion().

    Notes
    -----
    Assume nodes near cars have been added, with a negative index

    Arguments
    ---------
    G : networkx graph
        Input graph
    query_radiums_m : float
        Radius to query around each negative node for other negative nodes.
        Defaults to ``200`` (meters).
    min_query : int
        Number of nearby cars required to consider trafficy.
        Defaults to ``2``.
    max_edge_len : float
        Maximum edge length to consider reweighting.
        Defaults to ``50`` (meters).
    verbose : bool
        Switch to print relevant values

    Returns
    -------
    edge_update_dict : dict
        Dictionary with fraction of original speed for appropriate edges.
        Example entry:
             (-999, 258910932): 0.4}
             # (-999, 258910932): {'orig_mps': 11.176, 'new_mps': 3.3528}}
    """

    print ("Computing traffic...")
    # get all nodes less than zero (corresponding to a vehicle)
    neg_nodes = [n for n in list(G.nodes()) if n < 0]
    if verbose:
        print("compute_traffic(): neg_nodes:", neg_nodes)

    kd_idx_dic, kdtree = G_to_kdtree(G, node_subset=set(neg_nodes))
    if verbose:
        print("compute_traffic(): kd_idx_dic:", kd_idx_dic)

    # iterate throught each neg node
    edge_update_dict = {}
    edge_altered_set = set([])
    # can't search only negative nodes, because it's possible that normal
    # nodes will have tons of nearby cars too
    # for i, n in enumerate(neg_nodes):
    for i, n in enumerate(list(G.nodes())):
        x, y = G.nodes[n]['x'], G.nodes[n]['y']
        # get number of vehicles near each node
        node_names, idxs_refine, dists_m_refine = utils.query_kd_ball(
            kdtree, kd_idx_dic, x, y, query_radius_m, verbose=False)
        n_nearby = len(node_names)

        if n_nearby > min_nearby:
            if verbose:
                print("i, n:", i, n)
                print("node_names:", node_names)
                print("dists:", dists_m_refine)

            # get edges coincident on nodes
            coinc_e = G.edges(n)
            if verbose:
                print("coinc_e:", coinc_e)
            # check coincident edges
            for (u, v) in coinc_e:
                if verbose:
                    print("u, v,", u, v)
                if (u, v) not in edge_altered_set:
                    # data = G.get_edge_data(u, v)
                    # print("data:", data)
                    line_len = G.edges[u, v, 0]['length']
                    if line_len <= max_edge_len:
                        speed_frac = density_speed_conversion(n_nearby)
                        edge_update_dict[(u, v)] = speed_frac
                        if verbose:
                            print("line_len", line_len)
                            print("speed_frac:", speed_frac)
                        # speed_mps = G.edges[u, v, 0]['speed_m/s']
                        # new_speed = speed_mps * speed_frac
                        # if verbose:
                        #    print("line_len", line_len)
                        #    print("speed_mps:", speed_mps)
                        #    print("new_speed:", new_speed)
                        # edge_update_dict[(u, v)] = {'orig_mps': speed_mps,
                        #                             'new_mps': new_speed}
                        edge_altered_set.add((u, v))

    if verbose:
        print("edge_altered_set:", edge_altered_set)
        print("edge_update_dict:", edge_update_dict)

    return edge_update_dict


###############################################################################
def update_Gweights(G, update_dict,
                    speed_key1='inferred_speed_mph',
                    speed_key2='speed_traffic',
                    edge_id_key='uv',
                    congestion_key='congestion',
                    travel_time_key='Travel Time (h)',
                    travel_time_key2='Travel Time (h) Traffic',
                    travel_time_key_default='Travel Time (h) default',
                    verbose=False):
    """Update G and esource"""

    # color_dic, alpha_dic = define_colors_alphas()
    update_keys = set(list(update_dict.keys()))

    # update speed
    for j, (u, v, data) in enumerate(G.edges(data=True)):
        speed1 = data[speed_key1]
        if (u, v) in update_keys:
            frac = update_dict[(u, v)]
        elif (v, u) in update_keys:
            frac = update_dict[(v, u)]
            # if verbose:
            #    print("u, v, frac:", u, v, frac)
        else:
            frac = 1
        speed_traffic = speed1 * frac
        data[speed_key2] = speed_traffic
        # set congestion
        data[congestion_key] = frac
    # update travel time
    G = add_travel_time(G, speed_key=speed_key1,
                        travel_time_key=travel_time_key,
                        speed_key2=speed_key2,
                        travel_time_key2=travel_time_key2,
                        travel_time_key_default=travel_time_key_default)

    return G


###############################################################################
def add_travel_time(G_,
                    length_key='length',  # meters
                    speed_key='inferred_speed_mph',
                    travel_time_key='Travel Time (h)',
                    travel_time_s_key='travel_time_s',
                    speed_key2='speed_traffic',
                    travel_time_key2='Travel Time (h) Traffic',
                    default_speed=31.404,   # mph
                    travel_time_key_default='Travel Time (h) default',
                    verbose=False):
    '''Add travel time estimate to each edge
    if speed_key does not exist, use default
    Default speed is 31.404 mph'''

    for i, (u, v, data) in enumerate(G_.edges(data=True)):
        if speed_key in data:
            speed_mph = data[speed_key]
        else:
            data['inferred_speed'] = default_speed
            data[speed_key] = default_speed
            speed_mph = default_speed

        if verbose:
            print("data[length_key]:", data[length_key])
            print("speed:", speed_mph)

        speed_mps = 0.44704 * speed_mph
        travel_time_s = data['length'] / speed_mps
        travel_time_h = travel_time_s / 3600

        data[travel_time_s_key] = travel_time_s
        data[travel_time_key] = travel_time_h

        # get weights for speed_traffic
        if speed_key2 in data:
            speed_mph2 = data[speed_key2]
        else:
            speed_mph2 = speed_mph
        if verbose:
            print("speed_traffic:", speed_mph2)
        speed_mps2 = 0.44704 * speed_mph2
        travel_time_s2 = data['length'] / speed_mps2
        travel_time_h2 = travel_time_s2 / 3600
        data[travel_time_key2] = travel_time_h2

        # get weights for default
        speed_mps3 = 0.44704 * default_speed
        travel_time_s3 = data['length'] / speed_mps3
        travel_time_h3 = travel_time_s3 / 3600
        data[travel_time_key_default] = travel_time_h3

        # print(data)
        # print ("travel_time_h, travel_time_h2, travel_time_h3:",
        #     travel_time_h, travel_time_h2, travel_time_h3)

    return G_


###############################################################################
def choose_target(G, skiplists, direction='north'):
    '''Choose target node in G.nodes() thats not in skiplists
    pick the northernmost (or other direction) of all nodes
    direction options are 'north, south, east, west'
    '''

    # keep only nodes not in skiplists
    skipset = set([item for sublist in skiplists for item in sublist])
    Gset = set(G.nodes())
    nodes = np.asarray(list(Gset-skipset))
    # extract locations,
    xs, ys = [], []
    for n in nodes:
        xs.append(G.nodes[n]['x'])
        ys.append(G.nodes[n]['y'])

    # order by x
    fx = np.argsort(xs)
    nodes_xsort = nodes[fx]

    # order by lon
    fy = np.argsort(ys)
    nodes_ysort = nodes[fy]

    if direction.lower() == 'north':
        node = nodes_ysort[-1]
    elif direction.lower() == 'south':
        node = nodes_ysort[0]
    elif direction.lower() == 'east':
        node = nodes_xsort[-1]
    elif direction.lower() == 'west':
        node = nodes_xsort[0]
    else:
        node = None

    return node


###############################################################################
def choose_target_latlon(G, skiplists, direction='north'):
    '''Choose target node in G.nodes() thats not in skiplists
    pick the northernmost (or other direction) of all nodes
    direction options are 'north, south, east, west'
    '''

    # keep only nodes not in skiplists
    skipset = set([item for sublist in skiplists for item in sublist])
    Gset = set(G.nodes())
    nodes = np.asarray(list(Gset-skipset))
    # extract locations,
    lats, lons = [], []
    for n in nodes:
        lons.append(G.nodes[n]['lon'])
        lats.append(G.nodes[n]['lat'])

    # order by lat
    flat = np.argsort(lats)
    nodes_latsort = nodes[flat]

    # order by lon
    flon = np.argsort(lons)
    nodes_lonsort = nodes[flon]

    if direction.lower() == 'north':
        node = nodes_latsort[-1]
    elif direction.lower() == 'south':
        node = nodes_latsort[0]
    elif direction.lower() == 'east':
        node = nodes_lonsort[-1]
    elif direction.lower() == 'west':
        node = nodes_lonsort[0]
    else:
        node = None

    return node


###############################################################################
def G_to_csv(G, outfile, delim='|'):
    '''
    Create csv export.
    Assume encoding is unicode: utf-8
    '''

    print("Printing networkx graph to csv...")
    t0 = time.time()

    header = ['E_ID', 'Node', 'Source', 'Target', 'Link', 'Node Lat',
              'Node Lat2', 'Node Lon', 'Node Type', 'Node Degree',
              'Node Eigenvector Centrality', 'Road Type', 'Road Name', 'Bridge',
              'Ref', 'Num Lanes', 'Edge Length (km)', 'Path Length (km)',
              'Max Speed (km/h)', 'Travel Time (h)']
#    header = ['E_ID', 'Node', 'Source', 'Target', 'Link', 'Node Lat', \
#        'Node Lat2', 'Node Lon', 'Source Lat', 'Source Lon', 'Target Lat', \
#        'Target Lon', 'Road Type', 'Road Name', \
#        'Ref', 'Num Lanes', 'Path Length (km)', 'Max Speed (km/h)', \
#        'Travel Time (h)']
    colno = len(header)

    # compute node properties
    G = shackleton_create_g.node_props(G)

    #fout = open(outfile, 'w')
    #fout = codecs.open(outfile, "w", "utf-8")
    #fout.write(stringfromlist(header, delim) + '\n')

    # csv writer can't handle unicode!
    #writer = csv.writer(fout, delimiter = delim)

    # another option: unicodecsv
    fout = open(outfile, 'w')
    writer = unicodecsv.writer(fout, encoding='utf-8', delimiter=delim)
    writer.writerow(header)

    for i, e in enumerate(G.edges()):
        s, t = e
        e_props = G.edge[s][t]
        s_props = G.nodes[s]
        t_props = G.nodes[t]

        if G.edge[s][t]['e_id'].startswith('31242722'):
            print("i, edge, s, t", i, e, s, t)
            print("edge_props", G.edge[s][t])
            print("s_props", G.nodes[s])
            print("t_props", G.nodes[t])
            print('\n')

        # node properties
        slat, slon = s_props['lat'], s_props['lon']
        tlat, tlon = t_props['lat'], t_props['lon']
        stype, ttype = s_props['ntype'], t_props['ntype']
        sdeg, tdeg = s_props['deg'], t_props['deg']
        # G.nodes[s]['Eigenvector Centrality'], G.nodes[t]['Eigenvector Centrality']
        seig, teig = 0, 0,
        # edge properties
        link = e_props['Link']
        e_id = e_props['e_id']
        roadtype = e_props['Road Type']
        roadname = e_props['Road Name']
        bridge = e_props['Bridge']
        ref = e_props['ref']
        numlanes = e_props['Num Lanes']
        edgelen = e_props['Edge Length (km)']
        pathlen = e_props['Path Length (km)']
        #roadlen = e_props['Road Length (km)']
        maxspeed = e_props['Max Speed (km/h)']
        traveltime = e_props['Travel Time (h)']

        # initial row
        row = [e_id, s, s, t, link, slat, slat, slon, stype, sdeg, seig, roadtype,
               roadname, bridge, ref, numlanes, edgelen, pathlen, maxspeed, traveltime]
        if len(row) != colno:
            print("len header", len(header), "header", header)
            print("malformed row!:", "len", len(row), row)
            return
        # convert all ints and floats to strings prior to printing
        for j, item in enumerate(row):
            if type(item) == float or type(item) == int:
                string = str(item)
                row[j] = string  # .encode('utf-8', 'ignore')
            # print ("item, type(item)", row[i], type(row[i])

        # also enter target - source row
        rowrev = [e_id, t, s, t, link, tlat, tlat, tlon, ttype, tdeg, teig, roadtype,
                  roadname, bridge, ref, numlanes, edgelen, pathlen, maxspeed, traveltime]
        if len(rowrev) != colno:
            print("len header", len(header), "header", header)
            print("malformed reverse row!:", "len", len(rowrev), rowrev)
            return
        # convert all ints and floats to strings prior to printing
        for k, item in enumerate(rowrev):
            if type(item) == float or type(item) == int:
                rowrev[k] = str(item)

        writer.writerow(row)
        writer.writerow(rowrev)

        #outstring = stringfromlist(row, delim)
        #outstring2 = stringfromlist(rowrev, delim)
        #fout.write(outstring + '\n')
        #fout.write(outstring2 + '\n')

        # if (i % 1000) == 0:
        #    print i, "row:", row

    fout.close()

    print("Time to print", outfile, "graph to csv:", time.time() - t0, "seconds")


###############################################################################
def G_to_kdtree(G, coords='wmp', node_subset=[]):
    '''
    Create kd tree from node positions
    (x, y) = (lon, lat)
    return kd tree and kd_idx_dic
    kd_idx_dic maps kdtree entry to node name: kd_idx_dic[i] = n (n in G.nodes())
    node_subset is a list of nodes to consider, use all if []
    '''
    nrows = len(G.nodes())
    ncols = 2
    kd_idx_dic = {}
    arr = np.zeros((nrows, ncols))
    # populate node array
    t1 = time.time()

    i = 0
    # for i, (node, data) in enumerate(list(G.nodes(data=True))):
    for (node, data) in list(G.nodes(data=True)):
        if len(node_subset) > 0:
            if node not in node_subset:
                continue
        if coords == 'wmp':
            x, y = data['x'], data['y']
        elif coords == 'latlon':
            x, y = data['lon'], data['lat']
        arr[i] = [x, y]
        kd_idx_dic[i] = node
        i += 1

    # for i,n in enumerate(G.nodes()):
    #    n_props = G.nodes[n]
    #    lat, lon = n_props['lat'], n_props['lon']
    #    x, y = lon, lat
    #    arr[i] = [x,y]
    #    kd_idx_dic[i] = n

    # now create kdtree from numpy array
    kdtree = scipy.spatial.KDTree(arr)

    print("Time to create k-d tree:", time.time() - t1, "seconds")

    return kd_idx_dic, kdtree  # , arr


###############################################################################
def get_G_extent(G):
    '''Get extent of graph
    return [x0, y0, x1, y1]'''

    node_Xs = [float(x) for _, x in G.nodes(data='x')]
    node_Ys = [float(y) for _, y in G.nodes(data='y')]

    xmin, xmax = np.min(node_Xs), np.max(node_Xs)
    ymin, ymax = np.min(node_Ys), np.max(node_Ys)

    return xmin, ymin, xmax, ymax


###############################################################################
def make_ecurve_dic(G):
    '''Create dictionary of plotting coordinates for each graph edge
        node_Ys = [float(y) for _, y in G.nodes(data='y')]
    '''

    ecurve_dic = {}

    for u, v, data in G.edges(data=True):
        #print ("u,v,data:", u,v,data)
        if 'geometry' in data.keys():
            geom = data['geometry']
            coords = np.array(list(geom.coords))
            xs = coords[:, 0]
            ys = coords[:, 1]
        else:
            xs = np.array([G.nodes[u]['x'], G.nodes[v]['x']])
            ys = np.array([G.nodes[u]['y'], G.nodes[v]['y']])

        ecurve_dic[(u, v)] = (xs, ys)
        # also include reverse edge in list of coords?
        #ecurve_dic[(v,u)] = (xs, ys)

    return ecurve_dic


###############################################################################
### From apls.py
###############################################################################
def create_edge_linestrings(G, remove_redundant=True, verbose=False):
    '''Ensure all edges have 'geometry' tag, use shapely linestrings
    If identical edges exist, remove extras'''

    # clean out redundant edges with identical geometry
    edge_seen_set = set([])
    geom_seen = []
    bad_edges = []

    G_ = G.copy()
    for i, (u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
        # create linestring if no geometry reported
        if 'geometry' not in data:
            sourcex, sourcey = G_.nodes[u]['x'],  G_.nodes[u]['y']
            targetx, targety = G_.nodes[v]['x'],  G_.nodes[v]['y']
            line_geom = LineString([Point(sourcex, sourcey),
                                    Point(targetx, targety)])
            data['geometry'] = line_geom

            # get reversed line
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
            #G_.edge[u][v]['geometry'] = lstring

        else:
            # check which direction linestring is travelling (it may be going from
            # v -> u, which means we need to reverse the linestring)
            # otherwise splitting this edge yields a tangled edge
            line_geom = data['geometry']
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)

            # lets not reverse linestrings just yet...
#            #print (u,v,key,"create_edge_linestrings() line_geom:", line_geom)
#            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
#            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
#            geom_p0 = list(line_geom.coords)[0]
#            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
#            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
#            #print "dist_to_u, dist_to_v:", dist_to_u, dist_to_v
#            if dist_to_u > dist_to_v:
#                #data['geometry'].coords = list(line_geom.coords)[::-1]
#                data['geometry'] = line_geom_rev
#            #else:
#            #    continue
#
        # flag redundant edges
        if remove_redundant:
            if i == 0:
                edge_seen_set = set([(u, v)])
                edge_seen_set.add((v, u))
                geom_seen.append(line_geom)

            else:
                if ((u, v) in edge_seen_set) or ((v, u) in edge_seen_set):
                    # test if geoms have already been seen
                    for geom_seen_tmp in geom_seen:
                        if (line_geom == geom_seen_tmp) \
                                or (line_geom_rev == geom_seen_tmp):
                            bad_edges.append((u, v, key))
                            if verbose:
                                print("\nRedundant edge:", u, v, key)
                else:
                    edge_seen_set.add((u, v))
                    geom_seen.append(line_geom)
                    geom_seen.append(line_geom_rev)

    if remove_redundant:
        if verbose:
            print("\nedge_seen_set:", edge_seen_set)
            print("redundant edges:", bad_edges)
        for (u, v, key) in bad_edges:
            try:
                G_.remove_edge(u, v, key)
            except:
                if verbose:
                    print("Edge DNE:", u, v, key)
                pass

    return G_


###############################################################################
def cut_linestring(line, distance, verbose=False):
    '''
    Cuts a line in two at a distance from its starting point
    http://toblerity.org/shapely/manual.html#linear-referencing-methods
    '''
    """
    Cuts a shapely linestring at a specified distance from its starting point.

    Notes
    ----
    Return orignal linestring if distance <= 0 or greater than the length of
    the line.
    Reference:
        http://toblerity.org/shapely/manual.html#linear-referencing-methods

    Arguments
    ---------
    line : shapely linestring
        Input shapely linestring to cut.
    distanct : float
        Distance from start of line to cut it in two.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    [line1, line2] : list
        Cut linestrings.  If distance <= 0 or greater than the length of
        the line, return input line.
    """

    if verbose:
        print("Cutting linestring at distance", distance, "...")
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]

    # iterate through coorda and check if interpolated point has been passed
    # already or not
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pdl = line.project(Point(p))
        if verbose:
            print(i, p, "line.project point:", pdl)
        if pdl == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pdl > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

    # if we've reached here then that means we've encountered a self-loop and
    # the interpolated point is between the final midpoint and the the original
    # node
    i = len(coords) - 1
    cp = line.interpolate(distance)
    return [
        LineString(coords[:i] + [(cp.x, cp.y)]),
        LineString([(cp.x, cp.y)] + coords[i:])]


###############################################################################
def get_closest_edge_from_list(G_, edge_list_in, point, verbose=False):
    '''Return closest edge to point, and distance to said edge, from a list
    of possible edges
    Just discovered a similar function: 
        https://github.com/gboeing/osmnx/blob/master/osmnx/utils.py#L501'''

    # get distances from point to lines
    dist_list = []
    edge_list = []
    geom_list = []
    p = point # Point(point_coords)
    #print ("edge_list:", edge_list_in)
    for i, (u, v) in enumerate(edge_list_in):
        data = G_.get_edge_data(u ,v)
        
        if verbose:
            print(("get_closest_edge_from_list()  u,v,data:", u,v,data))
            #print ("data[0]:", data[0])
            #print ("data[0].keys:", data[0].keys())
            #print ("data[0]['geometry']:", data[0]['geometry'])
            #print(("  type data['geometry']:", type(data['geometry'])))
            
        try:
            line = data[0]['geometry']
        except:
            line = data['geometry']
            #line = data['attr_dict']['geometry']
        geom_list.append(line)
        dist_list.append(p.distance(line))
        edge_list.append([u, v])
    # get closest edge
    min_idx = np.argmin(dist_list)
    min_dist = dist_list[min_idx]
    best_edge = edge_list[min_idx]
    best_geom = geom_list[min_idx]

    return best_edge, min_dist, best_geom


###############################################################################
def insert_point_into_edge(G_, edge, point, node_id=100000,
                           max_distance_meters=10,
                           allow_renaming_once=False,
                           verbose=False, super_verbose=False):
    '''
    Insert a new node in the edge closest to the given point, if it is
    within max_distance_meters.  Assume all edges have a linestring geometry
    http://toblerity.org/shapely/manual.html#object.simplify
         
     Sometimes the point to insert will have the same coordinates as an 
     existing point.  If allow_renaming_once == True, relabel the existing 
     node once (after that add new nodes coincident and with edge length 0)
     # actually, renaming screws up dictionary of closest edges!!
     
    Return updated G_, 
                node_props, 
                min_dist,
                edges ([u1,v1], [u2,v2]), 
                list of edge_props,
                removed edge
    '''

    # check if node_id already exists in G
    G_node_set = set(G_.nodes())
    if node_id in G_node_set:
        print ("node_id:", node_id, "already in G, cannot insert node!")
        return
    
    # check if edge even exists
    u, v = edge
    if not G_.has_edge(u,v):
        print ("edge {} DNE!".format((u,v)))
        return
 
    p = Point(point[0], point[1])
    edge_props = G_.get_edge_data(u,v)
    try:
        line_geom = edge_props['geometry']
    except:
        line_geom = edge_props[0]['geometry']
    min_dist = p.distance(line_geom)
                 
    if verbose:
        print("Inserting point:", node_id, "coords:", point)
        print("best edge:", edge)
        print ("edge_props:", edge_props)
        print("  best edge dist:", min_dist)
        #u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
        #v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
        #print("ploc:", (point.x, point.y))
        #print("uloc:", u_loc)
        #print("vloc:", v_loc)
    
    if min_dist > max_distance_meters:
        if verbose:
            print("min_dist > max_distance_meters, skipping...")
        return G_, {}, min_dist, [], [], ()
    
    else:
        # update graph
        
        ## skip if node exists already
        #if node_id in G_node_set:
        #    if verbose:
        #        print("Node ID:", node_id, "already exists, skipping...")
        #    return #G_, {}, -1, -1

        # Length along line that is closest to the point
        line_proj = line_geom.project(p)

        # Now combine with interpolated point on line
        new_point = line_geom.interpolate(line_geom.project(p))
        x, y = new_point.x, new_point.y
        
        #################
        # create new node
        
        try:
            # first get zone, then convert to latlon
            _, _, zone_num, zone_letter = utm.from_latlon(G_.nodes[u]['lat'],
                                                          G_.nodes[u]['lon'])
            # convert utm to latlon
            lat, lon = utm.to_latlon(x, y, zone_num, zone_letter)
        except:
            lat, lon = y, x
        
        # set properties
        #props = G_.nodes[u]
        node_props = {'highway': 'insertQ',
                 'lat':     lat,
                 'lon':     lon,
                 'osmid':   node_id,
                 'x':       x,
                 'y':       y}
        ## add node
        ##G_.add_node(node_id, **node_props)
        #G_.add_node(node_id, **node_props)
        #
        ## assign, then update edge props for new edge
        #data = G_.get_edge_data(u ,v)
        #_, _, edge_props_new = copy.deepcopy(list(G_.edges([u,v], data=True))[0])
        ## remove extraneous 0 key
        
        #print ("edge_props_new.keys():", edge_props_new)
        #if list(edge_props_new.keys()) == [0]:
        #    edge_props_new = edge_props_new[0]
 
        # cut line
        split_line = cut_linestring(line_geom, line_proj)
        #line1, line2, cp = cut_linestring(line_geom, line_proj)
        if split_line == None:
            print("Failure in cut_linestring()...")
            print("type(split_line):", type(split_line))
            print("split_line:", split_line)
            print("line_geom:", line_geom)
            print("line_geom.length:", line_geom.length)
            print("line_proj:", line_proj)
            print("min_dist:", min_dist)
            return G_, {}, min_dist, [], [], ()

        if verbose:
            print("split_line:", split_line)
        
        #if cp.is_empty:        
        if len(split_line) == 1:
            if verbose:
                print("split line empty, min_dist:", min_dist)
            # get coincident node
            outnode = ''
            outnode_x, outnode_y = -1, -1
            x_p, y_p = new_point.x, new_point.y
            x_u, y_u = G_.nodes[u]['x'], G_.nodes[u]['y']
            x_v, y_v = G_.nodes[v]['x'], G_.nodes[v]['y']
            #if verbose:
            #    print "x_p, y_p:", x_p, y_p
            #    print "x_u, y_u:", x_u, y_u
            #    print "x_v, y_v:", x_v, y_v
            
            # sometimes it seems that the nodes aren't perfectly coincident,
            # so see if it's within a buffer
            buff = 0.05 # meters
            if (abs(x_p - x_u) <= buff) and (abs(y_p - y_u) <= buff):
                outnode = int(u)
                outnode_x, outnode_y = x_u, y_u
                # set node_props x,y as existing node
                node_props['x'] = outnode_x
                node_props['y'] = outnode_y
                #return G_, node_props, min_dist, [], [], ()
            elif (abs(x_p - x_v) <= buff) and (abs(y_p - y_v) <= buff):
                outnode = int(v)
                outnode_x, outnode_y = x_v, y_v
                # set node_props x,y as existing node
                node_props['x'] = outnode_x
                node_props['y'] = outnode_y
                #return G_, node_props, min_dist, [], [], ()
            elif u == v:
                # self-edge?
                outnode = int(u)
                outnode_x, outnode_y = x_u, y_u
                # set node_props x,y as existing node
                node_props['x'] = outnode_x
                node_props['y'] = outnode_y                
            else:
                print("Error in determining node coincident with node: " \
                + str(node_id) + " along edge: " + str(edge))
                print("x_p, y_p:", x_p, y_p)
                print("x_u, y_u:", x_u, y_u)
                print("x_v, y_v:", x_v, y_v)
                return #G_, (), {}, [], []
            
            if verbose:
                print ("u, v, outnode:", u, v, outnode)
                #print ("allow remaning?", allow_renaming)
                
            # if the line cannot be split, that means that the new node 
            # is coincident with an existing node.  Relabel, if desired
            # only relabel if the node value is positive.  If it's negative,
            # we'll invoke the next clause and add a new node and edge of 
            # length 0.  We do this because there could be multiple objects 
            # that have the nearest point in the graph be an original node,
            # so we'll relabel it once
            if outnode > 0 and allow_renaming_once:
                node_props = G_.nodes[outnode]
                # A dictionary with the old labels as keys and new labels as values. A partial mapping is allowed.
                mapping = {outnode: node_id}
                Gout = nx.relabel_nodes(G_, mapping)
                if verbose:
                    print("Swapping out node ids:", mapping)
                return Gout, node_props, min_dist, [], [], ()
            
            else:
            #elif 1 > 2:
                # new node is already added, presumably at the exact location
                # of an existing node.  So just  make
                # an edge from new node to existing node, length should be 0.0
                line1 = LineString([new_point, Point(outnode_x, outnode_y)])
                best_data = G_.get_edge_data(u ,v)[0]
                edge_props_line1 = copy.deepcopy(best_data)
                #edge_props_line1 = edge_props.copy()         
                edge_props_line1['length'] = line1.length
                edge_props_line1['geometry'] = line1
                edge_props_line1['travel_time'] = 0.0
                # make sure length is zero
                if line1.length > buff:
                    print("Nodes should be coincident and length 0!")
                    print ("  line1.length:", line1.length)
                    print ("  x_u, y_u :", x_u, y_u )
                    print ("  x_v, y_v :", x_v, y_v )
                    print ("  x_p, y_p :", x_p, y_p )
                    print ("  new_point:", new_point)
                    print ("  Point(outnode_x, outnode_y):", Point(outnode_x, outnode_y))
                    return
                
                if verbose:
                    print("add edge of length 0 from new node to nearest existing node")
                    print ("line1.length:", line1.length)
                G_.add_node(node_id, **node_props)
                # print ("type node_id:", type(node_id))
                # print ("type outonde:", type(outnode))
                # print (edge_props_line1:", edge_props_line1)
                G_.add_edge(node_id, outnode, **edge_props_line1)
                return G_, node_props, min_dist, \
                        [(node_id, outnode)], [edge_props_line1], \
                        ()
        
        
        else:
            # add node
            G_.add_node(node_id, **node_props)
            
            # assign, then update edge props for new edge
            best_data = G_.get_edge_data(u ,v)[0]
            edge_props_new = copy.deepcopy(best_data)
            if verbose:
                print ("edge_props_new:", edge_props_new)

            #_, _, edge_props_new = copy.deepcopy(list(G_.edges([u,v], data=True))[0])
            # remove extraneous 0 key
                # else, create new edges
                
            line1, line2 = split_line

            # get distances
            #print ("insert_point(), G_.nodes[v]:", G_.nodes[v])
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            # compare to first point in linestring
            geom_p0 = list(line_geom.coords)[0]
            # or compare to inserted point? [this might fail if line is very
            #    curved!]
            #geom_p0 = (x,y)
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # reverse edge order if v closer than u
            if dist_to_v < dist_to_u:
                line2, line1 = split_line
                
            if verbose:
                print("Creating two edges from split...")
                print("   original_length:", line_geom.length)
                print("   line1_length:", line1.length)
                print("   line2_length:", line2.length)
                print("   u, dist_u_to_point:", u, dist_to_u)
                print("   v, dist_v_to_point:", v, dist_to_v)
                print("   min_dist:", min_dist)

            # add new edges
            edge_props_line1 = edge_props_new.copy()
            edge_props_line1['length'] = line1.length
            edge_props_line1['geometry'] = line1
            # remove geometry?
            #edge_props_line1.pop('geometry', None) 
            # line2
            edge_props_line2 = edge_props_new.copy()
            edge_props_line2['length'] = line2.length
            edge_props_line2['geometry'] = line2
            # remove geometry?
            #edge_props_line1.pop('geometry', None) 

            # insert edge regardless of direction
            #G_.add_edge(u, node_id, **edge_props_line1)
            #G_.add_edge(node_id, v, **edge_props_line2)
            
            # check which direction linestring is travelling (it may be going from
            # v -> u, which means we need to reverse the linestring)
            # otherwise new edge is tangled
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            #if verbose:
            #    print "dist_to_u, dist_to_v:", dist_to_u, dist_to_v
            if dist_to_u < dist_to_v:
                G_.add_edge(u, node_id, **edge_props_line1)
                G_.add_edge(node_id, v, **edge_props_line2)
                edge_list = [(u, node_id), (node_id, v)]
            else:
                G_.add_edge(node_id, u, **edge_props_line1)
                G_.add_edge(v, node_id, **edge_props_line2)
                edge_list = [(node_id, u), (v, node_id)]

            if verbose:
                print("insert edges:", u, '-',node_id, 'and', node_id, '-', v)
                         
            # remove initial edge
            rem_edge = (u,v)
            if verbose:
                print ("removing edge:", rem_edge)
            G_.remove_edge(u, v)
                        
            return G_, node_props, min_dist, edge_list, \
                            [edge_props_line1, edge_props_line2], rem_edge



###############################################################################
### Graph props                 
###############################################################################
def create_subgraph(G, paths, weight='Travel Time (h)'):
    '''Compute properties of subgraph by removing paths from G'''
    
    # Create list of all nodes of interest 
    path_flat = set([item for sublist in paths for item in sublist])
    # make sure that sourcenodes are in path_flat
    #path_flat.update(sourcenodes)

    # find the nodes not in path_flat
    rem_nodes = set(G.nodes()) - path_flat
    #print ("len(rem_nodes)", len(rem_nodes)
    
    # copy graph and remove desired nodes 
    G3 = G.copy()
    G3.remove_nodes_from(list(rem_nodes))
    print ("Number of nodes in subgraph:", len(G3.nodes()))
    # compute ALL properties 
    G3 = graph_init._node_props(G3, weight=weight, compute_all=True) 
    return G3
       
###############################################################################
def compute_crit_nodes(G, nlist, plot_perc=10, sortlist=None):
    '''From a list of nodes, compute the critical nodes by count
    if sortlist=None, sort and group by unique counts of items in nlist
    else, sort nlist by sortlist
    keep only the top plot_perc percentage of points
    return sorted nlist and countlist'''

    if sortlist is None:    
        # determine critical nodes in paths
        # flatten array
        #node_flat = [item for sublist in paths for item in sublist]
        #freq = scipy.stats.itemfreq(node_flat)
        unique0, counts0 = np.unique(nlist, return_counts=True)
    else:
        unique0, counts0 = np.asarray(nlist), np.asarray(sortlist)
        
    # sort descending by max counts
    f0 = np.argsort(counts0)[::-1]
    crit_nodes = unique0[f0]
    crit_counts = counts0[f0]
    #####
    # optional: remove nodes with degree of <=2
    rem_idx = []
    for i,n in enumerate(crit_nodes):
        deg = G.degree[n]
        #print ( "compute_crit_nodes():' deg", deg)
        #deg = G.nodes[n]['deg']
        if deg <= 2:
            rem_idx.append(i)
    crit_nodes = np.delete(crit_nodes, rem_idx)
    crit_counts = np.delete(crit_counts, rem_idx)
    #####
    # keep all nodes above a percentile threshold (already filtered above...)
    #thresh = scipy.percentile(crit_counts, 80) 
    #f1 = np.where(crit_counts >= thresh)
    #crit_nodes = crit_nodes[f1]
    #crit_counts = crit_counts[f1]
    ####
    numN = int(len(unique0) * (plot_perc / 100.))
    #print ("Top", int(plot_perc), "%, critical nodes, counts:", \
    #        zip(crit_nodes[:numN], crit_counts[:numN])
    print ("Top 10, critical nodes, counts:", \
            (crit_nodes[:10], crit_counts[:10]))
    
#    ####################     
#    # set sizes
#    crit_size, Ac, Bc = log_scale(crit_counts[:numN], minS, maxS)
#    # finally, plot these points
#    p, sourceB = plot_nodes(G, p, crit_nodes[:numN], \
#        gmap_background=gmap_background, \
#        size=crit_size, color=crit_node_color, labels=crit_counts[:numN], \
#        nid=crit_nodes[:numN], count=crit_counts[:numN])   
#    #################### 
        
    return crit_nodes[:numN], crit_counts[:numN]
    
###############################################################################
def compute_paths(G, nodes, ecurve_dic=None, target=None, skipnodes=[], \
            weight='Travel Time (h)', alt_G=None, goodroutes=None,
            verbose=False):
    '''Compute and plot all shortest paths between nodes
    if target=None, compute all paths between set(nodes), else only find paths 
    between set(nodes) and target
    skipnodes is an option to remove certain nodes from the graph because 
        nodes may be impassable or blocked
    all plotting is external to this function
    return esource, sourcenodes, paths, lengths, sourcenode_vals, missingnodes
    alt_G is a separate graph (with original edge weights) to use for 
        computing path lengths'''

    if verbose:
        print("compute_paths() - weight:", weight)

    color_dic, alpha_dic = define_colors_alphas()
    # global_dic = global_vars()
    
    # make sure target is not in skipnodes!
    if len(skipnodes) != 0 and target is not None:
        skipnodes = list( set(skipnodes)  - set([target]) ) 
    
    # # set path color
    # if goodroutes == True:
    #     line_color = color_dic['compute_path_color_good']
    # elif goodroutes == False:
    #     line_color = color_dic['compute_path_color_bad']
    # else:
    #     line_color=color_dic['compute_path_color']
    # # set widths
    # const_width = True    # all paths are same width
    # line_width = global_dic['compute_path_width']
    # # set alpha
    # if alt_G:
    #     line_alpha = alpha_dic['compute_paths_sec']
    # else:
    #     line_alpha = alpha_dic['compute_paths']
    # #line_alpha=0.4
 
#    # copy graph  (otherwise changes are global?????)
#    G2 = G#.copy()
#    # remove desired nodes
#    if len(skipnodes) != 0:
#        G2.remove_nodes_from(skipnodes)

    # copy graph  (otherwise changes are global?????)
    # remove desired nodes
    if len(skipnodes) != 0:
        G2 = G.copy()
        G2.remove_nodes_from(skipnodes)
    else:
        G2 = G

    if target in skipnodes:
        print ("ERROR!!")
        print ("target", target)
        print ("skipnodes", skipnodes)
        
    # plot routes from nodes to target
    if target:
        t1 = time.time()
        lengthd, pathd = nx.single_source_dijkstra(G2, source=target, weight=weight) 
        # not all nodes may be reachable from N0, so find intersection
        sourcenodes = list(set(pathd.keys()).intersection(set(nodes)))
        missingnodes = list(set(nodes) - set(sourcenodes))
        #missingnodes_count = [g_node_props_dic[n]['count'] for n in missingnodes]
        paths = [pathd[k] for k in sourcenodes]
        lengths = [lengthd[k] for k in sourcenodes]
        # if alt_G, recompute lengths
        if alt_G:
            lengths = compute_path_lengths(alt_G, paths, weight=weight)
        # set vals as lengths
        sourcenode_vals = [str(round(l,2)) + 'H' for l in lengths]
            
    # plot LOCs between set of nodes
    else:     
        # alternative: overkill to compute all paths, is actually faster...
        t1 = time.time()
        paths, lengths, sourcenode_vals, sourcenodes, missingnodes = [],[], [], [], []
        #missingnodes = set([])
        for k,n0 in enumerate(nodes):
            # get all paths from source
            lengthd, pathd = nx.single_source_dijkstra(G2, source=n0, weight=weight) 
            # check if node is cut off from all other nodes, if so
            # the path dictionary will only contain the source node
            
            # not all nodes may be reachable from N0, so find intersection
            startnodes = list(set(pathd.keys()).intersection(set(nodes)) - set([n0]))
            #print ("n0, len(startnodes)", n0, len(startnodes)
            if len(startnodes) == 0:
                missingnodes.append(n0)
                print ("Node", n0, "cut off from other nodes"  )   
                continue 
            else:
                sourcenodes.append(n0)                
            pathsn = [pathd[k] for k in startnodes]
            lengthsn = [lengthd[k] for k in startnodes]
            if alt_G:
                lengthsn = compute_path_lengths(alt_G, pathsn, weight=weight)
            # set vals as medial path length 
            val = "Median time to nodes: " + str(round(np.median(lengthsn),2)) + 'H'
            sourcenode_vals.append(val)
            #print ("pathsn", pathsn
            #print ("lengthn", lengthsn
            paths.extend(pathsn)
            lengths.extend(lengthsn)
    
    if verbose:
        print ("compute_paths() - lengths:", lengths)
            
    print ("Time to compute paths:", time.time() - t1, "seconds")
    # Time to compute paths: 1.38523888588 seconds

    # # create the paths
    # # assume np arrays, not list (except for edgelist)
    # #ex0, ey0, ex1, ey1, emx, emy, elen, edgelist = get_path_pos(G2, paths, \
    # #        lengths, skipset=set())
    # ex0, ey0, ex1, ey1, elat0, elon0, elat1, elon1, emx, emy, elen, edgelist =\
    #         get_path_pos(G2, paths, lengths, skipset=set())
    #
    # # for constant width!
    # if const_width:
    #     ewidth = line_width*np.ones(len(ex0))#len(ex0)*[line_width]
    #
    # # create columndatasource
    # esource = bk.ColumnDataSource(
    #     data=dict(
    #         ex0=ex0, ey0=ey0,
    #         ex1=ex1, ey1=ey1,
    #         elat0=elat0, elon0=elon0,
    #         elat1=elat1, elon1=elon1,
    #         emx=emx, emy=emy,
    #         ewidth=ewidth,
    #         ecolor=np.asarray(len(ex0)*[line_color]),
    #         ealpha=line_alpha*np.ones(len(ex0))#len(ex0)*[line_alpha]
    #     )
    # )
    #
    # # add coords for MultiLine if desired
    # if ecurve_dic is not None:
    #     elx0, ely0, ellat0, ellon0 = get_ecurves(edgelist, ecurve_dic)
    #     # add to esource
    #     esource.data['elx0'] = elx0
    #     esource.data['ely0'] = ely0
    #     esource.data['ellon0'] = ellon0
    #     esource.data['ellat0'] = ellat0
       
    return sourcenodes, paths, lengths, sourcenode_vals, missingnodes
    
###############################################################################
def reweight_paths(G, bestpaths, weight='Travel Time (h)', weight_mult=3.):
    '''    Multiply bestpath edge weights in G by weight_mult
    The returned graph Gc will have altered edge weights so that secondary
    paths can be computed'''

    '''Compute secondary paths in G
    Multiply edge weights by weight_mult, and recompute best paths
    return best paths with these altered weights
    path lengths will be incorrect'''
    
    #print ("paths", bestpaths
    # copy graph (and remove skipnodes?)
    Gc = G.copy()
    #if len(skipnodes) != 0:
    #    Gc.remove_nodes_from(skipnodes)  
    # add weights to paths already seen
    # flatten paths
    edgeset = set()
    for path in bestpaths:
        for i in range(len(path)-2):
            s,t = path[i], path[i+1]
            # add edge and reversed edge to edgeset
            edgeset.add((s,t))
            edgeset.add((t,s))
    
    #print ("edgeset", edgeset
    #print ("len edgeset", len(edgeset)
    # increase weights to Gc
    seen_edges = set()
    for edge in edgeset:
        (s, t) = edge
        # since G is not directed, skip reversed edges
        if (t, s) in seen_edges:
            continue
        else:
            try :
                w0 = Gc.edges[s,t][weight]
                Gc.edges[s,t][weight] = w0 * weight_mult
            except:
                w0 = Gc.edges[s,t,0][weight]
                Gc.edges[s,t,0][weight] = w0 * weight_mult                
            #w0 = Gc.edges[s][t][weight]
            #Gc.edges[s][t][weight] = w0 * weight_mult
            seen_edges.add(edge)
            
    return Gc

###############################################################################
def compute_path_lengths(G, paths, weight='Travel Time (h)'):
    '''compute length of known paths'''
    lengths_out = np.zeros(len(paths))#len(paths) * [0.0]
    for i,path in enumerate(paths):
        edges = [(path[j], path[j+1]) for j in range(len(path)-2)]
        try:
            length = np.sum([G.edges[e[0], e[1]][weight] for e in edges])
        except:
            length = np.sum([G.edges[e[0], e[1], 0][weight] for e in edges])

        #ƒlength = np.sum([G.edges[e[0]][e[1]][weight] for e in edges])
        lengths_out[i] = length
    return lengths_out



###############################################################################                            
def path_counts(G, end_nodes, skipnodes=[], goodroutes=True,
                target=None, compute_secondary_routes=True,
                edge_weight='Travel Time (h)', verbose=False):
    '''Return counts of nodes traversed on paths
    Similar to compute_crit_nodes, though this function also computes paths'''
    
    #half_sec_counts = True      # switch to half secondary route counts

    # compute paths
    t01 =  time.time()
    sourcenodes, paths, lengths, sourcenode_vals, missingnodes =  \
                        compute_paths(G, end_nodes, target=target, \
                            skipnodes=skipnodes, \
                            weight=edge_weight, #global_dic['edge_weight'], 
                            alt_G=None, goodroutes=goodroutes) 
    if verbose:
        print ("Time to compute routes:", time.time()-t01, "seconds")

    #######
    # optional, compute secondary routes
    if compute_secondary_routes:
        t02 = time.time()
        Grw = reweight_paths(G, paths, 
                             weight=edge_weight, #global_dic['edge_weight'], 
                             #weight_mult=global_dic['weight_mult'] 
                             )  
        # now compute new paths
        sourcenodesrw, pathsrw, lengthsrw, sourcenode_valsrw, \
            missingnodesrw =  \
                        compute_paths(Grw, end_nodes, target=target, \
                                    skipnodes=skipnodes, \
                                    weight=edge_weight, #global_dic['edge_weight'], 
                                    alt_G=G, goodroutes=goodroutes) 
        if verbose:
            print ("Time to compute secondary routes:", time.time()-t02, "seconds")
    #######
                            
    #t03 = time.time()
    # compute critical nodes, sorting by counts of nodes in paths
    node_flat = [item for sublist in paths for item in sublist]
    if compute_secondary_routes:
        node_flatrw = [item for sublist in pathsrw for item in sublist]
        # combine two lists
        node_flat = node_flat + node_flatrw
    # compute all nodes and counts
    crit_perc = 100
    crit_nodes, crit_counts = compute_crit_nodes(G, node_flat, \
            plot_perc=crit_perc, sortlist=None) 
            
    return crit_nodes, crit_counts
    