#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 18:45:08 2018

@author: gykovacs
"""

import rare_databases as rd

#############################################################
# this code can be used to query attribute and feature nums #
#############################################################


# structure: loading function, number of rows, number of raw features, number of encoded features
descriptors= {'ada': [rd.load_ada, 4147, 48, 47],
              'cm1': [rd.load_cm1, 498, 21, 23],
              'german': [rd.load_german, 1000, 20, 68],
              'glass': [rd.load_glass, 214, 9, 9],
              'hepatitis': [rd.load_hepatitis, 155, 19, 39],
              'hiva': [rd.load_hiva, 3845, 1617, 1617],
              'hypothyroid': [rd.load_hypothyroid, 3163, 25, 27],
              'kc1': [rd.load_kc1, 2109, 21, 21],
              'pc1': [rd.load_pc1, 1109, 21, 21],
              'satimage': [rd.load_satimage, 6435, 36, 36],
              'spect_f': [rd.load_spectf, 267, 44, 44],
              'sylva': [rd.load_sylva, 13086, 216, 212],
              'vehicle': [rd.load_vehicle, 846, 18, 18]}

def read_and_validate(name):
    """
    Tests the dimensions of the datasets
    Args:
        name (str): the name of the dataset
    """
    data_0= descriptors[name][0](return_X_y= False, encode= False)
    data_1= descriptors[name][0](return_X_y= True, encode= False)
    data_2= descriptors[name][0](return_X_y= False, encode= True)
    data_3= descriptors[name][0](return_X_y= True, encode= True)
    
    assert len(data_0['data']) == descriptors[name][1]
    assert len(data_1[0]) == descriptors[name][1]
    assert len(data_2['data']) == descriptors[name][1]
    assert len(data_3[0]) == descriptors[name][1]
    
    assert len(data_0['target']) == descriptors[name][1]
    assert len(data_1[1]) == descriptors[name][1]
    assert len(data_2['target']) == descriptors[name][1]
    assert len(data_3[1]) == descriptors[name][1]
    
    assert len(data_0['data'][0]) == descriptors[name][2]
    assert len(data_1[0][0]) == descriptors[name][2]
    assert len(data_2['data'][0]) == descriptors[name][3]
    assert len(data_3[0][0]) == descriptors[name][3]
    
    assert len(data_0['feature_names']) == descriptors[name][2]
    assert len(data_2['feature_names']) == descriptors[name][3]

def test_ada(): read_and_validate('ada')
    
def test_cm1(): read_and_validate('cm1')

def test_german(): read_and_validate('german')
    
def test_glass(): read_and_validate('glass')
    
def test_hepatitis(): read_and_validate('hepatitis')
    
def test_hiva(): read_and_validate('hiva')
    
def test_hypothyroid(): read_and_validate('hypothyroid')
    
def test_kc1(): read_and_validate('kc1')
    
def test_pc1(): read_and_validate('pc1')
    
def test_satimage(): read_and_validate('satimage')
    
def test_spect_f(): read_and_validate('spect_f')
    
def test_sylva(): read_and_validate('sylva')
    
def test_vehicle(): read_and_validate('vehicle')
    
