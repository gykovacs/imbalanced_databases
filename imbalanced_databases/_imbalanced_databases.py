#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 00:00:09 2018

@author: gykovacs
"""

__all__= ['load_ada',
'load_cm1',
'load_german',
'load_glass',
'load_hepatitis',
'load_hiva',
'load_hypothyroid',
'load_kc1',
'load_pc1',
'load_satimage',
'load_spectf',
'load_sylva',
'load_vehicle',
'load_abalone_17_vs_7_8_9_10',
'load_abalone_19_vs_10_11_12_13',
'load_abalone_20_vs_8_9_10',
'load_abalone_21_vs_8',
'load_abalone_3_vs_11',
'load_abalone19',
'load_abalone9_18',
'load_car_good',
'load_car_vgood',
'load_cleveland_0_vs_4',
'load_dermatology_6',
'load_ecoli_0_1_3_7_vs_2_6',
'load_ecoli_0_1_4_6_vs_5',
'load_ecoli_0_1_4_7_vs_2_3_5_6',
'load_ecoli_0_1_4_7_vs_5_6',
'load_ecoli_0_1_vs_2_3_5',
'load_ecoli_0_1_vs_5',
'load_ecoli_0_2_3_4_vs_5',
'load_ecoli_0_2_6_7_vs_3_5',
'load_ecoli_0_3_4_6_vs_5',
'load_ecoli_0_3_4_7_vs_5_6',
'load_ecoli_0_3_4_vs_5',
'load_ecoli_0_4_6_vs_5',
'load_ecoli_0_6_7_vs_3_5',
'load_ecoli_0_6_7_vs_5',
'load_ecoli4',
'load_flaref',
'load_glass_0_1_4_6_vs_2',
'load_glass_0_1_5_vs_2',
'load_glass_0_1_6_vs_2',
'load_glass_0_1_6_vs_5',
'load_glass_0_4_vs_5',
'load_glass_0_6_vs_5',
'load_glass2',
'load_glass4',
'load_glass5',
'load_kddcup_buffer_overflow_vs_back',
'load_kddcup_guess_passwd_vs_satan',
'load_kddcup_land_vs_portsweep',
'load_kddcup_land_vs_satan',
'load_kddcup_rootkit_imap_vs_back',
'load_kr_vs_k_one_vs_fifteen',
'load_kr_vs_k_three_vs_eleven',
'load_kr_vs_k_zero_one_vs_draw',
'load_kr_vs_k_zero_vs_eight',
'load_kr_vs_k_zero_vs_fifteen',
'load_led7digit_0_2_4_5_6_7_8_9_vs_1',
'load_lymphography_normal_fibrosis',
'load_page_blocks_1_3_vs_4',
'load_poker_8_9_vs_5',
'load_poker_8_9_vs_6',
'load_poker_8_vs_6',
'load_poker_9_vs_7',
'load_shuttle_2_vs_5',
'load_shuttle_6_vs_2_3',
'load_shuttle_c0_vs_c4',
'load_shuttle_c2_vs_c4',
'load_vowel0',
'load_winequality_red_3_vs_5',
'load_winequality_red_4',
'load_winequality_red_8_vs_6',
'load_winequality_red_8_vs_6_7',
'load_winequality_white_3_9_vs_5',
'load_winequality_white_3_vs_7',
'load_winequality_white_9_vs_4',
'load_yeast_0_2_5_6_vs_3_7_8_9',
'load_yeast_0_2_5_7_9_vs_3_6_8',
'load_yeast_0_3_5_9_vs_7_8',
'load_yeast_0_5_6_7_9_vs_4',
'load_yeast_1_2_8_9_vs_7',
'load_yeast_1_4_5_8_vs_7',
'load_yeast_1_vs_7',
'load_yeast_2_vs_4',
'load_yeast_2_vs_8',
'load_yeast4',
'load_yeast5',
'load_yeast6',
'load_zoo_3',
'load_ecoli_0_vs_1',
'load_ecoli1',
'load_ecoli2',
'load_ecoli3',
'load_glass_0_1_2_3_vs_4_5_6',
'load_glass0',
'load_glass1',
'load_glass6',
'load_haberman',
'load_iris0',
'load_new_thyroid1',
'load_new_thyroid2',
'load_page_blocks0',
'load_pima',
'load_segment0',
'load_vehicle0',
'load_vehicle1',
'load_vehicle2',
'load_vehicle3',
'load_wisconsin',
'load_yeast1',
'load_yeast3',
'summary',
'get_all_data_loaders',
'get_filtered_data_loaders']

import copy
import sys
import pkgutil
import io

# for the representation of the data
import numpy as np
import pandas as pd

# for the encoding of the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.io import arff

def encode_column_onehot(column):
    lencoder= LabelEncoder().fit(column)
    lencoded= lencoder.transform(column)
    ohencoder= OneHotEncoder(sparse= False).fit(lencoded.reshape(-1,1))
    ohencoded= ohencoder.transform(lencoded.reshape(-1,1))
    
    return ohencoded

def encode_column_label(column):
    lencoder= LabelEncoder().fit(column)
    return lencoder.transform(column)

def encode_column_median(column, missing_values):
    column= copy.deepcopy(column)
    if np.sum(column.isin(missing_values)) > 0:
        column[column.isin(missing_values)]= np.median(column[~column.isin(missing_values)].astype(float))
    column= column.astype(float)
    return column.values

def encode_features(data, target= 'target', encoding_threshold= 4, missing_values= ['?', None, 'None'], verbose= True):
    columns= []
    column_names= []
    
    for c in data.columns:
        if verbose:
            sys.stdout.write('Encoding column %s' % c)
        if not c == target:
            # if the column is not the target variable
            n_values= len(np.unique(data[c]))
            if verbose:
                sys.stdout.write(' number of values: %d => ' % n_values)
            
            if n_values == 1:
                # there is no need for encoding
                if verbose:
                    sys.stdout.write('no encoding\n')
                continue
            if n_values == 2 or data[c].dtype == object:
                # applying label encoding
                if verbose:
                    sys.stdout.write('label encoding\n')
                columns.append(encode_column_label(data[c]))
                column_names.append(c)
            elif n_values < encoding_threshold:
                # applying one-hot encoding
                if verbose:
                    sys.stdout.write('one-hot encoding\n')
                ohencoded= encode_column_onehot(data[c])
                for i in range(ohencoded.shape[1]):
                    columns.append(ohencoded[:,i])
                    column_names.append(str(c) + '_onehot_' + str(i))
            else:
                # applying median encoding
                if verbose:
                    sys.stdout.write('no encoding, missing values replaced by medians\n')
                columns.append(encode_column_median(data[c], missing_values))
                column_names.append(c)

        if c == target:
            # in the target column the least frequent value is set to 1, the
            # rest is set to 0
            if verbose:
                sys.stdout.write(' target variable => least frequent value is 1\n')
            column= copy.deepcopy(data[c])
            val_counts= data[target].value_counts()
            if val_counts.values[0] < val_counts.values[1]:
                mask= (column == val_counts.index[0])
                column[mask]= 1
                column[~(mask)]= 0
            else:
                mask= (column == val_counts.index[0])
                column[mask]= 0
                column[~(mask)]= 1

            columns.append(column.astype(int).values)
            column_names.append(target)
    
    return pd.DataFrame(np.vstack(columns).T, columns= column_names)

def construct_return_set(database, descriptor, return_X_y, encode, encoding_threshold= 4, verbose= True):
    if return_X_y == True and encode == False:
        return database.drop('target', axis= 'columns').values, database['target'].values
    
    if encode == True:
        database= encode_features(database, encoding_threshold= encoding_threshold, verbose= verbose)
        if return_X_y == True:
            return database.drop('target', axis= 'columns').values, database['target'].values
    
    descriptors= {}
    descriptors['DESCR']= descriptor
    features= database.drop('target', axis= 'columns')
    descriptors['data']= features.values
    descriptors['feature_names']= list(features.columns)
    descriptors['target']= database['target'].values
    
    return descriptors

def read_csv_data(filename, sep= ',', usecols= None):
    return pd.read_csv(io.BytesIO(pkgutil.get_data('rare_databases', filename)), sep= sep, header= None, usecols= usecols)

def read_arff_data(filename, sep= ',', usecols= None):
    if sys.version_info >= (3,0):
        return arff.loadarff(io.StringIO(pkgutil.get_data('rare_databases', filename).decode('unicode_escape')))
    else:
        from cStringIO import StringIO
        return arff.loadarff(StringIO(unicode(str(pkgutil.get_data('rare_databases', filename)).decode('string_escape'), "utf-8")))

def load_hiva(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/hiva/hiva_train.data', sep= ' ')
    del db[db.columns[-1]]
    target= read_csv_data('data/hiva/hiva_train.labels')
    db['target']= target
    
    return construct_return_set(db, "HIVA", return_X_y, encode, verbose= verbose)

def load_hypothyroid(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/hypothyroid/hypothyroid.data.txt')
    db.columns= ['target'] + list(db.columns[1:])
    
    return construct_return_set(db, "hypothyroid dataset", return_X_y, encode, verbose= verbose)

def load_sylva(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/sylva/sylva_train.data', sep= ' ')
    del db[db.columns[-1]]
    target= read_csv_data('data/sylva/sylva_train.labels')
    db['target']= target
    
    return construct_return_set(db, "sylva", return_X_y, encode, verbose= verbose)

def load_pc1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/pc1/pc1.arff')
    db= pd.DataFrame(data)
    db.loc[db['defects'] == b'false', 'defects']= False
    db.loc[db['defects'] == b'true', 'defects']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "PC1", return_X_y, encode, verbose= verbose)

def load_cm1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/cm1/cm1.arff.txt')
    db= pd.DataFrame(data)
    db.loc[db['defects'] == b'false', 'defects']= False
    db.loc[db['defects'] == b'true', 'defects']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "CM1", return_X_y, encode, verbose= verbose)

def load_kc1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/kc1/kc1.arff.txt')
    db= pd.DataFrame(data)
    db.loc[db['defects'] == b'false', 'defects']= False
    db.loc[db['defects'] == b'true', 'defects']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "KC1", return_X_y, encode, verbose= verbose)

def load_spectf(return_X_y= False, encode= True, verbose= False):
    db0= read_csv_data('data/spect_f/SPECTF.train.txt')
    db1= read_csv_data('data/spect_f/SPECTF.test.txt')
    db= pd.concat([db0, db1])
    db.columns= ['target'] + list(db.columns[1:])
    
    return construct_return_set(db, "SPECT_F", return_X_y, encode, verbose= verbose)

def load_hepatitis(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/hepatitis/hepatitis.data.txt')
    db.columns= ['target'] + list(db.columns[1:])

    return construct_return_set(db, "hepatitis", return_X_y, encode, verbose= verbose)

def load_vehicle(return_X_y= False, encode= True, verbose= False):
    db0= read_csv_data('data/vehicle/xaa.dat.txt', sep= ' ', usecols= range(19))
    db1= read_csv_data('data/vehicle/xab.dat.txt', sep= ' ', usecols= range(19))
    db2= read_csv_data('data/vehicle/xac.dat.txt', sep= ' ', usecols= range(19))
    db3= read_csv_data('data/vehicle/xad.dat.txt', sep= ' ', usecols= range(19))
    db4= read_csv_data('data/vehicle/xae.dat.txt', sep= ' ', usecols= range(19))
    db5= read_csv_data('data/vehicle/xaf.dat.txt', sep= ' ', usecols= range(19))
    db6= read_csv_data('data/vehicle/xag.dat.txt', sep= ' ', usecols= range(19))
    db7= read_csv_data('data/vehicle/xah.dat.txt', sep= ' ', usecols= range(19))
    db8= read_csv_data('data/vehicle/xai.dat.txt', sep= ' ', usecols= range(19))
    
    db= pd.concat([db0, db1, db2, db3, db4, db5, db6, db7, db8])
    
    db.columns= list(db.columns[:-1]) + ['target']
    db.loc[db['target'] != 'van', 'target']= 'other'
    
    return construct_return_set(db, "vehicle", return_X_y, encode, verbose= verbose)

def load_ada(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/ada/ada_train.data', sep= ' ')
    del db[db.columns[-1]]
    target= read_csv_data('data/ada/ada_train.labels')
    db['target']= target
    
    return construct_return_set(db, "ADA", return_X_y, encode, verbose= verbose)

def load_german(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/german/german.data.txt', sep= ' ')
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "german", return_X_y, encode, encoding_threshold= 20, verbose= verbose)

def load_glass(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/glass/glass.data.txt')
    db.columns= list(db.columns[:-1]) + ['target']
    db.loc[db['target'] != 3, 'target']= 0
    del db[db.columns[0]]
    
    return construct_return_set(db, "glass", return_X_y, encode, verbose= verbose)

def load_satimage(return_X_y= False, encode= True, verbose= False):
    db0= read_csv_data('data/satimage/sat.trn.txt', sep= ' ')
    db1= read_csv_data('data/satimage/sat.tst.txt', sep= ' ')
    db= pd.concat([db0, db1])
    db.columns= list(db.columns[:-1]) + ['target']
    db.loc[db['target'] != 4, 'target']= 0
    
    return construct_return_set(db, "SATIMAGE", return_X_y, encode, verbose= verbose)

def load_abalone_17_vs_7_8_9_10(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/abalone-17_vs_7-8-9-10/abalone-17_vs_7-8-9-10.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone_17_vs_7_8_9_10", return_X_y, encode, verbose= verbose)

def load_abalone_19_vs_10_11_12_13(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/abalone-19_vs_10-11-12-13/abalone-19_vs_10-11-12-13.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone-19_vs_10-11-12-13", return_X_y, encode, verbose= verbose)

def load_abalone_20_vs_8_9_10(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/abalone-20_vs_8-9-10/abalone-20_vs_8-9-10.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone-20_vs_8-9-10", return_X_y, encode, verbose= verbose)

def load_abalone_21_vs_8(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/abalone-21_vs_8/abalone-21_vs_8.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone-21_vs_8", return_X_y, encode, verbose= verbose)

def load_abalone_3_vs_11(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/abalone-3_vs_11/abalone-3_vs_11.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone-3_vs_11", return_X_y, encode, verbose= verbose)

def load_abalone19(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/abalone19/abalone19.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone19", return_X_y, encode, verbose= verbose)

def load_abalone9_18(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/abalone9-18/abalone9-18.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone9-18", return_X_y, encode, verbose= verbose)

def load_car_good(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/car-good/car-good.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "car_good", return_X_y, encode, verbose= verbose)

def load_car_vgood(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/car-vgood/car-vgood.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "car-vgood", return_X_y, encode, verbose= verbose)

def load_cleveland_0_vs_4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/cleveland-0_vs_4/cleveland-0_vs_4_no_null.dat')
    db= pd.DataFrame(data)
    db.loc[db['num'] == b'negative', 'num']= False
    db.loc[db['num'] == b'positive', 'num']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "cleveland-0_vs_4", return_X_y, encode, verbose= verbose)

def load_dermatology_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/dermatology-6/dermatology-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "dermatology-6", return_X_y, encode, verbose= verbose)

def load_ecoli_0_1_3_7_vs_2_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0-1-3-7_vs_2-6/ecoli-0-1-3-7_vs_2-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1-3-7_vs_2-6", return_X_y, encode, verbose= verbose)

def load_ecoli_0_1_4_6_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0-1-4-6_vs_5/ecoli-0-1-4-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1-4-6_vs_5", return_X_y, encode, verbose= verbose)

def load_ecoli_0_1_4_7_vs_2_3_5_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0-1-4-7_vs_2-3-5-6/ecoli-0-1-4-7_vs_2-3-5-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1-4-7_vs_2-3-5-6", return_X_y, encode, verbose= verbose)

def load_ecoli_0_1_4_7_vs_5_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0-1-4-7_vs_5-6/ecoli-0-1-4-7_vs_5-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1-4-7_vs_5-6", return_X_y, encode, verbose= verbose)

def load_ecoli_0_1_vs_2_3_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0-1_vs_2-3-5/ecoli-0-1_vs_2-3-5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1_vs_2-3-5", return_X_y, encode, verbose= verbose)

def load_ecoli_0_1_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0-1_vs_5/ecoli-0-1_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1_vs_5", return_X_y, encode, verbose= verbose)

def load_ecoli_0_2_3_4_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0-2-3-4_vs_5/ecoli-0-2-3-4_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-2-3-4_vs_5", return_X_y, encode, verbose= verbose)

def load_ecoli_0_2_6_7_vs_3_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0-2-6-7_vs_3-5/ecoli-0-2-6-7_vs_3-5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-2-6-7_vs_3-5", return_X_y, encode, verbose= verbose)

def load_ecoli_0_3_4_6_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0-3-4-6_vs_5/ecoli-0-3-4-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-3-4-6_vs_5", return_X_y, encode, verbose= verbose)

def load_ecoli_0_3_4_7_vs_5_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0-3-4-7_vs_5-6/ecoli-0-3-4-7_vs_5-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-3-4-7_vs_5-6", return_X_y, encode, verbose= verbose)

def load_ecoli_0_3_4_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0-3-4_vs_5/ecoli-0-3-4_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-3-4_vs_5", return_X_y, encode, verbose= verbose)

def load_ecoli_0_4_6_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0-4-6_vs_5/ecoli-0-4-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-4-6_vs_5", return_X_y, encode, verbose= verbose)

def load_ecoli_0_6_7_vs_3_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0-6-7_vs_3-5/ecoli-0-6-7_vs_3-5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-6-7_vs_3-5", return_X_y, encode, verbose= verbose)

def load_ecoli_0_6_7_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0-6-7_vs_5/ecoli-0-6-7_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-6-7_vs_5", return_X_y, encode, verbose= verbose)

def load_ecoli4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli4/ecoli4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli4", return_X_y, encode, verbose= verbose)

def load_flaref(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/flare-F/flare-F.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "flare-F", return_X_y, encode, verbose= verbose)

def load_glass_0_1_4_6_vs_2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/glass-0-1-4-6_vs_2/glass-0-1-4-6_vs_2.dat')
    db= pd.DataFrame(data)
    db.loc[db['typeGlass'] == b'negative', 'typeGlass']= False
    db.loc[db['typeGlass'] == b'positive', 'typeGlass']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-4-6_vs_2", return_X_y, encode, verbose= verbose)

def load_glass_0_1_5_vs_2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/glass-0-1-5_vs_2/glass-0-1-5_vs_2.dat')
    db= pd.DataFrame(data)
    db.loc[db['typeGlass'] == b'negative', 'typeGlass']= False
    db.loc[db['typeGlass'] == b'positive', 'typeGlass']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-5_vs_2", return_X_y, encode, verbose= verbose)

def load_glass_0_1_6_vs_2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/glass-0-1-6_vs_2/glass-0-1-6_vs_2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-6_vs_2", return_X_y, encode, verbose= verbose)

def load_glass_0_1_6_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/glass-0-1-6_vs_5/glass-0-1-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-6_vs_5", return_X_y, encode, verbose= verbose)

def load_glass_0_4_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/glass-0-4_vs_5/glass-0-4_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['typeGlass'] == b'negative', 'typeGlass']= False
    db.loc[db['typeGlass'] == b'positive', 'typeGlass']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-4_vs_5", return_X_y, encode, verbose= verbose)

def load_glass_0_6_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/glass-0-6_vs_5/glass-0-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['typeGlass'] == b'negative', 'typeGlass']= False
    db.loc[db['typeGlass'] == b'positive', 'typeGlass']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-6_vs_5", return_X_y, encode, verbose= verbose)

def load_glass2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/glass2/glass2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass2", return_X_y, encode, verbose= verbose)

def load_glass4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/glass4/glass4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass4", return_X_y, encode, verbose= verbose)

def load_glass5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/glass5/glass5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass5", return_X_y, encode, verbose= verbose)

def load_kddcup_buffer_overflow_vs_back(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/kddcup-buffer_overflow_vs_back/kddcup-buffer_overflow_vs_back.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-buffer_overflow_vs_back", return_X_y, encode, verbose= verbose)

def load_kddcup_guess_passwd_vs_satan(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/kddcup-guess_passwd_vs_satan/kddcup-guess_passwd_vs_satan.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-guess_passwd_vs_satan", return_X_y, encode, verbose= verbose)

def load_kddcup_land_vs_portsweep(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/kddcup-land_vs_portsweep/kddcup-land_vs_portsweep.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-land_vs_portsweep", return_X_y, encode, verbose= verbose)

def load_kddcup_land_vs_satan(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/kddcup-land_vs_satan/kddcup-land_vs_satan.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-land_vs_satan", return_X_y, encode, verbose= verbose)

def load_kddcup_rootkit_imap_vs_back(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/kddcup-rootkit-imap_vs_back/kddcup-rootkit-imap_vs_back.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-rootkit-imap_vs_back", return_X_y, encode, verbose= verbose)

def load_kr_vs_k_one_vs_fifteen(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/kr-vs-k-one_vs_fifteen/kr-vs-k-one_vs_fifteen.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-one_vs_fifteen", return_X_y, encode, verbose= verbose)

def load_kr_vs_k_three_vs_eleven(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/kr-vs-k-three_vs_eleven/kr-vs-k-three_vs_eleven.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-three_vs_eleven", return_X_y, encode, verbose= verbose)

def load_kr_vs_k_zero_one_vs_draw(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/kr-vs-k-zero-one_vs_draw/kr-vs-k-zero-one_vs_draw.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-zero-one_vs_draw", return_X_y, encode, verbose= verbose)

def load_kr_vs_k_zero_vs_eight(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/kr-vs-k-zero_vs_eight/kr-vs-k-zero_vs_eight.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-zero_vs_eight", return_X_y, encode, verbose= verbose)

def load_kr_vs_k_zero_vs_fifteen(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/kr-vs-k-zero_vs_fifteen/kr-vs-k-zero_vs_fifteen.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-zero_vs_fifteen", return_X_y, encode, verbose= verbose)

def load_led7digit_0_2_4_5_6_7_8_9_vs_1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/led7digit-0-2-4-5-6-7-8-9_vs_1/led7digit-0-2-4-5-6-7-8-9_vs_1.dat')
    db= pd.DataFrame(data)
    db.loc[db['number'] == b'negative', 'number']= False
    db.loc[db['number'] == b'positive', 'number']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "led7digit-0-2-4-6-7-8-9_vs_1", return_X_y, encode, verbose= verbose)

def load_lymphography_normal_fibrosis(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/lymphography-normal-fibrosis/lymphography-normal-fibrosis.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "lymphography-normal-fibrosis", return_X_y, encode, verbose= verbose)

def load_page_blocks_1_3_vs_4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/page-blocks-1-3_vs_4/page-blocks-1-3_vs_4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "page-blocks-1-3_vs_4", return_X_y, encode, verbose= verbose)

def load_poker_8_9_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/poker-8-9_vs_5/poker-8-9_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "poker-8-9_vs_5", return_X_y, encode, verbose= verbose)

def load_poker_8_9_vs_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/poker-8-9_vs_6/poker-8-9_vs_6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "poker-8-9_vs_6", return_X_y, encode, verbose= verbose)

def load_poker_8_vs_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/poker-8_vs_6/poker-8_vs_6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "poker-8_vs_6", return_X_y, encode, verbose= verbose)

def load_poker_9_vs_7(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/poker-9_vs_7/poker-9_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "poker-9_vs_7", return_X_y, encode, verbose= verbose)

def load_shuttle_2_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/shuttle-2_vs_5/shuttle-2_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "shuttle-2_vs_5", return_X_y, encode, verbose= verbose)

def load_shuttle_6_vs_2_3(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/shuttle-6_vs_2-3/shuttle-6_vs_2-3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "shuttle-6_vs_2-3", return_X_y, encode, verbose= verbose)

def load_shuttle_c0_vs_c4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/shuttle-c0-vs-c4/shuttle-c0-vs-c4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "shuttle-c0-vs-c4", return_X_y, encode, verbose= verbose)

def load_shuttle_c2_vs_c4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/shuttle-c2-vs-c4/shuttle-c2-vs-c4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "shuttle-c2-vs-c4", return_X_y, encode, verbose= verbose)

def load_vowel0(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/vowel0/vowel0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vowel0", return_X_y, encode, verbose= verbose)

def load_winequality_red_3_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/winequality-red-3_vs_5/winequality-red-3_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-red-3_vs_5", return_X_y, encode, verbose= verbose)

def load_winequality_red_4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/winequality-red-4/winequality-red-4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-red-4", return_X_y, encode, verbose= verbose)

def load_winequality_red_8_vs_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/winequality-red-8_vs_6/winequality-red-8_vs_6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-red-8_vs_6", return_X_y, encode, verbose= verbose)

def load_winequality_red_8_vs_6_7(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/winequality-red-8_vs_6-7/winequality-red-8_vs_6-7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-red-8_vs_6-7", return_X_y, encode, verbose= verbose)

def load_winequality_white_3_9_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/winequality-white-3-9_vs_5/winequality-white-3-9_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-white-3-9_vs_5", return_X_y, encode, verbose= verbose)

def load_winequality_white_3_vs_7(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/winequality-white-3_vs_7/winequality-white-3_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-white-3_vs_7", return_X_y, encode, verbose= verbose)

def load_winequality_white_9_vs_4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/winequality-white-9_vs_4/winequality-white-9_vs_4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-white-9_vs_4", return_X_y, encode, verbose= verbose)

def load_yeast_0_2_5_6_vs_3_7_8_9(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/yeast-0-2-5-6_vs_3-7-8-9/yeast-0-2-5-6_vs_3-7-8-9.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-0-2-5-6_vs_3-7-8-9", return_X_y, encode, verbose= verbose)

def load_yeast_0_2_5_7_9_vs_3_6_8(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/yeast-0-2-5-7-9_vs_3-6-8/yeast-0-2-5-7-9_vs_3-6-8.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-0-2-5-7-9_vs_3-6-8", return_X_y, encode, verbose= verbose)

def load_yeast_0_3_5_9_vs_7_8(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/yeast-0-3-5-9_vs_7-8/yeast-0-3-5-9_vs_7-8.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-0-3-5-9_vs_7-8", return_X_y, encode, verbose= verbose)

def load_yeast_0_5_6_7_9_vs_4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/yeast-0-5-6-7-9_vs_4/yeast-0-5-6-7-9_vs_4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-0-5-6-7-9_vs_4", return_X_y, encode, verbose= verbose)

def load_yeast_1_2_8_9_vs_7(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/yeast-1-2-8-9_vs_7/yeast-1-2-8-9_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-1-2-8-9_vs_7", return_X_y, encode, verbose= verbose)

def load_yeast_1_4_5_8_vs_7(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/yeast-1-4-5-8_vs_7/yeast-1-4-5-8_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-1-4-5-8_vs_7", return_X_y, encode, verbose= verbose)

def load_yeast_1_vs_7(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/yeast-1_vs_7/yeast-1_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-1_vs_7", return_X_y, encode, verbose= verbose)

def load_yeast_2_vs_4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/yeast-2_vs_4/yeast-2_vs_4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-2_vs_4", return_X_y, encode, verbose= verbose)

def load_yeast_2_vs_8(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/yeast-2_vs_8/yeast-2_vs_8.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-2_vs_8", return_X_y, encode, verbose= verbose)

def load_yeast4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/yeast4/yeast4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast4", return_X_y, encode, verbose= verbose)

def load_yeast5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/yeast5/yeast5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast5", return_X_y, encode, verbose= verbose)

def load_yeast6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/yeast6/yeast6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast6", return_X_y, encode, verbose= verbose)

def load_zoo_3(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/zoo-3/zoo-3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "zoo-3", return_X_y, encode, verbose= verbose)

#########################

def load_ecoli_0_vs_1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli-0_vs_1/ecoli-0_vs_1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'class']= False
    db.loc[db['Class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0_vs_1", return_X_y, encode, verbose= verbose)

def load_ecoli1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli1/ecoli1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli1", return_X_y, encode, verbose= verbose)

def load_ecoli2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli2/ecoli2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli2", return_X_y, encode, verbose= verbose)

def load_ecoli3(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/ecoli3/ecoli3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli3", return_X_y, encode, verbose= verbose)

def load_glass_0_1_2_3_vs_4_5_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/glass-0-1-2-3_vs_4-5-6/glass-0-1-2-3_vs_4-5-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-2-3_vs_4-5-6", return_X_y, encode, verbose= verbose)

def load_glass0(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/glass0/glass0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass0", return_X_y, encode, verbose= verbose)

def load_glass1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/glass1/glass1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass1", return_X_y, encode, verbose= verbose)

def load_glass6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/glass6/glass6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass6", return_X_y, encode, verbose= verbose)

def load_haberman(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/haberman/haberman.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "habarman", return_X_y, encode, verbose= verbose)

def load_iris0(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/iris0/iris0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "iris0", return_X_y, encode, verbose= verbose)

def load_new_thyroid1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/new_thyroid1/new-thyroid1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "new_thyroid1", return_X_y, encode, verbose= verbose)

def load_new_thyroid2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/new_thyroid2/new_thyroid2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "new_thyroid2", return_X_y, encode, verbose= verbose)

def load_page_blocks0(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/page-blocks0/page-blocks0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "page_blocks0", return_X_y, encode, verbose= verbose)

def load_pima(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/pima/pima.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "pima", return_X_y, encode, verbose= verbose)

def load_segment0(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/segment0/segment0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "segment0", return_X_y, encode, verbose= verbose)

def load_vehicle0(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/vehicle0/vehicle0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vehicle0", return_X_y, encode, verbose= verbose)

def load_vehicle1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/vehicle1/vehicle1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vehicle1", return_X_y, encode, verbose= verbose)

def load_vehicle2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/vehicle2/vehicle2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vehicle2", return_X_y, encode, verbose= verbose)

def load_vehicle3(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/vehicle3/vehicle3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vehicle3", return_X_y, encode, verbose= verbose)

def load_wisconsin(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/wisconsin/wisconsin.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "wisconsin", return_X_y, encode, verbose= verbose)

def load_yeast1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/yeast1/yeast1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast1", return_X_y, encode, verbose= verbose)

def load_yeast3(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/yeast3/yeast3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast3", return_X_y, encode, verbose= verbose)

def generate_artificial_data(dim, n, imbalanced_ratio):
    np.random.seed(2)
    
    center_majority= np.repeat(0.0, dim)
    center_minority= np.repeat(1.0, dim)
    stdev= np.linalg.norm(center_minority)/3.0
    n_maj= int(n*imbalanced_ratio/(imbalanced_ratio - 1.0))
    n_min= n - n_maj
    
    X_maj= center_majority + np.random.normal(scale= stdev, size= (n_maj, dim))
    X_min= center_minority + np.random.normal(scale= stdev, size= (n_min, dim))
    
    return np.vstack([X_maj, X_min]), np.hstack([np.repeat(0, n_min), np.repeat(1, n_maj)])

def load_artificial_data(dim, n, imbalanced_ratio, return_X_y= False, verbose= False):
    X, y= generate_artificial_data(dim, n, imbalanced_ratio)
    db= pd.DataFrame(X)
    db['target']= y
    return construct_return_set(db, "artificial_dim_%d_n_%d_ir_%f" % (dim, n, imbalanced_ratio), return_X_y, encode= False, verbose= verbose)

def summary():
    results= []
    # fixing the globals dictionary keys
    d= list(globals().keys())
    for func_name in d:
        if func_name.startswith('load_') and not func_name.startswith('load_artificial'):
            data_not_encoded= globals()[func_name](return_X_y= False, encode= False)
            data_encoded= globals()[func_name](return_X_y= False, encode= True)
            
            results.append({'loader_function': globals()[func_name],
                            'name': data_not_encoded['DESCR'],
                            'len': len(data_not_encoded['data']),
                            'non_encoded_n_attr': len(data_not_encoded['data'][0]),
                            'encoded_n_attr': len(data_encoded['data'][0]),
                            'imbalanced_ratio': np.sum(data_encoded['target'] == 0)/np.sum(data_encoded['target'] == 1)})
    
    df_results= pd.DataFrame(results)
    print(df_results)

    return df_results

def get_all_data_loaders():
    results= []
    # fixing the globals dictionary keys
    d= list(globals().keys())
    results= [globals()[f] for f in d if f.startswith('load_') and not f.startswith('load_artificial')]
        
    return results

def get_filtered_data_loaders(num_features_lower_bound= 1,
                              num_features_upper_bound= 1e10,
                              len_lower_bound= 2,
                              len_upper_bound= 1e10,
                              imbalanced_ratio_lower_bound= 0,
                              imbalanced_ratio_upper_bound= 1e10):
    descriptors= summary()
    return descriptors[(descriptors['len'] >= len_lower_bound) & 
                       (descriptors['len'] < len_upper_bound) & 
                       (descriptors['encoded_n_attr'] >= num_features_lower_bound) & 
                       (descriptors['encoded_n_attr'] < num_features_upper_bound) & 
                       (descriptors['imbalanced_ratio'] >= imbalanced_ratio_lower_bound) & 
                       (descriptors['imbalanced_ratio'] < imbalanced_ratio_upper_bound)]['loader_function'].values
