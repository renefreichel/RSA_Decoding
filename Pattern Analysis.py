#!/usr/bin/env python
# coding: utf-8

# # Pattern Analysis: RSA vs. Decoding

# The overarching goal of this study is to examine how RSA performance relates to the decoding classification in the context of face gender classification. Based on the theoretical background, we hypothesize that both the RSA and decoding methods are able to accurately distinguish between face gender. H1: The decoding classifier shows a significant above chance accuracy in the classification of face gender. H2: The RSA will show a high correlation between neural patterns and events. Based upon an accurate classification in decoding and a high correlation between brain and event patterns, we expect a high correlation between RSA performance and decoding accuracy. H3: RSA performance is strongly correlated to decoding accuracy. Importantly, we expect high performance of both the RSA and decoding. However, if both RSA and decoding yield chance - level or unsatisfying results, we would expect no correlation between RSA performance and decoding accuracy as both methods fail to classify face gender. The findings of this project will complement the literature with empirical insight that complement the existing conceptual differences between RSA and decoding. Given the importance of machine learning and artificial intelligence, our research could result in findings that could inform model selection (RSA/ decoding) for the proposed feature of classifying face gender.
# 

# ## Pattern estimation and pre-processing

# In[ ]:


import os
import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob
from niedu.global_utils import get_data_dir
from niedu.utils.nipa import filter_pattern_drift
from nilearn import image, datasets, plotting, masking
from sklearn.preprocessing import LabelEncoder, StandardScaler, Robu from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, mean_squared_error, pairw from sklearn.linear_model import RidgeClassifier, LogisticRegression from sklearn.model_selection import LeavePGroupsOut
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, spearmanr
import matplotlib.pyplot as plt


# setting the path directory for loading in the data data_dir = get_data_dir()
subs = sorted([os.path.basename(d) for d in glob(os.path.join(data_d sessions = ['ses-1', 'ses-2']
runs = ['run-1', 'run-2', 'run-3', 'run-4', 'run-5', 'run-6'] pattern_dir = os.path.join(data_dir, 'derivatives', 'nibetaseries_l', events_dir = data_dir) # path for the .events files, where the event
                                                                                         

def get_R(pattern_directory, subject, MNI_T1, region_of_interest): 
""" Function that loads in the pattern data and returns the neural m Parameters are the pattern directory, the subject you want to ob and the region of interest as a string. """
pattern_files = []
for i, session in enumerate(sessions):
pattern_files.append(sorted(glob(os.path.join(pattern_direct
R_4D = image.concat_imgs(pattern_files)
ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr ho_map = ho_atlas['maps']
r_rTOF_idx = ho_atlas['labels'].index(region_of_interest)
ho_map_resamp = image.resample_to_img(ho_map, R_4D, interpolatio r_rTOF_roi_bool = ho_map_resamp.get_fdata() == r_rTOF_idx r_rTOF_roi = nib.Nifti1Image(r_rTOF_roi_bool.astype(int), affine
return masking.apply_mask(R_4D, r_rTOF_roi)

def get_S(events_directory, subject, separator, _filter): """
Function that loads in the event data and returns the feature ma Parameters are the directory where the event data is stored, the (.tsv = '/t', .csv = ',')
"""
events_file = []
for i, session in enumerate(sessions):
events_file = np.concatenate((events_file, sorted(glob(os.pa
lab_enc = LabelEncoder()
events = [pd.read_csv(file, sep=separator) for file in events_fi events_filt = [event.loc[event['trial_type'].str.contains('STIM' S_filt = [e[_filter].to_numpy() for e in events_filt]
S_expr = [lab_enc.fit_transform(e) for e in S_filt]
return np.concatenate(S_expr)
                                                                                          
                                                                                         


# ## Representational Similarity Analysis (RSA)

# In[ ]:


def rsa(R, R_metric, S, S_metric, plot = False): """
Function that performs the Representational Similarity Analysis. Parameters are the R matrix, the R_metric is the distance of int (.tsv = '/t', .csv = ',')
"""
rdm_R = pairwise_distances(R, metric = R_metric)
rdv_R = squareform(np.round(rdm_R, 5), force = 'tovector') # onl rdm_S = pairwise_distances(S[:, np.newaxis], metric = S_metric) rdv_S = squareform(rdm_S)
RDV_all.append(rdv_R)
if plot:
fig, (rdmS, rdmR) = plt.subplots(1, 2) fig.suptitle('Feature and Neural RDM of Face Gender') rdmS.imshow(rdm_S)
rdmR.imshow(rdm_R)
rdmS.set_title("Feature RDM")
rdmR.set_title('Neural RDM')
plt.show()
return kendalltau(rdv_S, rdv_R)


# ## Decoding Pipeline

# In[ ]:


def decoder(R, S, scaler, classifier, numberoffolds): """
            Function that performs decoding based on the classifier that is
            Parameters are the R matrix, the S matrix, the scaler of interes
"""
pipe = make_pipeline(scaler, classifier)
groups = np.concatenate([[i] * 40 for i in range(12)]) lpgo = LeavePGroupsOut(n_groups = numberoffolds)
folds = lpgo.split(R,S,groups) acc, mse = [], []
for train_idx, test_idx in folds:
R_train, R_test = R[train_idx,:], R[test_idx,:] S_train, S_test = S[train_idx], S[test_idx] pipe.fit(R_train, S_train)
preds = pipe.predict(R_test) acc.append(roc_auc_score(S_test, preds)) mse.append(mean_squared_error(S_test, preds))
return np.mean(acc), np.mean(mse)


# Pre-allocate the accuracy, mean squared error, correlation and p-v av_acc_all = np.zeros(len(subs))
av_mse_all = np.zeros(len(subs)) corr_all = np.zeros(len(subs)) pval_all = np.zeros(len(subs)) RDV_all = []
for i, sub in enumerate(subs):
print('Start processing of {}...'.format(sub))
R = get_R(pattern_dir, sub, 'MNI', 'Right Temporal Occipital Fus print('R of {} done...'.format(sub))
S = get_S(events_dir, sub, '\t', 'face_sex')
print('S of {} done...'.format(sub))
corr_all[i], pval_all[i] = rsa(R, 'euclidean', S, 'manhattan') print('RSA of {} done...'.format(sub))
av_acc_all[i], av_mse_all[i] = decoder(R, S, StandardScaler(), print('Decoding of {} done...'.format(sub))
print("Accuracy : {}".format(av_acc_all)) print("Correlation : {}".format(corr_all))


# ## Model Visualization

# In[ ]:


print("The overall Spearman correlation is", round(spearmanr(av_acc_


# In[ ]:


av_acc_all = np.zeros(len(subs)) av_mse_all = np.zeros(len(subs)) corr_all = np.zeros(len(subs)) pval_all = np.zeros(len(subs)) RDV_all = []
for i, sub in enumerate(subs):
print('Start processing of {}...'.format(sub))
R = get_R(pattern_dir, sub, 'MNI', 'Right Temporal Occipital Fus print('R of {} done...'.format(sub))
S = get_S(events_dir, sub, '\t', 'face_sex')
print('S of {} done...'.format(sub))
corr_all[i], pval_all[i] = rsa(R, 'euclidean', S, 'manhattan', print('RSA of {} done...'.format(sub))
av_acc_all[i], av_mse_all[i] = decoder(R, S, StandardScaler(), print('Decoding of {} done...'.format(sub))
                                       
                                       
     

