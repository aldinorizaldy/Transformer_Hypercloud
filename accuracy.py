import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

# lwir ascn
dir_labels = 'predict'
labels_path = 'test_lwir_degr_ascn_pct_100epochs'
labels = np.loadtxt(os.path.join(dir_labels,labels_path,'prediction_all.txt'))

true = labels[:,-2]
pred = labels[:,-1]
# print(accuracy_score(true,pred))
OA = accuracy_score(true,pred)
matrix = confusion_matrix(true, pred)
# print(matrix.diagonal()/matrix.sum(axis=1))
# print(np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true)))
AA = np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true))
# print(precision_recall_fscore_support(true, pred, average='weighted'))
# print(f1_score(true, pred, average='weighted'))
P, R, F1, _ = precision_recall_fscore_support(true, pred, average='weighted')
acc = np.array([OA, AA, P, R, F1])
np.savetxt(os.path.join(dir_labels,labels_path,'accuracy_all.txt'),acc)

# lwir naive
dir_labels = 'predict'
labels_path = 'test_lwir_degr_naive_pct_100epochs'
labels = np.loadtxt(os.path.join(dir_labels,labels_path,'prediction_all.txt'))

true = labels[:,-2]
pred = labels[:,-1]
# print(accuracy_score(true,pred))
OA = accuracy_score(true,pred)
matrix = confusion_matrix(true, pred)
# print(matrix.diagonal()/matrix.sum(axis=1))
# print(np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true)))
AA = np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true))
# print(precision_recall_fscore_support(true, pred, average='weighted'))
# print(f1_score(true, pred, average='weighted'))
P, R, F1, _ = precision_recall_fscore_support(true, pred, average='weighted')
acc = np.array([OA, AA, P, R, F1])
np.savetxt(os.path.join(dir_labels,labels_path,'accuracy_all.txt'),acc)

# lwir pt
dir_labels = 'predict'
labels_path = 'test_lwir_degr_pt_pct_100epochs'
labels = np.loadtxt(os.path.join(dir_labels,labels_path,'prediction_all.txt'))

true = labels[:,-2]
pred = labels[:,-1]
# print(accuracy_score(true,pred))
OA = accuracy_score(true,pred)
matrix = confusion_matrix(true, pred)
# print(matrix.diagonal()/matrix.sum(axis=1))
# print(np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true)))
AA = np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true))
# print(precision_recall_fscore_support(true, pred, average='weighted'))
# print(f1_score(true, pred, average='weighted'))
P, R, F1, _ = precision_recall_fscore_support(true, pred, average='weighted')
acc = np.array([OA, AA, P, R, F1])
np.savetxt(os.path.join(dir_labels,labels_path,'accuracy_all.txt'),acc)

# lwir spct
dir_labels = 'predict'
labels_path = 'test_lwir_degr_spct_100epochs'
labels = np.loadtxt(os.path.join(dir_labels,labels_path,'prediction_all.txt'))

true = labels[:,-2]
pred = labels[:,-1]
# print(accuracy_score(true,pred))
OA = accuracy_score(true,pred)
matrix = confusion_matrix(true, pred)
# print(matrix.diagonal()/matrix.sum(axis=1))
# print(np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true)))
AA = np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true))
# print(precision_recall_fscore_support(true, pred, average='weighted'))
# print(f1_score(true, pred, average='weighted'))
P, R, F1, _ = precision_recall_fscore_support(true, pred, average='weighted')
acc = np.array([OA, AA, P, R, F1])
np.savetxt(os.path.join(dir_labels,labels_path,'accuracy_all.txt'),acc)

# swir ascn
dir_labels = 'predict'
labels_path = 'test_swir_degr_ascn_pct_100epochs'
labels = np.loadtxt(os.path.join(dir_labels,labels_path,'prediction_all.txt'))

true = labels[:,-2]
pred = labels[:,-1]
# print(accuracy_score(true,pred))
OA = accuracy_score(true,pred)
matrix = confusion_matrix(true, pred)
# print(matrix.diagonal()/matrix.sum(axis=1))
# print(np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true)))
AA = np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true))
# print(precision_recall_fscore_support(true, pred, average='weighted'))
# print(f1_score(true, pred, average='weighted'))
P, R, F1, _ = precision_recall_fscore_support(true, pred, average='weighted')
acc = np.array([OA, AA, P, R, F1])
np.savetxt(os.path.join(dir_labels,labels_path,'accuracy_all.txt'),acc)

# swir naive
dir_labels = 'predict'
labels_path = 'test_swir_degr_naive_pct_100epochs'
labels = np.loadtxt(os.path.join(dir_labels,labels_path,'prediction_all.txt'))

true = labels[:,-2]
pred = labels[:,-1]
# print(accuracy_score(true,pred))
OA = accuracy_score(true,pred)
matrix = confusion_matrix(true, pred)
# print(matrix.diagonal()/matrix.sum(axis=1))
# print(np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true)))
AA = np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true))
# print(precision_recall_fscore_support(true, pred, average='weighted'))
# print(f1_score(true, pred, average='weighted'))
P, R, F1, _ = precision_recall_fscore_support(true, pred, average='weighted')
acc = np.array([OA, AA, P, R, F1])
np.savetxt(os.path.join(dir_labels,labels_path,'accuracy_all.txt'),acc)

# swir pt
dir_labels = 'predict'
labels_path = 'test_swir_degr_pt_pct_100epochs'
labels = np.loadtxt(os.path.join(dir_labels,labels_path,'prediction_all.txt'))

true = labels[:,-2]
pred = labels[:,-1]
# print(accuracy_score(true,pred))
OA = accuracy_score(true,pred)
matrix = confusion_matrix(true, pred)
# print(matrix.diagonal()/matrix.sum(axis=1))
# print(np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true)))
AA = np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true))
# print(precision_recall_fscore_support(true, pred, average='weighted'))
# print(f1_score(true, pred, average='weighted'))
P, R, F1, _ = precision_recall_fscore_support(true, pred, average='weighted')
acc = np.array([OA, AA, P, R, F1])
np.savetxt(os.path.join(dir_labels,labels_path,'accuracy_all.txt'),acc)

# swir spct
dir_labels = 'predict'
labels_path = 'test_swir_degr_spct_100epochs'
labels = np.loadtxt(os.path.join(dir_labels,labels_path,'prediction_all.txt'))

true = labels[:,-2]
pred = labels[:,-1]
# print(accuracy_score(true,pred))
OA = accuracy_score(true,pred)
matrix = confusion_matrix(true, pred)
# print(matrix.diagonal()/matrix.sum(axis=1))
# print(np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true)))
AA = np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true))
# print(precision_recall_fscore_support(true, pred, average='weighted'))
# print(f1_score(true, pred, average='weighted'))
P, R, F1, _ = precision_recall_fscore_support(true, pred, average='weighted')
acc = np.array([OA, AA, P, R, F1])
np.savetxt(os.path.join(dir_labels,labels_path,'accuracy_all.txt'),acc)

# vnir ascn
dir_labels = 'predict'
labels_path = 'test_vnir_degr_ascn_pct_100epochs'
labels = np.loadtxt(os.path.join(dir_labels,labels_path,'prediction_all.txt'))

true = labels[:,-2]
pred = labels[:,-1]
# print(accuracy_score(true,pred))
OA = accuracy_score(true,pred)
matrix = confusion_matrix(true, pred)
# print(matrix.diagonal()/matrix.sum(axis=1))
# print(np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true)))
AA = np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true))
# print(precision_recall_fscore_support(true, pred, average='weighted'))
# print(f1_score(true, pred, average='weighted'))
P, R, F1, _ = precision_recall_fscore_support(true, pred, average='weighted')
acc = np.array([OA, AA, P, R, F1])
np.savetxt(os.path.join(dir_labels,labels_path,'accuracy_all.txt'),acc)

# vnir naive
dir_labels = 'predict'
labels_path = 'test_vnir_degr_naive_pct_100epochs'
labels = np.loadtxt(os.path.join(dir_labels,labels_path,'prediction_all.txt'))

true = labels[:,-2]
pred = labels[:,-1]
# print(accuracy_score(true,pred))
OA = accuracy_score(true,pred)
matrix = confusion_matrix(true, pred)
# print(matrix.diagonal()/matrix.sum(axis=1))
# print(np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true)))
AA = np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true))
# print(precision_recall_fscore_support(true, pred, average='weighted'))
# print(f1_score(true, pred, average='weighted'))
P, R, F1, _ = precision_recall_fscore_support(true, pred, average='weighted')
acc = np.array([OA, AA, P, R, F1])
np.savetxt(os.path.join(dir_labels,labels_path,'accuracy_all.txt'),acc)

# vnir pt
dir_labels = 'predict'
labels_path = 'test_vnir_degr_pt_pct_100epochs'
labels = np.loadtxt(os.path.join(dir_labels,labels_path,'prediction_all.txt'))

true = labels[:,-2]
pred = labels[:,-1]
# print(accuracy_score(true,pred))
OA = accuracy_score(true,pred)
matrix = confusion_matrix(true, pred)
# print(matrix.diagonal()/matrix.sum(axis=1))
# print(np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true)))
AA = np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true))
# print(precision_recall_fscore_support(true, pred, average='weighted'))
# print(f1_score(true, pred, average='weighted'))
P, R, F1, _ = precision_recall_fscore_support(true, pred, average='weighted')
acc = np.array([OA, AA, P, R, F1])
np.savetxt(os.path.join(dir_labels,labels_path,'accuracy_all.txt'),acc)

# vnir spct
dir_labels = 'predict'
labels_path = 'test_vnir_degr_spct_100epochs'
labels = np.loadtxt(os.path.join(dir_labels,labels_path,'prediction_all.txt'))

true = labels[:,-2]
pred = labels[:,-1]
# print(accuracy_score(true,pred))
OA = accuracy_score(true,pred)
matrix = confusion_matrix(true, pred)
# print(matrix.diagonal()/matrix.sum(axis=1))
# print(np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true)))
AA = np.sum(matrix.diagonal()/matrix.sum(axis=1))/len(np.unique(true))
# print(precision_recall_fscore_support(true, pred, average='weighted'))
# print(f1_score(true, pred, average='weighted'))
P, R, F1, _ = precision_recall_fscore_support(true, pred, average='weighted')
acc = np.array([OA, AA, P, R, F1])
np.savetxt(os.path.join(dir_labels,labels_path,'accuracy_all.txt'),acc)
