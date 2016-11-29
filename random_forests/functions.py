import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import sparse 

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,NuSVC,SVR,NuSVR
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import neighbors
from sklearn.decomposition import TruncatedSVD

def MachineLearningPipeline(filename):
    # reading the raw date
    df = pd.read_csv(filename, sep = '\t', engine = 'python')

    # Extracting brightness values and storing them separately in file
    y = df[["medianBrightness"]].values.ravel()
    std = df[["std"]].values

    threshold = 0.95 #threshold for target values

    y_bin = np.where(y >= y[0]*threshold,1,0) # binary respresentation of target values based on threshold

    #extract the column with aa mutations and split the mutations
    mutants_series = df.aaMutations.str.split(':')
    counts = df.uniqueBarcodes.values #number of unique barcodes for each mutant

    del df

    # create a list of mutants and lookup table of all unique mutations
    mutants = mutants_series.tolist()
    mutants[0] = [] #mutant 0 is the base case and has no mutations 

    m = len(mutants) # number of mutants

    mutations = {}
    col_names = []

    m_count = 0
    for mutant in xrange(m):
        for mutation in mutants[mutant]:
            if mutation not in mutations:
                mutations[mutation] = {"id":m_count,"mutants":[mutant]}
                col_names.append(mutation)
                m_count += 1
            else:
                mutations[mutation]["mutants"].append(mutant)

    n = len(mutations.keys()) # number of unique mutations

    #creating the feature matrix in LIL format and converting it to CSR format
    categ_mat_lil = sparse.lil_matrix((m,n),dtype = np.int8)

    for i in xrange(1,m):
        cols,data = zip(*[(mutations[mutation]["id"],1) for mutation in mutants[i]])
        categ_mat_lil.rows[i],categ_mat_lil.data[i] = list(cols),list(data)
    
    X = categ_mat_lil.tocsr()
    del categ_mat_lil


    # Retrieve all the mutants with single mutations 
    # and create the look up table for this single mutations
    single_mutations = {}
    single_mut_idx = []

    for i in xrange(X.shape[0]):
        if X.getrow(i).indices.shape[0] == 1:
            index = X.getrow(i).indices[0]
            mutation = col_names[index]
            #print mutation,i
            if mutation not in single_mutations:
                single_mutations[mutation] = {"id":index,"mutants":[i]}
                single_mut_idx.append(i)
            else:
                single_mutations[mutation]["mutants"].append(i)


    # train the classifer using full data set
    clf = RandomForestClassifier(min_samples_split=20,n_estimators=30)
    clf.fit(X,y_bin)

    # predict the category for all possible single mutations available in the dataset
    pos_mutations = []
    neg_mutations = []
    count = 0
    for i in clf.feature_importances_.argsort()[::-1]:
        if col_names[i] not in single_mutations:
            occurence = X[:,i].sum()
            #med_brigtness = np.median(y[X[:,i]>0])
            mutation_vector = np.zeros((1,X.shape[1]))
            mutation_vector[0,i] = 1
            h = clf.predict_proba(mutation_vector)
            if h[0][1] > 0.9:
                pos_mutations.append((col_names[i],h[0][1],occurence,count))

            if h[0][0] > 0.9:
                neg_mutations.append((col_names[i],h[0][0],occurence,count))
        
        count += 1
    
    # pos_mutaions and neg_mutations are list of tuples. Each tuple contains 4 values:
    #   - mutation name
    #   - prediction confidence of classifier
    #   - occurence of mutation in the dataset
    #   - importance of mutation as feature for the classifier
        
    # sort the positive and negative single mutations
    pos_mutations.sort(key=lambda x:x[2],reverse = True)
    neg_mutations.sort(key=lambda x:x[2],reverse = True)
    
    # Create a list of feature(mutation) importances according to classifier  
    mutation_importances = [(col_names[i],clf.feature_importances_[i]) \
                            for i in clf.feature_importances_.argsort()[::-1]]
    
                            
    return pos_mutations,neg_mutations,mutation_importances

MachineLearningPipeline()
