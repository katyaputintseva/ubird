import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import sparse 

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from time import time

def MachineLearningPipeline(filename):
    t0 = time()
    # reading the raw date
    ## print "Reading the data..."
    t1 = time()
    df = filename

    # Extracting brightness values and storing them separately in file
    y = df[["brightness"]].values.ravel()
    std = df[["std"]].values

    threshold = 0.95 #threshold for target values

    y_bin = np.where(y >= y[0]*threshold,1,0) # binary respresentation of target values based on threshold
    
    class_weights = (y_bin.sum()*1./(y_bin.shape[0]-y_bin.sum())*1.,1.) # class weights based on relative frequencies

    #extract the column with aa mutations and split the mutations
    mutants_series = df.mutations.str.split(':')
    sample_weights = df.barcodes.values #number of unique barcodes for each mutant
    sample_weights = np.log10(sample_weights*10)
    sample_weights = (sample_weights/sample_weights.max()).reshape(-1)

    del df
    ##print "Time taken: %.2f s\n"%(time()-t1)
    
    ##print "Creating the data matrix..."
    t1 = time()
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
    ##print "Time taken: %.2f s\n"%(time()-t1)


    # Retrieve the indices of all the mutants with single mutations 
    # and store the id of correspoding single mutations                       
    single_mutants_idx = np.asarray((np.sum(X,axis = 1) == 1)).ravel()
    single_mutation_idx = np.asarray(np.sum(X[single_mutants_idx,:],axis=0) == 1).ravel()


    # train the classifer using full data set
    ##print "Training the classifier..."
    t1 = time()
    clf = RandomForestClassifier(n_estimators=30,min_samples_split=50,max_features=10,class_weight={0:class_weights[0],1:class_weights[1]})
    clf.fit(X,y_bin,sample_weight = sample_weights)
    ##print "Time taken: %.2f s\n"%(time()-t1)
    
    #print metrics.accuracy_score(y_bin,clf.predict(X),sample_weight = sample_weights)

    # predict the category for all possible single mutations available in the dataset
    ##print "Predicting single mutations..."
    t1 = time()
    S = sparse.diags(np.ones(X.shape[1]))
    ys = clf.predict(S)
    hs = clf.predict_proba(S)
    del S
    pos_idx = (hs[:,1]>0.9) & ~single_mutation_idx
    neg_idx = (hs[:,0]>0.9) & ~single_mutation_idx
    
    pos_mutations = zip([col_names[i] for i in xrange(pos_idx.shape[0]) if pos_idx[i]],hs[pos_idx][:,1],np.sum(X[:,pos_idx],axis = 0).tolist()[0])
    neg_mutations = zip([col_names[i] for i in xrange(neg_idx.shape[0]) if neg_idx[i]],hs[neg_idx][:,0],np.sum(X[:,neg_idx],axis = 0).tolist()[0])
    ##print "Time taken: %.2f s\n"%(time()-t1)
    
    # pos_mutaions and neg_mutations are list of tuples. Each tuple contains 4 values:
    #   - mutation name
    #   - prediction confidence of classifier
    #   - occurence of mutation in the dataset
        
    # sort the positive and negative single mutations
    ##print "Sorting the mutation predictions..."
    t1 = time()
    pos_mutations.sort(key=lambda x:x[2],reverse = True)
    neg_mutations.sort(key=lambda x:x[2],reverse = True)
    ##print "Time taken: %.2f\n"%(time()-t1)
    
    # Create a list of feature(mutation) importances according to classifier
    ##print "Extracting feature importances from classifier..."  
    t1 = time()
    order = clf.feature_importances_.argsort()[::-1]
    mutation_importances = zip([col_names[i] for i in order],clf.feature_importances_[order])
    del order
    ##print "Time taken: %.2f\n"%(time()-t1)
    
    ##print "Overall time taken: %.2f s\n"%(time()-t0)  
    #print pos_mutations        
             
    return pos_mutations #,neg_mutations,mutation_importances

#MachineLearningPipeline('amino_acid_genotypes_to_brightness.tsv')
