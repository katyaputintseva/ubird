
import numpy as np
import pandas as pd 


def	categorize_location(df):

	'''
	Returns a DataFrame with the aminoacid location as columns and the data as categories
	of 0 and 1 
	

	'''

	mutants = df['aaMutations']       #split to list of aminoacids
	mutants = mutants.str.split(':')
	mutants = mutants[1:]
	
	mut_location = []
	for i in xrange(1,len(mutants)):
		mut_location.append(pd.Series(mutants[i]).str.extract('(^[A-Z]{2}\d+)'))
		
	df_mut_location = pd.DataFrame(mut_location)
	
	uniq_mut_los=np.unique(df_mut_location.values.ravel()) # unique values of locations

	uniq_mut_los = uniq_mut_los[1:] #droping the NaN entry 
	
	cols = uniq_mut_los.tolist()  #unique values of locations -> new columns


	categ_mat = np.zeros((len (df_mut_location), len (cols)))

	for i in xrange(len(df_mut_location)):
		for j in xrange(len(cols)):
			if cols[j] in df_mut_location.values[i]:
				categ_mat[i,j]=1
			else:
				categ_mat[i,j]=0
				
    	#all(boleans_test_hist == categ_mat.sum(axis=1)) #out: TRUE!

    #########################
	# Now build the DataFrame
    #########################
	locations_df = pd.DataFrame(categ_mat, columns=cols)

	brightness_df = df.ix[1:]
	brightness_df = brightness_df.reset_index()
	brightness_df = brightness_df.drop(['index'], axis=1)


	locations_df = pd.concat((locations_df,brightness_df),axis=1)
	locations_df = locations_df.drop(['aaMutations'], axis = 1)
    
	return locations_df






