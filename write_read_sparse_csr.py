# function to write and read csr matrices to and from files    
def save_sparse_csr_to_csv(filename,array,col_names):
    np.savetxt(filename+"_indices.csv",categ_mat_csr.indices,delimiter=",",fmt="%d")
    np.savetxt(filename+"_indptr.csv",categ_mat_csr.indptr,delimiter=",",fmt="%d")
    np.savetxt(filename+"_col_names.csv",col_names,delimiter=",",fmt="%s")
    np.savetxt(filename+"_shape.csv",array.shape,delimiter=",",fmt="%d")

# returns the csr matrix of samples and list of colnames
def load_sparse_csr_from_csv(filename):
    indices = np.loadtxt(filename+"_indices.csv",unpack=True,delimiter=",")
    indptr = np.loadtxt(filename+"_indptr.csv",unpack=True,delimiter=",")
    shape = np.loadtxt(filename+"_shape.csv",unpack=True,delimiter=",")
    col_names = np.loadtxt(filename+"_col_names.csv",unpack=True,delimiter=",",dtype=str)
    data = np.ones_like(indices,dtype=np.int8)
    
    return sparse.csr_matrix((data, indices, indptr),
                         shape = shape),col_names