import torch 

def nul_matrix_except_one_column_of_ones(shape,index_col):
     m = torch.zeros(shape)
     m[:,index_col]=1
     return m

def normalize(x, x_min,x_max):
     return (x - x_min)/(x_max-x_min)

def denormalize(x, x_min,x_max):
     return x_min + (x_max - x_min)*x
