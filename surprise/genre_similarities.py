import numpy as np
import pandas as pd

def compute_f_matrix(user_based, trainset, genre_file):
    """
    Compute (n_x)-by-(n_g) matrix of relative frequencies in pd.Data.Frame, where
      n_x is the size of users if user_based is True;
      n_x is the size of items if user_based is not True.
      n_g is the number of genres.
    The input genre_file is a csv file that consists of
      genre.columns[0] is the items column if user_based is True; users column if user_based is False.
      genre.columns[1] is an unused column.
      genre.columns[2:] are the genre levels.
    """
    if user_based:
        n_x = trainset.n_users
    else:
        n_x = trainset.n_items
    
    # Build genre and ratings in pandas data frame
    genre = pd.read_csv(genre_file) # item-by-genre
    genre[genre.columns[0]] = genre[genre.columns[0]].apply(str)
    if user_based:
        ratings = pd.DataFrame(trainset.build_testset(), columns=['x','y','r'])
    else:
        ratings = pd.DataFrame(trainset.build_testset(), columns=['y','x','r'])
    
    # Merge ratings and genre
    ratings_genre = pd.merge(ratings, genre, how='left', left_on='y', right_on=genre.columns[0])
    ratings_genre = ratings_genre.drop(['y','r',genre.columns[0],genre.columns[1]], axis=1)
    
    # Compute relative genre frequency matrix
    z = ratings_genre.groupby('x').sum()
    denom = z.sum(axis=1)
    f = z.div(denom, axis=0)  ## Check div by zero

    # Sort rows of f by x's inner id
    if user_based:
        inner_id_f = [trainset.to_inner_uid(x) for x in f.index]
    else:
        inner_id_f = [trainset.to_inner_iid(x) for x in f.index]
    f = f.set_index([inner_id_f])
    f.sort_index(inplace=True, ascending=True)

    return n_x, f


def squared_deviance(n_x, f):
    """
    Compute genre similarity matrix based on squared deviance
    """
    fa = np.array(f)
    n_g = fa.shape[1]
    num = np.zeros((n_x, n_x))
    den = np.zeros((n_x,n_x))
    for k in range(0, n_g):
        ff_max = np.maximum.outer(fa[:, k], fa[:, k])
        ff_sub = np.subtract.outer(fa[:, k], fa[:, k])
        num += ff_max * ff_sub * ff_sub
        den += ff_max

    return 1 - np.divide(num, den, out=np.ones_like(num), where=den!=0)


def absolute_deviance(n_x, f):
    """
    Compute genre similarity matrix based on absolute deviance
    """
    fa = np.array(f)
    n_g = fa.shape[1]
    num = np.zeros((n_x, n_x))
    den = np.zeros((n_x, n_x))
    for k in range(0, n_g):
        ff_max = np.maximum.outer(fa[:, k], fa[:, k])
        ff_sub = np.subtract.outer(fa[:, k], fa[:, k])
        num += ff_max * np.abs(ff_sub)
        den += ff_max

    return 1 - np.divide(num, den, out=np.ones_like(num), where=den!=0)
