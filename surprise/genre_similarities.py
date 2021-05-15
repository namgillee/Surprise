import numpy as np
import pandas as pd
import rdkit.Chem as Chem

def compute_f_matrix(user_based, trainset, genre_file):
    """
    Compute (n_x)-by-(n_g) matrix of relative frequencies in pd.Data.Frame, where
    n_x is the size of users if user_based is True;
    n_x is the size of items if user_based is not True.
    n_g is the number of genres.

    :param user_based: True to indicate user-based collaborative filtering method.
    :param trainset: a Trainset class instance.
    :param genre_file: a csv file that consists of
      genre.columns[0] is the items column if user_based is True; users column if user_based is False.
      genre.columns[1] is an unused column.
      genre.columns[2:] are the genre levels.
    :returns: (n_x, f), where f is a DataFrame of size (n_x)-by-(n_g)
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


def compute_fpsim(user_based, trainset, fpsim_file):
    """
    Compute (n_x)-by-(n_x) matrix of fingerprint similarity
    n_x is the size of users if user_based is True, and "CID" indicates user;
    n_x is the size of items if user_based is not True, and "CID" indicates item.

    :param user_based: True to indicate user-based collaborative filtering method.
    :param trainset: a Trainset class instance.
    :param fpsim_file: a pickle file containing dataframe including columns "CID" and "fp".
    :returns: (n_x, n_x) numpy array
    """
    # Read fp
    fp_df = pd.read_pickle(fpsim_file) # with columns "CID" and "fp"

    # Select subset of fp
    if user_based:
        n_x = trainset.n_users
        raw_uid_list = [trainset.to_raw_uid(x) for x in range(n_x)]
        fp_df_subset = fp_df[fp_df["CID"].isin(raw_uid_list)].copy()
        fp_df_subset["inner_id"] = fp_df_subset["CID"].apply(trainset.to_inner_uid)
    else:
        n_x = trainset.n_items
        raw_iid_list = [trainset.to_raw_iid(x) for x in range(n_x)]
        fp_df_subset = fp_df[fp_df["CID"].isin(raw_iid_list)].copy()
        fp_df_subset["inner_id"] = fp_df_subset["CID"].apply(trainset.to_inner_iid)

    # Sort fp by inner-id of trainset
    fp_df_subset.sort_values(by='inner_id', inplace=True)

    # Compute fp similarity
    fp_sim = np.ones(shape=(n_x, n_x), dtype=np.float32)
    for i in range(n_x):
        fp_sim[i, :] = Chem.DataStructs.BulkTanimotoSimilarity(
            fp_df_subset["fp"].iloc[i], fp_df_subset["fp"].to_list())

    return fp_sim
