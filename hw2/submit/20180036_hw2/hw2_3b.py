import sys
import numpy as np
from numpy.linalg import norm
import re
import time


# Take average of top-k similar user's ratings
topk_users_to_average = 10
# Take average of top-k similar items ratings
topk_items_to_average = 10
# Considering items 1 to 1000
num_items_for_prediction = 1000
# Top-k predictions of items with highest ratings
topk_items = 5
# Target user's id
target_user_id = 600


def cosine(a, b):
    """
    INPUT: two vectors a and b
    OUTPUT: cosine similarity between a and b

    DESCRIPTION:
    Takes two vectors and returns the cosine similarity.
    """
    a_copy = np.nan_to_num(a)
    b_copy = np.nan_to_num(b)
    
    magnitude_a = norm(a_copy)
    magnitude_b = norm(b_copy)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    return np.dot(a_copy, b_copy) / (magnitude_a * magnitude_b)

def get_matrix(file_name):
    """
    INPUT: file name
    OUTPUT: utility matrix from the file

    DESCRIPTION:
    Reads the utility matrix from the file. 
    """
    utility_dict = {}
    normalized_utility_dict = {}
    user_set = set();
    movie_set = set();
    with open(file_name,'r') as file:
        for line in file:
            utility = [float(utility) for utility in re.split('[^\d+.\d+]+',line) if utility]
            user_id = utility[0]
            movie_id = utility[1]
            rating = utility[2]
            timestamp = utility[3]
            user_set.add(user_id)
            movie_set.add(movie_id)
            if user_id in utility_dict:
                utility_dict[user_id][movie_id] = rating
            else:
                utility_dict[user_id] = {}
                utility_dict[user_id][movie_id] = rating
            
    user_list = list(user_set)
    movie_list = list(movie_set)
    user_length = len(user_list)
    movie_length = len(movie_list)
    
    user_dicts = {value: index for index, value in enumerate(user_list)}
    movie_dicts = {value: index for index, value in enumerate(movie_list)}
    
    utility_matrix = np.full((user_length, movie_length),np.nan);
    
    for user in utility_dict:
        utility = utility_dict[user]
        for movie in utility:
            rating = utility[movie]
            user_index = user_dicts[user]
            movie_index = movie_dicts[movie]
            utility_matrix[user_index][movie_index] = rating
    
    row_average = np.nanmean(utility_matrix, axis=1)
    row_average_dict = {index: average for index, average in enumerate(row_average)}
    
    col_average = np.nanmean(utility_matrix, axis=0)
    col_average_dict = {index:average for index, average in enumerate(col_average)}
    
    """
    row_average = {
        index of numpy array : average value
    }
    
    numpy matrix = 
    [user_index][movie_index] = rating
    
    user_list = [user_ids]
    movie_list = [movie_ids]
    user_dict = {
        'user_id' : user_index
    }
    movie_dict = {
        'movie_id' : movie_index
    }
    
    """
    
    return utility_matrix,row_average_dict,col_average_dict, user_list, movie_list, user_dicts, movie_dicts

def top10_users(utility_matrix, similarity_matrix, target_movie, user_dicts, movie_dicts):
    target_movie_index = movie_dicts[target_movie]
    # print(target_movie_index)
    top10_users_index = []
    for i in range(topk_users_to_average):
        user_index = int(similarity_matrix[i][1])
        # print('user_index:', user_index)
        rating = utility_matrix[user_index][target_movie_index]
        # print('rating:', rating)
        if ~np.isnan(rating):
            # print(rating,user_index)
            top10_users_index.append(user_index)
    # print(utility_matrix[top10_users_index,target_movie_index])
    return top10_users_index

    


def user_based(umatrix, avg_score_dict, user_id, target_movies, user_dicts, movie_dicts):
    """
    INPUT: utility matrix, user id
    OUTPUT: top k recommended items

    DESCRIPTION:
    Returns the top recommendations using user-based collaborative
    filtering.
    """
    # print(umatrix[:,movie_dicts[5]])
    # print(umatrix[:,movie_dicts[27751]])
    normalized_umatrix = np.copy(umatrix)
    # print('user_index: ', user_dicts[user_id])
    user_length = len(user_dicts)
    avg_score_numpy_array = np.zeros(user_length)
    for i in range(user_length):
        avg_score_numpy_array[i] = avg_score_dict[i]
    
    ## normalizing utility matrix
    avg_score_numpy_array = np.reshape(avg_score_numpy_array,(-1,1))
    normalized_umatrix = normalized_umatrix - avg_score_numpy_array
    
    ## for calculating similarity of user_matrix, duplicate the user utility 
    user_index = user_dicts[user_id]
    user_utility = normalized_umatrix[user_index]
    user_matrix = np.tile(user_utility,(user_length,1))
    
    ## calculate similarity_matrix
    similarity_matrix = np.reshape(np.vectorize(cosine, signature='(n),(n)->()')(normalized_umatrix,user_matrix),(-1,1))
    
    ## add the row index as a column of similarity matrix
    num_rows = similarity_matrix.shape[0]
    row_indices = np.arange(num_rows).reshape(-1, 1)
    similarity_matrix = np.hstack((similarity_matrix, row_indices))    
    ## sort the similairty matrix in descending order of similarity
    sorted_indices = np.argsort(similarity_matrix[:, 0])
    descending_indices = sorted_indices[::-1]
    sorted_similarity_matrix = similarity_matrix[descending_indices]
    
    ## remove the first row, because the first row is always 1
    sorted_similarity_matrix = sorted_similarity_matrix[1:]
    
    predicted_rating = {}
    # print(movie_dicts)
    for target_movie in target_movies:
        if(target_movie not in movie_dicts):
            continue;
        target_movie_index = movie_dicts[target_movie]
        top10_user_list = top10_users(umatrix,sorted_similarity_matrix, target_movie, user_dicts, movie_dicts)
        # print(top10_user_list)
        if(len(top10_user_list)>0):
            selected_utility = umatrix[top10_user_list,target_movie_index]
            average_rating = np.mean(selected_utility, axis=0)
            predicted_rating[target_movie] = average_rating
    
    predicted_rating = dict(sorted(predicted_rating.items(), key=lambda item: item[1],reverse=True))
    # print(predicted_rating)
    
    predicted_rating_keys = list(predicted_rating.keys())
    # print(predicted_rating_keys)
    for i in range(topk_items):
    # for i in range(2):

        movie_id = predicted_rating_keys[i];
        movie_rating = predicted_rating[movie_id]
        print(f'{movie_id}\t{movie_rating}')
    
    return predicted_rating



def item_based(umatrix, avg_score_dict, user_id, target_movies, user_dicts, movie_dicts, movie_list):
    """
    INPUT: utility matrix, user id
    OUTPUT: top k recommended items

    DESCRIPTION:
    Returns the top recommendations using item-based collaborative
    filtering.
    """
    ## construct avg_score_numpy_array for movie
    rating_matrix = np.copy(umatrix.T)
    # print(umatrix_T[movie_dicts[5]])
    # print(umatrix_T[movie_dicts[27751]])
    # print('user_index: ', user_dicts[user_id])
    # print(len(avg_score_dict))
    movie_length= len(movie_dicts)
    user_length = len(user_dicts)
    # print(movie_length)
    avg_score_np_array = np.zeros(user_length)
    for i in range(user_length):
        avg_score_np_array[i] = avg_score_dict[i]
        
    ## normalizing utility matrix
    avg_score_np_array = np.reshape(avg_score_np_array, (-1,1))
    normalized_umatrix = umatrix - avg_score_np_array
    normalized_rating_matrix = normalized_umatrix.T

    ## calculating similarity of movie_matrix
    movie_indicies = [ movie_dicts[i] for i in range(1,num_items_for_prediction+1) if i in movie_dicts]
    start = time.time()
    predicted_rating = {}
    for target_movie in target_movies:
        if target_movie not in movie_dicts:
            continue
        # if(target_movie%10==0):
        #     end = time.time()
        #     print(f'elapsed_time: {end-start}, movie_id: {target_movie}')
        index = movie_dicts[target_movie]
        # print(f'target_movie: {target_movie}, index: {index}')
        ratings = normalized_rating_matrix[index]
        # print(movie_rating.shape)
        # print(normalized_umatrix_T.shape)
        # print(len(user_dicts))
        movie_ = np.tile(ratings, (movie_length,1))
        
        ## calculate similarity matrix
        similarity_matrix = np.reshape(np.vectorize(cosine, signature='(n),(n)->()')(normalized_rating_matrix, movie_),(-1,1))
        
        ## add the row index as a column of similarity matrix
        num_rows = similarity_matrix.shape[0]
        row_indices = np.arange(num_rows).reshape(-1, 1)
        movie_ids = np.array([ movie_list[row[0]] for row in row_indices]).reshape(-1,1)
        # print(row_indices.shape)
        similarity_matrix = np.hstack((similarity_matrix, row_indices,movie_ids))    
        # print(similarity_matrix[movie_dicts[3318]])
        
        ## remove the similarity with movies which id is in target_movies
        # print(similarity_matrix.shape)
        # print(len(movie_indicies))
        mask = np.ones(len(similarity_matrix), dtype=bool)
        mask[movie_indicies] = False
        # print(movie_indicies)
        similarity_matrix = similarity_matrix[mask]
        # print(similarity_matrix.shape)

        row_tuples = [tuple(row) for row in similarity_matrix]
        row_tuples.sort(key=lambda x: (-x[0], x[2]))
        top10_similarity_matrix = np.array(row_tuples[:10])
        # print(row_tuples[:100])
        # print(f'similarity_matrix: {target_movie}',top10_similarity_matrix)
        
        top10_similar_movie_index = np.copy(top10_similarity_matrix[:,1]).astype(int)
        # print('cosine: ', cosine(normalized_umatrix_T[index],normalized_umatrix_T[movie_dicts[int(new_top10_matrix[0,1])]]))
        similar_ratings = rating_matrix[top10_similar_movie_index, user_dicts[user_id]]
        # print(f'movie_id: {target_movie}, movie_index:{movie_dicts[target_movie]}, user_id: {user_id}, user_index: {user_dicts[user_id]}')
        # for i in range(10):
        #     movie_index = top10_similar_movie_index[i]
        #     print(f'{i},movie_index: {movie_index},movie_id: {movie_list[movie_index]}, rating: {rating_matrix[movie_index][user_dicts[user_id]]}')
        
        if(~np.all(np.isnan(similar_ratings))):
            average_rating = np.nanmean(similar_ratings)
            # print(f'similar_movie_index: {top10_similar_movie_index}, similar_ratings: {similar_ratings}, ')
            predicted_rating[target_movie] = average_rating
        
    predicted_rating = dict(sorted(predicted_rating.items(), key=lambda item: item[1],reverse=True))
    # print(predicted_rating)
    predicted_rating_keys = list(predicted_rating.keys())
    # print(normalized_rating_matrix[movie_dicts[5]])
    # print(normalized_rating_matrix[movie_dicts[3318]])
    # print('cosine:: ',cosine(normalized_rating_matrix[movie_dicts[5]],normalized_rating_matrix[movie_dicts[3318]]))
    for i in range(topk_items):
        movie_id = predicted_rating_keys[i];
        movie_rating = predicted_rating[movie_id]
        print(f'{movie_id}\t{movie_rating}')

    return predicted_rating


if __name__ == "__main__":
    """
    This is just an example of the main function.
    """
    target_movies = list(range(1,num_items_for_prediction+1))
    # target_movies = list(range(1,5+1))

    utility_matrix,row_average_dict, col_average_dict, user_list, movie_list, user_dicts, movie_dicts = get_matrix(sys.argv[1])
    predicted_rating_user = user_based(utility_matrix,row_average_dict, target_user_id,target_movies, user_dicts, movie_dicts)
    # for i in range(similarity_matrix.shape[0]):
    #     similarity_matrix[i,1] = user_list[int(similarity_matrix[i,1])]
    # # print(similarity_matrix[:10])
    
    movie_utility_matrix = np.copy(utility_matrix.T)
    predicted_ratings_items = item_based(utility_matrix, row_average_dict, target_user_id, target_movies, user_dicts, movie_dicts, movie_list)