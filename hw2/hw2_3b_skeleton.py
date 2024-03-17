import sys

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
    return 


def get_matrix(file_name):
    """
    INPUT: file name
    OUTPUT: utility matrix from the file

    DESCRIPTION:
    Reads the utility matrix from the file. 
    """
    return


def user_based(umatrix, avg_score_dict, user_id):
    """
    INPUT: utility matrix, user id
    OUTPUT: top k recommended items

    DESCRIPTION:
    Returns the top recommendations using user-based collaborative
    filtering.
    """
    return


def item_based(umatrix, avg_score_dict, user_id):
    """
    INPUT: utility matrix, user id
    OUTPUT: top k recommended items

    DESCRIPTION:
    Returns the top recommendations using item-based collaborative
    filtering.
    """
    return


if __name__ == "__main__":
    """
    This is just an example of the main function.
    """
    umatrix = get_matrix(sys.argv[1])
    print(user_based(umatrix, target_user_id))
    print(item_based(umatrix, target_user_id))