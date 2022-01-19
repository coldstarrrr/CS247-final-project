import numpy as np
import pandas as pd

def RMSE(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))
    
def precision_recall_at_k(df, label_pred, k=10, threshold=3.5, predict_relevance=True):
  ## df should have the same length as the test set, and should have 4 columns: user_id, beer_id, review_overall, and the predicted value column [label_pred]
  users = df.user_id.unique()
  prec_at_k = {}
  rec_at_k = {}
  for user in users:
    user_ratings = df[df['user_id'] == user]
    # sort the predicted ratings (or probability that an item is relevant to the user) in descending order
    user_ratings=user_ratings.sort_values(by=label_pred, ascending=False)
    # number of relevant items in the test data
    n_relevant_items = len(user_ratings[user_ratings['review_overall']>=threshold])
    top_k = user_ratings[:k]
    if predict_relevance:
      # number of recommended items in the top k
      n_rec_k = len(top_k[top_k[label_pred]>0.5])
      # number of relevant and recommended items in top k
      n_rel_and_rec_k = len(top_k.loc[(top_k['review_overall'] >= threshold) & (top_k[label_pred] > 0.5)])
    else:
      n_rec_k = len(top_k[top_k[label_pred]>threshold])
      n_rel_and_rec_k = len(top_k.loc[(top_k['review_overall'] >= threshold) & (top_k[label_pred] >= threshold)])
    
    # Precision@K: Proportion of recommended items that are relevant
    # When n_rec_k is 0, Precision is undefined. We here set it to 0.
    prec_at_k[user] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
    # Recall@K: Proportion of relevant items that are recommended
    # When n_relevant_items is 0, Recall is undefined. We here set it to 0.
    rec_at_k[user] = n_rel_and_rec_k / n_relevant_items if n_relevant_items != 0 else 0
  
  return prec_at_k, rec_at_k
  

def accuracy(pred, target):
  return sum(target==pred)/len(pred)