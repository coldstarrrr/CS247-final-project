import pandas as pd
import numpy as np

def get_user_by_id(id):
    user_id_mapping = pd.read_pickle("/content/drive/MyDrive/CS247/Data/user_id_mapping.pkl")
    return user_id_mapping[user_id_mapping['id']==id]
    
def get_id_by_user(user):
    user_id_mapping = pd.read_pickle("/content/drive/MyDrive/CS247/Data/user_id_mapping.pkl")
    return user_id_mapping[user_id_mapping['review_profilename']==user]
    
def get_beer_by_id(id):
    beer_id_mapping = pd.read_pickle("/content/drive/MyDrive/CS247/Data/beer_id_mapping.pkl")
    return beer_id_mapping[beer_id_mapping['beer_beerid']==id]
    
def get_id_by_beer(beer):
    beer_id_mapping = pd.read_pickle("/content/drive/MyDrive/CS247/Data/beer_id_mapping.pkl")
    return beer_id_mapping[beer_id_mapping['beer_name']==beer]
    
def merge_user_id(df, on):
    user_id_mapping = pd.read_pickle("/content/drive/MyDrive/CS247/Data/user_id_mapping.pkl")
    return pd.merge(df, user_id_mapping, on=on, how='inner')
    
def merge_beer_id(df, on):
    beer_id_mapping = pd.read_pickle("/content/drive/MyDrive/CS247/Data/beer_id_mapping.pkl")
    return pd.merge(df, beer_id_mapping, on=on, how='inner')