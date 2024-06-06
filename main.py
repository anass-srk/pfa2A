from flask import Flask,request

app = Flask(__name__)

def add_cors_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, PUT, PATCH, DELETE, OPTIONS'
    return response

app.after_request(add_cors_header)

@app.route("/calc",methods=['POST'])
def hello_world():
    result = request.json
    result = calculate_price(result)
    return {"result": result.item(0)}

@app.route("/build",methods=['GET'])
def Build():
    typee = request.args.get("typee")
    category = request.args.get("category")
    if typee not in categories:
        return f"{typee} is not a valid real estate type",400
    if category not in ['vendre','louer']:
        return f"{category} is not a valid real estate transaction",400
    return f"Built model-{typee}-{category}.pkl'"

import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import pymongo
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
import time
import pickle
import json
import os
import MONGO_URI

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

MONGO_DB_RENT = 'rent_db'
MONGO_DB_SELL = 'sell_db'
CLEAN_MONGO_DB_RENT = 'clean_rent_db'
CLEAN_MONGO_DB_SELL = 'clean_sell_db'

collection_name_map = {
    'appartements': 'appartements',
    'maisons': 'maisons',
    'villas_riad': 'villas',
    'bureaux_et_plateaux': 'bureaux',
    'magasins_et_commerces': 'locaux_commerciaux'
}

categories = list(collection_name_map.values())

def create_model(category: str,typee: str):
    # Deal with errors
    if typee not in categories:
        return f'built model-{typee}-{category}',200

    client = pymongo.MongoClient(MONGO_URI)
    db_rent = client[MONGO_DB_RENT]
    db_sell = client[MONGO_DB_SELL]
    clean_db_rent = client[CLEAN_MONGO_DB_RENT]
    clean_db_sell = client[CLEAN_MONGO_DB_SELL]

    db = clean_db_sell if 'vendre' in category else clean_db_rent
    cursor = db[typee].find({})
    df = pd.DataFrame(list(cursor))
    df.set_index("_id",inplace=True)
    client.close()
    not_features = ['_id','scraped_at','link','title','description','category','typee','price']
    features = [col for col in df.columns if col not in not_features]
    str_features = [col for col in ["city","property_age","area"] if col in df.columns]
    target = 'price'

    q1 = df['price'].quantile(0.25)
    q3 = df['price'].quantile(0.75)
    iqr = q3 - q1

    df = df[(df['price'] >= q1 - 1.5*iqr) & (df['price'] <= q3  + 1.5*iqr)]

    encoder = OneHotEncoder(sparse_output=False)  # Set sparse=False for easier handling
    encoded_features = encoder.fit_transform(df[str_features])
    encoded_features = pd.DataFrame(encoded_features)
    encoded_features.index = df.index

    # Combine encoded features with numerical features (if any)
    dfn = pd.concat([df[[col for col in df.columns if col not in str_features]],encoded_features], axis=1) 
    dfn.dropna(inplace=True)

    if 'vendre' in category:
        dfn['price'] = dfn['price']*dfn['habitable_size'] if 'habitable_size' in dfn.columns else dfn['price']*dfn['size']
    
    dfn.columns = dfn.columns.astype(str)
    dfn['list_time'] = dfn['list_time'].astype(str).apply(datetime.datetime.fromisoformat).apply(datetime.datetime.timestamp)
    features = [col for col in dfn.columns if col not in not_features]
    X = dfn[features]
    y = dfn[target]

    rf = RandomForestRegressor()
    rf.fit(X,y)
    
    with open(f'model-{typee}-{category}.pkl','wb') as f:
        pickle.dump(rf,f)

def calculate_price(elem: dict):
    test = pd.DataFrame([elem])
    
    typee = test.loc[test.index[0],'typee']
    category = test.loc[test.index[0],'category']

    if not os.path.exists(f'./model-{typee}-{category}.pkl'):
        create_model(category=category,typee=typee)
    
    if not os.path.exists(f'./model-{typee}-{category}.pkl'):
        return -1
    
    rf = None
    with open(f'model-{typee}-{category}.pkl','rb') as f:
        rf = pickle.load(f)

    client = pymongo.MongoClient(MONGO_URI)
    db_rent = client[MONGO_DB_RENT]
    db_sell = client[MONGO_DB_SELL]
    clean_db_rent = client[CLEAN_MONGO_DB_RENT]
    clean_db_sell = client[CLEAN_MONGO_DB_SELL]
    db = clean_db_sell if 'vendre' in category else clean_db_rent

    cursor = db[typee].find({})
    df = pd.DataFrame(list(cursor))
    df.set_index("_id",inplace=True)
    client.close()

    not_features = ['_id','scraped_at','link','title','description','category','typee','price']
    features = [col for col in df.columns if col not in not_features]
    str_features = [col for col in ["city","property_age","area"] if col in df.columns]
    target = 'price'

    q1 = df['price'].quantile(0.25)
    q3 = df['price'].quantile(0.75)
    iqr = q3 - q1

    df = df[(df['price'] >= q1 - 1.5*iqr) & (df['price'] <= q3  + 1.5*iqr)]

    encoder = OneHotEncoder(sparse_output=False)  # Set sparse=False for easier handling
    encoded_features = encoder.fit_transform(df[str_features])
    encoded_features = pd.DataFrame(encoded_features)
    encoded_features.index = df.index

        # Combine encoded features with numerical features (if any)
    dfn = pd.concat([df[[col for col in df.columns if col not in str_features]],encoded_features], axis=1) 
    dfn.dropna(inplace=True)
    dfn.columns = dfn.columns.astype(str)
    dfn['list_time'] = dfn['list_time'].astype(str).apply(datetime.datetime.fromisoformat).apply(datetime.datetime.timestamp)

    for col in ['scraped_at', 'link', 'title', 'description']:
        if col in test.columns:
            test.drop(columns=[col],inplace=True)

    for col in [col for col in ['city','area','property_age'] if col in df.columns]:
        if col not in test.columns:
            most_frequent = df[col].mode(dropna=True)
            # Replace null values with the most frequent value
            test[col] = most_frequent.iloc[0]
            
    for col in [col for col in ['city','area','property_age'] if col in df.columns]:
        for i in test.index :
            if test.loc[i,col] not in df[col].values:
                most_frequent = df[col].mode(dropna=True)
                # Replace null values with the most frequent value
                test.loc[i,col] = most_frequent.iloc[0]

    for col in [col for col in ['list_time','habitable_size','spare_rooms', 'rooms', 'size', 'bathrooms', 'floors','floor','office_units'] if col in df.columns]:
        if col not in test.columns:
            most_frequent = df[col].mean()
            test[col] = most_frequent
    
    for col in [col for col in ['Ascenseur','Balcon','Chauffage','Climatisation','Concierge','Cuisine équipée','Garage','Jardin','Meublée','Parking','Piscine','Sécurité','Terrasse'] if col in df.columns]:
        if col not in test.columns:
            true_count = df[col].fillna(False).value_counts().get(True, 0)
            false_count = df[col].fillna(True).value_counts().get(False, 0)
            # Determine most frequent (True or False)
            most_frequent = True if true_count >= false_count else False
            test[col] = most_frequent
    
    test.drop(columns=list(set(test.columns).difference(set(df.columns))),inplace=True)

    encoded_features = encoder.transform(test[str_features])
    encoded_features = pd.DataFrame(encoded_features)
    encoded_features.index = test.index

    features = [col for col in dfn.columns if col not in not_features]

    # Combine encoded features with numerical features (if any)
    testn = pd.concat([test[[col for col in test.columns if col not in str_features]],encoded_features], axis=1) 
    testn.columns = testn.columns.astype(str)
    testn['list_time'] = testn['list_time'].astype(str).apply(datetime.datetime.fromisoformat).apply(datetime.datetime.timestamp)

    return rf.predict(testn[features])