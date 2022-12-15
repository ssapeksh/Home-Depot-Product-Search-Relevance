import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import zipfile
import time
from datetime import datetime
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
from rank_bm25 import BM25
from scipy.sparse import hstack
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import streamlit as st
import flask
from flask import Flask, jsonify, request,render_template

import warnings
warnings.filterwarnings('ignore')

train_data = zipfile.ZipFile('G:/Applied_AI/case_study_1/train.csv.zip')
train_data = pd.read_csv(train_data.open('train.csv'),encoding = "ISO-8859-1")
#print('train_data',train_data.shape)

attribute_data = zipfile.ZipFile('G:/Applied_AI/case_study_1/attributes.csv.zip')
attribute_data = pd.read_csv(attribute_data.open('attributes.csv'),encoding = "ISO-8859-1")
#print('Attribute_data',attribute_data.shape)

description_data = zipfile.ZipFile('G:/Applied_AI/case_study_1/product_descriptions.csv.zip')
description_data = pd.read_csv(description_data.open('product_descriptions.csv'),encoding = "ISO-8859-1")
#print('description_data',description_data.shape)

with open('G:/Final Data_1/Features/title.pkl','rb') as f:
    vectorizer_title = pickle.load(f)

with open('G:/Final Data_1/Features/search.pkl','rb') as f:
    vectorizer_search = pickle.load(f)

with open('G:/Final Data_1/Features/brand.pkl','rb') as f:
    vectorizer_brand = pickle.load(f)
    
with open('G:/Final Data_1/Features/description.pkl','rb') as f:
    vectorizer_des = pickle.load(f)
    
with open('G:/Final Data_1/Features/combine_feature_.pkl','rb') as f:
    vectorizer_combine_ft = pickle.load(f)
    
with open('G:/Final Data_1/Features/common_word_ST.pkl','rb') as f:
    vectorizer_common_ST = pickle.load(f)
    
with open('G:/Final Data_1/Features/common_word_SD.pkl','rb') as f:
    vectorizer_common_SD = pickle.load(f)
    
with open('G:/Final Data_1/Features/common_word_SB.pkl','rb') as f:
    vectorizer_common_SB = pickle.load(f)
    
with open('G:/Final Data_1/Features/len_product_title.pkl','rb') as f:
    normalizer_ft1 = pickle.load(f)
    
with open('G:/Final Data_1/Features/len_search_term.pkl','rb') as f:
    normalizer_ft2 = pickle.load(f)
    
with open('G:/Final Data_1/Features/len_of_brand.pkl','rb') as f:
    normalizer_ft3 = pickle.load(f)
    
with open('G:/Final Data_1/Features/len_product_description.pkl','rb') as f:
    normalizer_ft4 = pickle.load(f)
    
with open('G:/Final Data_1/Features/len_combine_feature_.pkl','rb') as f:
    normalizer_ft5 = pickle.load(f)
    
with open('G:/Final Data_1/Features/num_common_word_SB.pkl','rb') as f:
    normalizer_ft6 = pickle.load(f)
    
with open('G:/Final Data_1/Features/num_common_word_SD.pkl','rb') as f:
    normalizer_ft7 = pickle.load(f)

with open('G:/Final Data_1/Features/num_common_word_ST.pkl','rb') as f:
    normalizer_ft8 = pickle.load(f)

    
with open('G:/Final Data_1/Features/BM25_ST.pkl','rb') as f:
    normalizer_ft9 = pickle.load(f)

with open('G:/Final Data_1/Features/BM25_SB.pkl','rb') as f:
    normalizer_ft10 = pickle.load(f)

with open('G:/Final Data_1/Features/BM25_SD.pkl','rb') as f:
    normalizer_ft11 = pickle.load(f)

with open('G:/Final Data_1/Best_Model/GBDT.pkl','rb') as f:
    model_gbdt = pickle.load(f)
    
with open('G:/Final Data_1/BM25_model.pkl','rb') as f:
    bm25_model = pickle.load(f)
    



dataset = pd.read_pickle('G:/Final Data_1/Final_Database_bm25.pkl')

def merge_attributes(df):
    attr = attribute_data.copy()
    product_uid = df['product_uid'].values
    
    temp = attr.loc[attr['product_uid'].isin(product_uid)] 
    temp['combine_feature'] = temp['name'] + ' ' + temp['value']
    
    brands = temp[temp['name']=='MFG Brand Name']
    brands['brand'] = brands['value']
    brands.drop(['name','value','combine_feature'],axis=1,inplace=True)

    temp= temp.merge(brands,on='product_uid',how='left')
    temp['combine_feature_'] = temp.groupby('product_uid')['combine_feature'].transform(lambda x :''.join(str(x)))
    temp = temp.drop_duplicates(subset=['product_uid'])
    df = df.merge(temp,on='product_uid',how='left').set_index(df.index)
    df.drop(['name','value','combine_feature'],axis=1,inplace=True)
    return df


def merge_description(df):
    descrip = description_data.copy()
    product_uid = df['product_uid'].values
    temp = descrip.loc[descrip['product_uid'].isin(product_uid)]
    df = df.merge(temp,on='product_uid',how='left').set_index(df.index)
    return df


def extract_n_words(n,text):
    if n>len(text.split()):
        return 'invalid'
    return ' '.join(text.split()[:n])

def fill_brand(df):
    null_brand_values = df[df['brand'].isna()]
    unique_brands = df['brand'].unique()

    for i,j in null_brand_values.iterrows():
        title=j['product_title']
        if extract_n_words(6,title) in unique_brands:
            null_brand_values['brand'].loc[i] = extract_n_words(6, title)
        elif extract_n_words(5,title) in unique_brands:
            null_brand_values['brand'].loc[i] = extract_n_words(5, title)
        elif extract_n_words(4,title) in unique_brands:
            null_brand_values['brand'].loc[i] = extract_n_words(4, title)
        elif extract_n_words(3,title) in unique_brands:
            null_brand_values['brand'].loc[i] = extract_n_words(3, title)
        elif extract_n_words(2,title) in unique_brands:
            null_brand_values['brand'].loc[i] = extract_n_words(2, title)
        else:
            null_brand_values['brand'].loc[i] = extract_n_words(1, title)
            
    df['brand'].loc[null_brand_values.index]=null_brand_values['brand'].values
    return df

def fill_attributes(df):
    null_df = df[df['combine_feature_'].isna()]
    null_df['combine_feature_'] = null_df['product_description'].copy()
    df['combine_feature_'].loc[null_df.index] = null_df['combine_feature_'].values
    return df

#Reference : https://towardsdatascience.com/modeling-product-search-relevance-in-e-commerce-home-depot-case-study-8ccb56fbc5ab

def standardize_units(text):
    text = " " + text + " "
    text = re.sub('( gal | gals | galon )',' gallon ',text)
    text = re.sub('( ft | fts | feets | foot | foots )',' feet ',text)
    text = re.sub('( squares | sq )',' square ',text)
    text = re.sub('( lb | lbs | pounds )',' pound ',text)
    text = re.sub('( oz | ozs | ounces | ounc )',' ounce ',text)
    text = re.sub('( yds | yd | yards )',' yard ',text)
    return text

def preprocessing(text):
    
    text = text.replace('in.','inch')  # Replace in. with inch
    text = re.sub('[^A-Za-z0-9.]+',' ',text) # remove special characters except '.'
    text = re.sub(r"(?<!\d)[.,;:](?!\d)",'',text,0) # https://stackoverflow.com/questions/43142710/remove-all-punctuation-from-string-except-if-its-between-digits
    text = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ", text)
    text = standardize_units(text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

stop_words = stopwords.words('english')
ps = PorterStemmer()

def stopwords_stemming(text):
    words = text.split()
    words = [w for w in words if w not in stop_words] # Stopwords
    words = [ps.stem(word) for word in words] # stemming
    return ' '.join(words)

def stemming_search(text):
    words = text.split()
    words = [ps.stem(word) for word in words] # stemming
    return ' '.join(words)


def common_word(feature_1,feature_2):
    common_word=[]
    words,count = feature_1.split(),0
    for i in words:
        set_1 = set(feature_1.split())
        set_2 = set(feature_2.split())
        common_ = set_1.intersection(set_2)
        common_ = ' '.join(common_)
        common_word.append(common_)
        return ''.join([i for i in common_word])
    
def cosine_similarity(feature_1,feature_2):
    f_1 = set(feature_1.split())
    f_2 = set(feature_2.split())
    num = len(f_1.intersection(f_2))
    deno = np.sqrt(len(f_1)) * np.sqrt(len(f_2))
    
    if deno == 0:
        return 0
    else:
        return num/deno
    
def jaccard_coefficient(feature_1,feature_2):
    f_1 = set(feature_1.split())
    f_2 = set(feature_2.split())
    num = len(f_1.intersection(f_2))
    deno = len(f_1 | f_2)
    if deno == 0:
        return 0
    else:
        return num/deno
    
from rank_bm25 import BM25

def bm25_params(corpus):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    idf_name_value = dict(zip(vectorizer.get_feature_names(),(list(vectorizer.idf_))))
    length = [len(i.split()) for i in corpus]
    avgdl = np.average(length)                      
    param = {'idf_name_value':idf_name_value,'avgdl':avgdl,'len_corpus':len(corpus)}
    return param

def bm25_scores(param,text,query,k=1.5,b=0.75):
    idf_name_value = param['idf_name_value']
    avgdl = param['avgdl']
    N=param['len_corpus']
    score = 0
    
    for word in query.split():
        mod_d = len(text.split())  # len of document
        n_tf = text.count(word)   # no of times query occur in document
        
        if word in idf_name_value.keys():  # check if word present in document
            idf_score = idf_name_value[word]
        else:
            idf_score = np.log(1+N)+1    #  idf for words not in document    
        score_ = idf_score * (n_tf*(k+1) / (n_tf + k * (1-b + b * (mod_d / avgdl))))
        score+=score_
    return score


def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('G:/Final Data_1/dataset_title_brand_descrip.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
def corrected_terms(text):
    temp = text.split()
    temp = [correction(word) for word in temp]
    return ' '.join(temp)


temp_1 = pd.read_pickle('G:/Final Data_1/clean_test_df.pkl')
test_data = train_data.loc[temp_1.index[:]]
true_labels = train_data.loc[temp_1.index[:]]['relevance']


def final(data_input):
    temp = data_input.copy()
    dataset = merge_attributes(temp)
    dataset = merge_description(dataset)
    dataset = fill_brand(dataset)
    dataset = fill_attributes(dataset)

    data = dataset.copy()

    data['product_title'] = data['product_title'].apply(lambda x: preprocessing(x))
    data['search_term'] = data['search_term'].apply(lambda x: preprocessing(x)) 
    data['brand'] = data['brand'].apply(lambda x: preprocessing(x))
    data['combine_feature_'] = data['combine_feature_'].apply(lambda x: preprocessing(x))
    data['product_description'] =data['product_description'].apply(lambda x: preprocessing(x))

    """
    furthur preprocessing
    """

    data['product_title'] = data['product_title'].apply(lambda x: stopwords_stemming(x))
    data['search_term'] = data['search_term'].apply(lambda x: stemming_search(x))
    data['brand'] = data['brand'].apply(lambda x: stopwords_stemming(x))
    data['combine_feature_'] = data['combine_feature_'].apply(lambda x: stopwords_stemming(x))
    data['product_description'] = data['product_description'].apply(lambda x: stopwords_stemming(x))

 
    data['product_info'] = data['product_title'] + ' \t ' + data['product_description'] + ' \t ' + data['brand']+ ' \t ' + data['search_term']


    data['len_product_title'] = data['product_title'].str.split().apply(len)
    data['len_search_term'] = data['search_term'].str.split().apply(len)
    data['len_of_brand'] = data['brand'].str.split().apply(len)
    data['len_product_description'] = data['product_description'].str.split().apply(len)
    data['len_combine_feature_'] = data['combine_feature_'].str.split().apply(len)


    data['common_word_ST'] = data['product_info'].map(lambda x: common_word(x.split('\t')[3],x.split('\t')[0]))
    data['common_word_SD'] = data['product_info'].map(lambda x: common_word(x.split('\t')[3],x.split('\t')[1]))
    data['common_word_SB'] = data['product_info'].map(lambda x: common_word(x.split('\t')[3],x.split('\t')[2]))


    # num common words
    data['num_common_word_ST'] = data['common_word_ST'].apply(lambda x: len(x.split()))
    data['num_common_word_SD'] = data['common_word_SD'].apply(lambda x: len(x.split()))
    data['num_common_word_SB'] = data['common_word_SB'].apply(lambda x: len(x.split()))

    # cosine distance
    data['cosine_ST'] = data.apply(lambda x: cosine_similarity(x['search_term'],x['product_title']),axis=1)
    data['cosine_SB'] = data.apply(lambda x: cosine_similarity(x['search_term'],x['brand']),axis=1)
    data['cosine_SD'] = data.apply(lambda x: cosine_similarity(x['search_term'],x['product_description']),axis=1)

    # Jaccard Coefficient
    data['jaccard_ST'] = data.apply(lambda x: jaccard_coefficient(x['search_term'],x['product_title']),axis=1)
    data['jaccard_SB'] = data.apply(lambda x: jaccard_coefficient(x['search_term'],x['brand']),axis=1)
    data['jaccard_SD'] = data.apply(lambda x: jaccard_coefficient(x['search_term'],x['product_description']),axis=1)

    data['ratio_title_search'] = data['len_product_title'] / data['len_search_term']
    data['ratio_descrip_search'] = data['len_product_description'] / data['len_search_term']
    data['ratio_common_ST_to_search_term'] = data['num_common_word_ST'] / data['len_search_term']
    data['ratio_common_SD_to_search_term'] = data['num_common_word_SD'] / data['len_search_term']
    data['ratio_common_SB_to_search_term'] = data['num_common_word_SB'] / data['len_search_term']


    #---------------------------------search to title------------------------------------
    param_title = bm25_params(data['product_title'])
    data['BM25_ST']  = data.apply(lambda x : bm25_scores(param_title,x['product_title'],x['search_term']),axis=1)
    #---------------------------------search to brand------------------------------------
    param_brand = bm25_params(data['brand'])
    data['BM25_SB']  = data.apply(lambda x : bm25_scores(param_brand,x['brand'],x['search_term']),axis=1)
    #---------------------------------search to description------------------------------------
    param_desc = bm25_params(data['product_description'])
    data['BM25_SD']  = data.apply(lambda x : bm25_scores(param_brand,x['product_description'],x['search_term']),axis=1)

    title =vectorizer_title.transform(data['product_title'].values)
    search =vectorizer_search.transform(data['search_term'].values)
    brand =vectorizer_brand.transform(data['brand'].values)
    des =vectorizer_des.transform(data['product_description'].values)
    combine_ft =vectorizer_combine_ft.transform(data['combine_feature_'].values)

    common_ST =vectorizer_common_ST.transform(data['common_word_ST'].values)
    common_SD =vectorizer_common_SD.transform(data['common_word_SD'].values)
    common_SB =vectorizer_common_SB.transform(data['common_word_SB'].values)

    len_title =normalizer_ft1.transform(data["len_product_title"].values.reshape(-1,1))
    len_search_term =normalizer_ft2.transform(data["len_search_term"].values.reshape(-1,1))
    len_brand =normalizer_ft3.transform(data["len_of_brand"].values.reshape(-1,1))
    len_des =normalizer_ft4.transform(data["len_product_description"].values.reshape(-1,1))
    len_combine_ft =normalizer_ft5.transform(data["len_combine_feature_"].values.reshape(-1,1))

    num_SB =normalizer_ft6.transform(data["num_common_word_SB"].values.reshape(-1,1))
    num_SD =normalizer_ft7.transform(data["num_common_word_SD"].values.reshape(-1,1))
    num_ST =normalizer_ft8.transform(data["num_common_word_ST"].values.reshape(-1,1))

    bm25_st =normalizer_ft9.transform(data["BM25_ST"].values.reshape(-1,1))
    bm25_sb =normalizer_ft10.transform(data["BM25_SB"].values.reshape(-1,1))
    bm25_sd =normalizer_ft11.transform(data["BM25_SD"].values.reshape(-1,1))

    cosine_ST = data['cosine_ST'].values.reshape(-1,1)
    cosine_SD = data['cosine_SD'].values.reshape(-1,1)
    cosine_SB = data['cosine_SB'].values.reshape(-1,1)

    jaccard_ST = data['jaccard_ST'].values.reshape(-1,1)
    jaccard_SD = data['jaccard_SD'].values.reshape(-1,1)
    jaccard_SB = data['jaccard_SB'].values.reshape(-1,1)

    ratio_title_search = data['ratio_title_search'].values.reshape(-1,1)
    ratio_desc_search = data['ratio_descrip_search'].values.reshape(-1,1)
    ratio_common_ST_search = data['ratio_common_ST_to_search_term'].values.reshape(-1,1)
    ratio_common_SD_search = data['ratio_common_SD_to_search_term'].values.reshape(-1,1)
    ratio_common_SB_search = data['ratio_common_SB_to_search_term'].values.reshape(-1,1)

    stack_data=hstack((title,search,des,brand,combine_ft,len_brand,len_combine_ft,len_des,len_search_term,len_title,
                      common_SB,common_SD,common_ST,num_SB,num_SD,num_ST,cosine_SB,cosine_SD,cosine_ST,jaccard_SB,jaccard_SD,
                      jaccard_ST,ratio_common_SB_search,ratio_common_SD_search,ratio_common_ST_search,
                      ratio_desc_search,ratio_title_search,bm25_sb,bm25_sd,bm25_st)).tocsr()
    
    
    predict = model_gbdt.predict(stack_data)
    return predict


corpus = dataset['product_info'].values
def get_data(search,N):
    corrected_search_query = corrected_terms(search)
    tokenized_query = corrected_search_query.split(' ')
    data = bm25_model.get_top_n(documents=corpus , query=tokenized_query,n=N)
    data_result = dataset[dataset['product_info'].isin(data)]
    data_result['search_term'] = corrected_search_query
    features =['product_uid','product_title','search_term']
    return data_result[features]

def generate_result(search,N):
    test = get_data(search,N)
    test['relevance'] =final(test)
    return test.sort_values('relevance',ascending=False)['product_title'].values

app =Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    search = request.form.to_dict()['search_term']
    results = generate_result(search, 10)
    params = {'results':results, 'search':search}
    return render_template('prediction.html', parameters=params)

if __name__=='__main__':
    app.run(debug=True)
