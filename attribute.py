

import pandas as pd
import numpy as np
import datetime
from math import ceil
import statsmodels.api as sm
from tslearn.barycenters import euclidean_barycenter
from tslearn.barycenters import softdtw_barycenter
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
import statistics





def matching_features(traits_data):
    matching_counter_values = np.array([])
    for index, traits in traits_data.iterrows():
        traits.dropna(inplace=True)
        matching_counter = 0
        for i in range(0, len(traits)-1):
            for j in range(i+1, len(traits)):
                if len(set(str(traits[i]).split()) & set(str(traits[j]).split())): 
                    matching_counter += 1
        matching_counter_values = np.append(matching_counter_values, matching_counter)
    traits_data["# matching traits"] = matching_counter_values
    return traits_data


# In[4]:


def score_rarity(traits_data):
    labels = traits_data.columns
    rarity_values = {}
    for index, traits in traits_data.iterrows():
        rarity_value = []
        for i in range(0, len(traits)):
            if not(traits[i] != traits[i]):
                rarity = 1 / (traits_data[labels[i]].value_counts()[traits[i]] / len(traits_data))
                rarity_value.append(rarity)
            else:
                rarity_value.append(np.nan)
        rarity_values[index] = rarity_value
    return(pd.DataFrame.from_dict(rarity_values, orient='index', 
                                  columns = traits_data.columns+' rarity score'))


# In[5]:


# functions


# In[6]:


def attributes_data_preprocessing(attributes_data):
    attributes_data.drop("Unnamed: 0", inplace=True, axis=1)
    attributes_data = matching_features(attributes_data)
    attributes_data.fillna("Not given", inplace=True)
    attributes_data = pd.concat([attributes_data['Name'], score_rarity(attributes_data)], axis=1)
    attributes_data.rename(columns={'Name':'tokenID'}, inplace=True)
    uniqueness_column_check(attributes_data)
    return attributes_data


# In[7]:


def uniqueness_column_check(attributes_data):
    for column in attributes_data:
        if (attributes_data[column] == attributes_data[column][0]).all():
            attributes_data.drop(column, axis=1, inplace=True)


# In[8]:


def transactions_data_preprocessing(transactions_data, collection_name):
    transactions_data.drop("Unnamed: 0", inplace=True, axis=1)
    transactions_data.drop(transactions_data[transactions_data['from'] == "0x0000000000000000000000000000000000000000"].index, 
                      inplace=True)
    transactions_data['tokenID'] = transactions_data['tokenID'].apply(lambda x: 
                                                                      collection_name + ' #' + str(x))
    transactions_data.drop(list(set(transactions_data.columns.tolist()) 
                           - set(['value', 'timeStamp', 'tokenID'])), axis=1, inplace=True)
    return transactions_data


# In[9]:


def collection_name(attributes_data):
    return attributes_data.tokenID[0][:attributes_data.tokenID[0].rfind(' ')]


# In[10]:


def attributes_transactions_merge(attributes_data, transaction_data):
    attributes_transactions = pd.merge(attributes_data, transaction_data, on='tokenID')
    attributes_transactions.sort_values("timeStamp", inplace=True)
    attributes_transactions["timeStamp"] = attributes_transactions['timeStamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    return attributes_transactions


# In[11]:


def train_test_week_split(attributes_transactions):
    nWeeks = ceil(((attributes_transactions.iloc[-1].timeStamp) 
                   - (attributes_transactions.iloc[0].timeStamp)).days / 7)
    ratio = 3
    nTestWeeks = ceil(nWeeks / ratio)
    nTrainWeeks = nWeeks - nTestWeeks
    return nTrainWeeks, nTestWeeks


# In[12]:


def week_split(attributes_transactions):
    nWeeks = ceil(((attributes_transactions.iloc[-1].timeStamp) 
                   - (attributes_transactions.iloc[0].timeStamp)).days / 7)
    
    return nWeeks


# In[13]:


def get_column_names(attributes_transactions):
    column_names = attributes_transactions.columns.tolist()
    column_names = [e for e in column_names if e not in ("tokenID", 'timeStamp', 'value')]
    return column_names


# In[24]:


def coef_estimation(n_weeks, data, column_names):
    coef = {col:[] for col in column_names}
    
    # splitting into 1-week datasets
    for i in range(0, 2):
        data_i = data[(data.iloc[0].timeStamp + datetime.timedelta(weeks=i) <=data.timeStamp) &
                      (data.timeStamp<= data.iloc[0].timeStamp + datetime.timedelta(weeks=i+2))]
        #dropping all transaction values except the very last
        data_i.sort_values('timeStamp').drop_duplicates('tokenID',keep='last')
        
        y = data_i.value.astype(float)
        y = (y - min(y)) / (max(y) - min(y))
        X = data_i.drop(['value', 'tokenID', 'timeStamp'], axis=1).astype(float)

        alphas = np.logspace(-2, 0, 2000)
        
#         coefs_graph = []
#         for a in alphas:
#             model = linear_model.Lasso(alpha=a, fit_intercept=False, positive=True)
#             fitted_model = model.fit(X=X, y=y)
#             coefs_graph.append(fitted_model.coef_)
#         ax = plt.gca()

        tuned_parameters = [{"alpha": alphas}]
        n_folds = 5

        lasso = linear_model.Lasso(fit_intercept=False, positive=True)

        gridSearch = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
        gridSearch.fit(X, y)
       

        lasso_best = linear_model.Lasso(alpha=gridSearch.best_params_['alpha'], fit_intercept=False, 
                                     positive=True)

        fitted_lasso_best = lasso_best.fit(X=X, y=y)

        
        for i in range(len(column_names)):
            coef[column_names[i]].append(fitted_lasso_best.coef_[i])
    return coef


# In[15]:


def consensus_search(coef):
    traits_coef = pd.DataFrame.from_dict(coef)

    euclidean = euclidean_barycenter(traits_coef).flatten()
#     soft_dtw = softdtw_barycenter(traits_coef, gamma=0.5).flatten()
    
    traits_coef.loc[len(traits_coef)] = euclidean
    
    return(traits_coef)


# In[16]:

def rarity_score_calculation(attributes, transactions):

    # attributes = pd.read_csv(path_to_attributes)
    attributes = attributes_data_preprocessing(attributes)

    # transactions = pd.read_csv(path_to_transactions)
    collectionName = collection_name(attributes)
    transactions = transactions_data_preprocessing(transactions, collectionName)


    attributes_transactions = attributes_transactions_merge(attributes, transactions)


    # In[19]:


    nWeeks = week_split(attributes_transactions)


    # In[20]:


    columnNames = get_column_names(attributes_transactions)


    # In[ ]:


    c = coef_estimation(nWeeks, attributes_transactions, columnNames)


    # In[ ]:


    traits_coef = consensus_search(c)


    # In[ ]:


    attributes_scores = attributes.drop(['tokenID'], axis=1) * traits_coef.iloc[-1]
    attributes_scores['final score'] = attributes_scores.sum(axis='columns')
    attributes_scores= attributes_scores.join(attributes['tokenID']).fillna(0)
    attributes_scores.sort_values(by='final score', ascending=False).to_csv('./static/uploads/' + collectionName.replace(" ", "_") + '.csv')

    return {collectionName.replace(" ", "_") : attributes_scores.sort_values(by='final score', ascending=False)}

