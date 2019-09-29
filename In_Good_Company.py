#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/python

import sys, getopt, os
import pandas as pd
import datetime
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import Pool

def main(argv):
    script_path = os.path.dirname(os.path.realpath(__file__))
    newsfile = ''
    ratedfile = ''
    try:
        opts, args = getopt.getopt(argv,"hn:r:",["nfile=","rfile="])
    except getopt.GetoptError:
        print('ERROR! Must run as: In_Good_Company.py -n <news_json_file_path> -r <rated_json_file_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('ERROR! Must run as: In_Good_Company.py -n <news_json_file_path> -r <rated_json_file_path>')
            sys.exit()
        elif opt == "-n":
            newsfile = arg
        elif opt == "-r":
            ratedfile = arg
    if (newsfile == '') or (ratedfile == ''):
        print('ERROR! Must run as: In_Good_Company.py -n <news_json_file_path> -r <rated_json_file_path>')
        sys.exit(2)
    
    news = pd.read_json(newsfile)
    companies_to_rate = pd.read_json(ratedfile)

    # predict the theme of each news object - Social, Environment or Governance

    vectorizer = joblib.load(os.path.join(script_path ,'Vontobelvectorizer.pkl'))
    desc_vectors = vectorizer.transform(news.description.values)

    theme_predictor = joblib.load(os.path.join(script_path ,'xgboost_ThemePredictor.pkl'))
    news['theme'] = theme_predictor.predict(desc_vectors)
    theme_dict = {0: 'Environment', 1: 'Governance', 2: 'Social'}
    news.replace({'theme': theme_dict}, inplace=True)

    news[['day', 'month', 'year']] = news['incident_date'].str.split('.', expand=True).astype(int)

    # predict the category of the severity of the news item - between 0 to 5

    precition_cols = ['theme', 'number_incidents_in_chain', 'incident_type', 'source', 'location', 'sector', 'year', 'month']
    X = news[precition_cols]
    labeled_X = X.loc[:, (X.columns != 'number_incidents_in_chain')].apply(LabelEncoder().fit_transform)
    labeled_X['number_incidents_in_chain'] = X['number_incidents_in_chain']
    labeled_X['is_weekend'] = news.apply(lambda row: datetime.datetime(row['year'], row['month'], row['day']).weekday() in [5,6], axis=1)

    labeled_X.head()
    cat_features = [0, 1, 2, 3, 4]

    dataset_to_predict = Pool(data=labeled_X,
                              cat_features=cat_features)

    answer_category_predictor = joblib.load(os.path.join(script_path ,'catboost_CategoryPredictor.pkl'))
    news['answer_category'] = answer_category_predictor.predict(dataset_to_predict)

    # predict the sustainability class of each company based on the news, their theme and severity category

    full_category_list = ['Environment_Category 0', 'Environment_Category 1', 'Environment_Category 2', 'Environment_Category 3',
                          'Environment_Category 4', 'Environment_Category 5', 'Governance_Category 0', 'Governance_Category 1',
                          'Governance_Category 2', 'Governance_Category 3', 'Governance_Category 4', 'Governance_Category 5', 
                          'Social_Category 0', 'Social_Category 1', 'Social_Category 2', 'Social_Category 3', 
                          'Social_Category 4', 'Social_Category 5']

    categorized_news = news.groupby(['company', 'theme', 'answer_category']).size().reset_index()
    categorized_news['theme_category'] = categorized_news[['theme', 'answer_category']].apply(lambda x: '_'.join(x), axis=1)
    categorized_news.drop(['answer_category', 'theme'], axis = 1, inplace=True)
    categorized_news = categorized_news.pivot(index='company', columns='theme_category', values=0)
    categorized_news.fillna(0, inplace=True)
    categorized_news.reset_index(inplace=True)
    for col in full_category_list:
        if col not in categorized_news.columns.tolist():
            categorized_news[col] = 0
    categorized_news = categorized_news.reindex(sorted(categorized_news.columns), axis=1)

    sustainavility_predictor = joblib.load(os.path.join(script_path ,'xgboost_CategorizedSustainability.pkl'))
    predicted_sustainability = pd.DataFrame()
    predicted_sustainability['company'] = categorized_news['company']
    predicted_sustainability['sustainability_class'] = sustainavility_predictor.predict(categorized_news.drop(['company'], axis=1))
    predicted_sustainability.to_csv('predicted_sustainability.csv')
    print(predicted_sustainability)
    
if __name__ == "__main__":
    main(sys.argv[1:])
    

