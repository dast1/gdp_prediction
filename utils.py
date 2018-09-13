import numpy as np
import pandas as pd
from functools import wraps
from time import time
import matplotlib.mlab as mlab
import scipy
from scipy.stats import norm
import math
import matplotlib.pyplot as plt

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from textblob import TextBlob, Blobber
from textblob.sentiments import NaiveBayesAnalyzer

from pyramid.arima import auto_arima, ARIMA

def plot_multi(data, xlabel, cols=None, spacing=.1, **kwargs):

    from pandas import plotting

    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    colors = getattr(getattr(plotting, '_style'), '_get_standard_colors')(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)])
        ax_new.set_ylabel(ylabel=cols[n])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label
    
    ax.legend(lines, labels, loc=0)
    ax.grid(color='k', linestyle='--', linewidth=1)
    ax.set_xlabel(xlabel=xlabel)
    return ax
 
    
def norm_dist_test(data, label):
    # Fit a normal distribution to the data
    mu, std = norm.fit(data)
    
    # Plot the histogram.
    plt.hist(data, bins=100, density=True, alpha=0.6, label=[label])
    
    # Plot the PDF
    # xmin, xmax = plt.xlim()
    x = np.linspace(mu - 4*std, mu + 4*std, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, linewidth=2, label=label+' PDF')
    plt.legend()
    
    

class Model:
    def __init__(self):
        self.y_pred = None
        self.y_prob = None
        self.alg = None

    def modelfit(self, alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
        #Fit the algorithm on the data
        alg.fit(dtrain[predictors], dtrain[target])
        self.alg = alg
        
        #Predict training set:
        dtrain_predictions = alg.predict(dtrain[predictors])
        self.y_pred = dtrain_predictions
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        self.y_prob = dtrain_predprob

        #Perform cross-validation:
        if performCV:
            cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=cv_folds, scoring='roc_auc')

        #Print model report:
        print("\nModel Report")
        print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
        print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

        if performCV:
            print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

        #Print Feature Importance:
        if printFeatureImportance:
            feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')
            

            
            
def format_raw_documents(raw_documents):
    '''
    This accepts a list of pandas data frames and returns a single pandas data frame with only `Index`,`Date`,`Title`, and `Content` fields
    '''
    df_list = []
    for raw_doc in raw_documents:
        df = pd.DataFrame()
        try:
            df['Date'] = raw_doc['Date']
            df['Title'] = raw_doc['Heading']
            df['Content'] = raw_doc['Article']
        except:
            df['Date'] = raw_doc['date']
            df['Title'] = raw_doc['title']
            df['Content'] = raw_doc['content']
        
        df_list.append(df)
     
    docs = pd.concat(df_list,axis=0).sort_values('Date').reset_index(drop=True)
    
    # Let's delete the articles with no Dates.
    docs = docs.drop(docs[pd.isnull(docs['Date'])].index)
    
    # Let's delete the articles with no Titles.
    docs = docs.drop(docs[pd.isnull(docs['Title'])].index)
    
    return docs

class myNLP:
    def __init__(self):
        self.stop = set(stopwords.words('english'))
        self.exclude = string.punctuation
        self.lemma = WordNetLemmatizer()
        self.porter = PorterStemmer()
        self.cnt_vect = CountVectorizer(max_df=0.95, min_df=2, max_features=3000)
        self.tfidf_vect = TfidfVectorizer(max_df=0.95, min_df=2, max_features=3000)
        self.tfidf = None
        self.nmf = None
        self.M = None
        self.tfidf = None
        self.W = None
        self.H = None
        
    def prep_docs_stem(self, doc):
        '''
        This function does the following:
        1. lowercases and removes stop words
        2. removes puctuations
        3. lematizes
        4. and finally, return word stems only
        '''
        stop_free = " ".join([i for i in doc.lower().split() if i not in self.stop])
        punc_free = stop_free.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        lematized = " ".join(self.lemma.lemmatize(word) for word in punc_free.split())
        normalized = ' '.join([self.porter.stem(word) for word in lematized.split()])
        return normalized
    
    def prep_docs_lematize(self, doc):
        '''
        This function does the following:
        1. lowercases and removes stop words
        2. removes puctuations
        3. and finally, return word lemmas only
        '''
        stop_free = " ".join([i for i in doc.lower().split() if i not in self.stop])
        punc_free = stop_free.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        normalized = " ".join(self.lemma.lemmatize(word) for word in punc_free.split())
        return normalized
    
    def fit_nmf(self, docs):
        '''
        NMF is able to use tf-idf.
        Return two lists:
        1. list of top 5 topics for each document
        2. list of top 20 words in each topic
        '''
        self.tfidf = self.tfidf_vect.fit_transform(docs)
        self.tfidf_feature_names = self.tfidf_vect.get_feature_names()
        
        self.nmf = NMF(n_components=100, 
                       random_state=1, 
                       alpha=.1, 
                       l1_ratio=.5, 
                       init='nndsvd').fit(self.tfidf)
        
        # get top 5 topics for each document
        self.doc_topic_NMF = self.nmf.transform(self.tfidf)
        top_5_topics = []
        for n in range(self.doc_topic_NMF.shape[0]):
            top_5_topics.append(self.doc_topic_NMF[n].argsort()[::-1][:5])
        self.top_5_topics_NMF = np.array(top_5_topics).T
            
        # get top 20 words for each topic
        self.top_20_stem_words_NMF = []
        topics = self.nmf.components_
        for n in range(topics.shape[0]):
            idx = topics[n].argsort()[::-1][:20]
            top_20_words_n = []
            for i in idx:
                top_20_words_n.append(self.tfidf_feature_names[i])
            self.top_20_stem_words_NMF.append(top_20_words_n)
        
        
        
        
        return self.top_5_topics_NMF, self.top_20_stem_words_NMF, self.nmf, self.tfidf, self.tfidf_vect
    
    
    def fit_lda(self, docs):
        '''
        LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        '''
        self.tf = self.cnt_vect.fit_transform(docs)
        self.tf_feature_names = self.cnt_vect.get_feature_names()
        
        self.lda = LatentDirichletAllocation(n_components=100, 
                                             max_iter=5, 
                                             learning_method='online',
                                             learning_offset=50.,
                                             random_state=0,
                                             n_jobs=-1).fit(self.tf)
        
        # get top 5 topics for each document
        self.doc_topic_LDA = self.lda.transform(self.tf)
        top_5_topics = []
        for n in range(self.doc_topic_LDA.shape[0]):
            top_5_topics.append(self.doc_topic_LDA[n].argsort()[::-1][:5])
        self.top_5_topics_NMF = np.array(top_5_topics).T
        
        # get top 20 words for each topic
        self.top_20_stem_words_LDA = []
        topics = self.lda.components_
        for n in range(topics.shape[0]):
            idx = topics[n].argsort()[::-1][:20]
            top_20_words_n = []
            for i in idx:
                top_20_words_n.append(self.tf_feature_names[i])
            self.top_20_stem_words_LDA.append(top_20_words_n)
        
        return self.top_5_topics_LDA, self.top_20_stem_words_NMF, self.lda, self.tf, self.cnt_vect
        

def merge_2_string_lists(string_list1, string_list2):
    '''
    pretty much does what the name says.
    '''
    return [a + ' ' + b for a, b in zip(string_list1, string_list2)]

def scree_plot(ax, pca, n_components_to_plot=8, title=None):
    """Make a scree plot showing the variance explained (i.e. variance
    of the projections) for the principal components in a fit sklearn
    PCA object.
    
    Parameters
    ----------
    ax: matplotlib.axis object
      The axis to make the scree plot on.
      
    pca: sklearn.decomposition.PCA object.
      A fit PCA object.
      
    n_components_to_plot: int
      The number of principal components to display in the scree plot.
      
    title: str
      A title for the scree plot.
    """
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    ax.plot(ind, vals, color='blue')
    ax.scatter(ind, vals, color='blue', s=50)

    for i in range(num_components):
        ax.annotate(r"{:2.2f}%".format(vals[i]), 
               (ind[i]+0.2, vals[i]+0.005), 
               va="bottom", 
               ha="center", 
               fontsize=12)

    ax.set_xticklabels(ind, fontsize=12)
    ax.set_ylim(0, max(vals) + 0.05)
    ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=16)
        
def get_sentiment(text):
    tb = Blobber(text, analyzer=NaiveBayesAnalyzer())
    return opinion.sentiment.p_pos, opinion.polarity, opinion.subjectivity

def add_top_5_topics(df, top_5_topics, NLP_algo_name):
    '''
    Adds top 5 topics in separate columns to the `docs` dataframe.
    '''
    for i, n in enumerate(range(1,6)):
        col_name = 'Top #{} topic ({})'.format(str(n),NLP_algo_name)
        df[col_name] = np.array(top_5_topics)[i].T
    return df

def clean_ff(ff_in):
    ff_out = pd.DataFrame()
    ff_out['Date'] = pd.to_datetime(ff_in['DATE'], infer_datetime_format=True)
    ff_out['FedFundsRate'] = ff_in['FEDFUNDS']
    ff_out.set_index('Date', drop=True, inplace=True)
    return ff_out

def clean_gdp(gdp_in):
    gdp_out = pd.DataFrame()
    gdp_out['Date'] = pd.to_datetime(gdp_in['DATE'], infer_datetime_format=True)
    gdp_out['GDP'] = gdp_in['GDPC1']
    gdp_out.set_index('Date', drop=True, inplace=True)
    return gdp_out

def grid_search_SARIMA(timeseries):
    stepwise_model = auto_arima(timeseries, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
    print(stepwise_model.aic())
    return stepwise_model

def plot_resid_SARIMA(arima_model, gdp):
    try:
        residuals = arima_model.resid()
    except:
        residuals = arima_model.resid
    # dates=gdp.index[len(gdp)-len(residuals):]
    dates = gdp.index
    df = pd.DataFrame(data=residuals,index=dates,columns=['residuals'])
    df.plot(figsize=(12,6), grid=True, legend=None)
    plt.title('ARMA Fit Residual Error Line Plot')
    plt.show()

    df.plot(kind='kde', figsize=(12,6), grid=True, legend=None)
    plt.title('ARMA Fit Residual Error Density Plot')
    plt.show()
    print(df.describe())
    
def one_ste_ahead_forecast(arima_model, timeseries, start_yr, forecast_start):
    pred = arima_model.get_prediction(start=pd.to_datetime(forecast_start), dynamic=False)
    pred_ci = pred.conf_int()

    # plot
    ax = timeseries[start_yr:].plot(label='observed', figsize=(15,10), logy=False)
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('GDP')
    plt.legend()
    plt.show()
    
def dynamic_forecast(arima_model, timeseries, start_yr, forecast_start):
    pred_dynamic = arima_model.get_prediction(start=pd.to_datetime(forecast_start), dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()

    # plot
    ax = timeseries[start_yr:].plot(label='observed', figsize=(15, 10))
    pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(forecast_start), timeseries.index[-1],
                     alpha=.1, zorder=-1)

    ax.set_xlabel('Date')
    ax.set_ylabel('GDP')

    plt.legend()
    plt.show()
    
def validate_topics(docs, top_words,top_topic_num, NLP_algo, random_topic_num = np.random.randint(0,100)):
    random_topic = docs[docs['Top #{} topic ({})'.format(top_topic_num, NLP_algo)] == random_topic_num]
    print('Random Topic #: ',random_topic_num)
    print('Top words in topic: ', top_words.iloc[random_topic_num].tolist())
    print()
    print('Article 1:')
    print(random_topic['Content'].iloc[np.random.randint(0,random_topic.shape[0])])
    print()
    print('Article 2:')
    print(random_topic['Content'].iloc[np.random.randint(0,random_topic.shape[0])])
    print()
    print('Article 3:')
    print(random_topic['Content'].iloc[np.random.randint(0,random_topic.shape[0])])
    
def norm_pdf(y):
    mu = np.mean(y)
    sigma = np.std(y)
    x = np.linspace(mu - 6*sigma, mu + 6*sigma, 100)

    np.random.seed(12)
    norm_pdf = scipy.stats.norm.pdf(x, mu, sigma)
    return norm_pdf    