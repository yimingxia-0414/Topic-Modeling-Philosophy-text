'''
Filename: functions.py
Author: Yiming Xia
'''

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from collections import defaultdict
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import word2vec  
from sklearn.manifold import TSNE
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import copy


def remove_stopwords(word_tokenized):
    '''Remove stop words.'''
    return [word for word in word_tokenized if not word.lower() in stopwords.words('english')]

def lemmatize_sentence(word_tokenized):
    '''Lemmatize tokenized sentence.'''
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lmtzr = WordNetLemmatizer()
    return [lmtzr.lemmatize(token, tag_map[tag[0]]) for token, tag in pos_tag(word_tokenized)]

def lemmatized_sentence(word_tokenized):
    '''Combine steps of removing stop words and lemmatizing sentences.'''
    word_remove_stopwords = remove_stopwords(word_tokenized)
    word_lemmatized = lemmatize_sentence(word_remove_stopwords)
    return " ".join(word_lemmatized)


def plot_wordcloud(df,school,maxword):
    '''Plot wordcloud'''
    word = df.loc[df["school"] == school, ["lemmatized_str"]].values

    wordcloud = WordCloud(max_font_size=60, 
                          max_words=maxword,
                          background_color="white",
                          scale=6,
                          random_state=66,
                          relative_scaling=.5).generate(str(word))

    # plot:
    fig=plt.figure(figsize = (12, 12), facecolor = None)
    plt.title(school, fontsize=40) 
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()

    # save image
    fig.savefig('/Users/xiayiming/Documents/GitHub/spring-2022-prj1-yimingxia-0414/figs/wordcloud-{}.png'.format(school), bbox_inches="tight")




def display_topics(model, feature_names, no_top_words):
    '''For generating topics'''
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)

def generate_topics(df,school):
    '''Generate topics'''
    test=df.loc[df["school"] == school]
    coun_vect = CountVectorizer(max_df=0.9, min_df=25)
    count_matrix = coun_vect.fit_transform(test['lemmatized_str'])
    count_array = count_matrix.toarray()
    test_df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names())
  
    # modeling
    number_of_topics = 10
    model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)
    model.fit(test_df)
    
    #generate topics
    no_top_words = 10
    s=display_topics(model, coun_vect.get_feature_names(), no_top_words)
    s.to_json('/Users/xiayiming/Documents/GitHub/spring-2022-prj1-yimingxia-0414/output/topics-{}.json'.format(school))
    
    return s



def build_corpus(data):  
    '''Creates a list of lists containing words per sentence'''
    corpus = []  
    for sentence in data:  
        word_list = sentence.split(" ")  
        corpus.append(word_list)   

    return corpus 

def tsne_plot(model):

    labels = []
    tokens = []

    for word in model.wv.key_to_index:
        tokens.append(model.wv[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        

def tsne_plot_school(df,school1,school2='Null'):
    if school2=='Null':
        test=df[(df['school']==school1)]
    else:
        test=df[(df['school']==school1) | (df['school']==school2)]
    
    corpus = build_corpus(test['lemmatized_str'])
    model = word2vec.Word2Vec(corpus, vector_size=100, window=5, min_count=1000, workers=4)
    tsne_plot(model)
    # save image
    plt.savefig('/Users/xiayiming/Documents/GitHub/spring-2022-prj1-yimingxia-0414/figs/tsne-{}.png'.format(school1+'_'+school2), bbox_inches="tight")
    plt.show()

    
def sentiment_generate(df,schools):
    '''For sentiment analysis'''
    analyzer = SentimentIntensityAnalyzer()
    test_df=pd.DataFrame()
    for school in schools:
        test = copy.deepcopy(df.loc[df["school"] == school])
        test['compound'] = test['sentence_lowered'].apply(lambda x:analyzer.polarity_scores(x)['compound'])
        test['sent_classification'] = test['compound'].apply(lambda x: 'positive' if x>=0.05 else 'negative' if x <=-0.05 else 'neutral')
        test_new=pd.DataFrame({school:test.groupby('sent_classification')['compound'].count()/test.shape[0]})
        test_df=test_df.append(test_new.T)
    test_df['school']=test_df.index
    test_df.plot(x='school', kind='bar', stacked=True,
        title='Stacked Bar Graph by dataframe')
        # save image
    plt.savefig('/Users/xiayiming/Documents/GitHub/spring-2022-prj1-yimingxia-0414/figs/sentiment-{}.png'.format(schools), bbox_inches="tight")