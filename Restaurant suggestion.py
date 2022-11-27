#Importing Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import flask

app = flask.Flask(__name__, template_folder='templates')

orders = pd.read_csv('C:/Users/Test/Downloads/orders data.csv')
vendor = pd.read_csv('C:/Users/Test/Downloads/vendor_data.csv')

vendor_new = vendor.copy().drop(['dish_liked','phone'],axis=1)


vendor_new.duplicated().sum()


vendor_new.isnull().sum()


vendor_new = vendor_new.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',
                                  'listed_in(city)':'city'})



#Some Transformations
vendor_new['cost'] = vendor_new['cost'].astype(str) #Changing the cost to string
vendor_new['cost'] = vendor_new['cost'].apply(lambda x: x.replace(',','.')) #Using lambda function to replace ',' from cost
vendor_new['cost'] = vendor_new['cost'].astype(float) # Changing the cost to Float
vendor_new.info()


vendor_new['rate'].unique()

#Removing '/5' from Rates
vendor_new = vendor_new.loc[vendor_new.rate !='NEW']
vendor_new = vendor_new.loc[vendor_new.rate !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
vendor_new.rate = vendor_new.rate.apply(remove_slash).str.strip().astype('float')
vendor_new['rate'].head()

# Adjust the column names
vendor_new.name = vendor_new.name.apply(lambda x:x.title())
vendor_new.online_order.replace(('Yes','No'),(True, False),inplace=True)
vendor_new.book_table.replace(('Yes','No'),(True, False),inplace=True)


## Computing Mean Rating
restaurants = list(vendor_new['name'].unique())
vendor_new['Mean Rating'] = 0

for i in range(len(restaurants)):
    vendor_new['Mean Rating'][vendor_new['name'] == restaurants[i]] = vendor_new['rate'][vendor_new['name'] == restaurants[i]].mean()


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (1,5))

vendor_new[['Mean Rating']] = scaler.fit_transform(vendor_new[['Mean Rating']]).round(2)


# 5 examples of these columns before text processing:
vendor_new[['reviews_list', 'cuisines']].sample(5)



## Lower Casing
vendor_new["reviews_list"] = vendor_new["reviews_list"].str.lower()
vendor_new[['reviews_list', 'cuisines']].sample(5)

## Removal of Puctuations
import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

vendor_new["reviews_list"] = vendor_new["reviews_list"].apply(lambda text: remove_punctuation(text))
vendor_new[['reviews_list', 'cuisines']].sample(5)

STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

vendor_new["reviews_list"] = vendor_new["reviews_list"].apply(lambda text: remove_stopwords(text))


## Removal of URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

vendor_new["reviews_list"] = vendor_new["reviews_list"].apply(lambda text: remove_urls(text))


vendor_new[['reviews_list', 'cuisines']].sample(5)

# RESTAURANT NAMES:
restaurant_names = list(vendor_new['name'].unique())

def get_top_words(column, top_nu_of_words, nu_of_word):
    
    vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')
    
    bag_of_words = vec.fit_transform(column)
    
    sum_words = bag_of_words.sum(axis=0)
    
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return words_freq[:top_nu_of_words]

vendor_new=vendor_new.drop(['address','rest_type', 'type', 'menu_item', 'votes'],axis=1)


# Creating tf-idf matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(vendor_new['reviews_list'])


cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(vendor_new.index, index=vendor_new['name']).drop_duplicates()
def get_recommendations(title):
    global sim_scores
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_similarities[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # return list of similar movies
    return_df = pd.DataFrame(columns=['Title','Homepage'])
    return_df['Title'] = vendor_new['name'].iloc[movie_indices]
    return_df['Homepage'] = vendor_new['url'].iloc[movie_indices]
    return_df['Cuisines'] = vendor_new['cuisines'].iloc[movie_indices]
    return return_df


# create array with all movie titles
all_titles = [vendor_new['name'][i] for i in range(len(vendor_new['name']))]

# Set up the main route
@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
            
    if flask.request.method == 'POST':
        m_name = " ".join(flask.request.form['movie_name'].split())
#check = difflib.get_close_matches(m_name,all_titles,cutout=0.50,n=1)
        if m_name not in all_titles:
            return(flask.render_template('notFound.html',name=m_name))
        else:
            result_final = get_recommendations(m_name)
            names = []
            homepage = []
            cuisines = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                cuisines.append(result_final.iloc[i][2])
                if(len(str(result_final.iloc[i][1]))>3):
                    homepage.append(result_final.iloc[i][1])
                else:
                    homepage.append("#")
                

            return flask.render_template('found.html',movie_names=names,movie_homepage=homepage,search_name=m_name, movie_releaseDate=cuisines, movie_simScore=sim_scores)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080)
    #app.run()