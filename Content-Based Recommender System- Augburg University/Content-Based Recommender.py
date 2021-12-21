import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

""" Reading the csv file that would act as a content  to teach the content based recommender model """
df = pd.read_csv("Dept_Activities.csv")

""" This allow the model to ignore any stop English words such as 'a' or 'is' and so on """
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['description'])

""" This will then use cosine similarity lib to run the matrix and see which character matches which position """
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
""" Array used to store the results """
results = {}


"""  This will run though the items and decide to which items matches the model criteria from the CSV file """
for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]

    results[row['id']] = similar_items[1:]
    
print('done!')

""" This only focuses a certain column and use it as the content to study then recommender  """
def item(id):
    return ds.loc[ds['id'] == id]['description'].tolist()[0].split(' - ')[0]

""" Just reads the results out of the dictionary. """
def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")

recommend(item_id=11, num=5)
