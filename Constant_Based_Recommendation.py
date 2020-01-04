import pandas as pd
import  numpy as np

genre = pd.read_csv('ml-100k\\u.genre',sep='|')
genre= genre.values[:,0]
user_info_col = ['id','age','sex','job','zip']
user = pd.read_csv('ml-100k\\u.user',header=None,sep='|',names=user_info_col)
data_train = pd.read_csv('ml-100k\\ua.base',sep='\t',header=None, names=['user_id','movie_id','rate','time']).as_matrix()
data_test = pd.read_csv('ml-100k\\ua.test',sep='\t',header=None, names=['user_id','movie_id','rate','time']).as_matrix()
item_col = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
            'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k\\u.item',names=item_col,sep='|',encoding='latin-1')
items = items.values[:,-19:]
n_user = user.shape[0]
n_items = items.shape[0]
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True, norm ='l2')
tfidf = transformer.fit_transform(items.tolist()).toarray()

def get_item_rate_by_user(matrix_rate,user_id):
    index = np.where(matrix_rate[:,0]==user_id)[0]
    items_id = matrix_rate[index,1]
    score= matrix_rate[index,2]
    return items_id, score
from sklearn import  linear_model
W= np.zeros((tfidf.shape[1],n_user))
b=np.zeros((1,n_user))
for n in range(n_user):
    items_id, score= get_item_rate_by_user(data_train,n+1)
    rigde= linear_model.Ridge(fit_intercept=True,alpha=0.01)
    rigde.fit(tfidf[items_id-1,:],score)
    W[:,n]=rigde.coef_
    b[0,n] = rigde.intercept_
def prefict(user_id,movie_id):
    result = np.dot(tfidf[movie_id-1,:],W[:,user_id-1]) + b[:,user_id-1]
    return result

def recommend(user_id, movie_id):
	pred = prefict(user_id, movie_id)
	print("Predicted user %d "%user_id +'rate item %d: '%movie_id +'%.2f sao'%pred)

recommend(1,5)
