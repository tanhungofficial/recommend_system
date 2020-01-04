import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
data=np.array([0, 0, 5, 0, 1, 4,0, 3, 2, 0, 4, 2, 1, 0, 5, 1,2, 4,1 ,3 ,2,1, 4, 0,2, 0, 2,2, 2, 1,2, 3, 3,
			   2, 4, 4,3, 0, 0,3, 1, 0,3, 3, 4,4 ,0 ,1,4 ,3, 4,5, 1 ,2,5 ,2, 1,6 ,2 ,1,6 ,3 ,4,6, 4 ,5])
data= data.reshape(22,3)
database_org=pd.read_csv("ml-100k\\ua.base",sep="\t").as_matrix()
datatest_org=pd.read_csv("ml-100k\\ua.test",sep="\t").as_matrix()
database= database_org[:,:3]
datatest= datatest_org[:,:3]
database[:,:2]-=1
datatest[:,:2]-=1
class CollaborativeFiltering(object):
    def __init__(self,data,k_similarity,similarity_function=cosine_similarity,uuCF=False):
    	#Neu uuCF=1 thi la la user rate item, nguoc lai la item rate user
        self.data= data if uuCF else data[:,[1,0,2]]
        self.n_user = np.max(self.data[:,0])+1
        self.n_item = np.max(self.data[:,1])+1
        self.k= k_similarity
        self.similarity_function= similarity_function
        self.uuCF=uuCF

    def add(self,new_data):
        self.data= np.concatenate((self.data,new_data),axis=0)
    def noramlize(self):
        self.data_noramlize= self.data.copy().astype(np.float32)
        self.mean_ratings= np.zeros((self.n_user,))
        for u in range(self.n_user):
        	#
            ids= np.where(self.data[:,0]==u)[0]
            item_rated_by_u= self.data[ids,1]
            ratings= self.data[ids,2]
            self.mean_ratings[u] = np.mean(ratings)
            if self.mean_ratings[u]== np.nan:
            	mean_ratings[u]=0
            self.data_noramlize[ids,2]-= self.mean_ratings[u]
        self.utility_matrix= coo_matrix((self.data_noramlize[:,2],(self.data_noramlize[:,1],self.data_noramlize[:,0])),
								shape=(self.n_item,self.n_user)).toarray()
    def similarity(self):
    	self.similarity_matrix = self.similarity_function(self.utility_matrix.T,self.utility_matrix.T)
    def refresh(self):
    	self.noramlize()
    	self.similarity()
    def fit(self):
    	self.refresh()
    def __predict(self,user_id,item_id, noramlize=True):
    	ids= np.where(self.data[:,1]==item_id)[0]
    	users_rated= self.data[ids,0]
    	sim_vector_user_rated = self.similarity_matrix[user_id,users_rated]
    	arg = np.argsort(sim_vector_user_rated)[-self.k:]
    	#k user co similarity lon nhat voi user can predict
    	nearest_sim = sim_vector_user_rated[arg]
    	#rating cua nhung user co similarity cao nhat for iterm can predict
    	rating_of_nearest= self.utility_matrix[item_id,users_rated[arg]]
    	if noramlize:
    		return (np.dot(nearest_sim,rating_of_nearest))/(np.sum(np.abs(nearest_sim))+1e-8)
    	else:
    		return (np.dot(nearest_sim,rating_of_nearest))/(np.sum(np.abs(nearest_sim))+1e-8) + self.mean_ratings[user_id]
    def predict(self,user_id,item_id,noramlize=True):
    	if self.uuCF: return self.__predict(user_id,item_id,noramlize)
    	return self.__predict(item_id,user_id,noramlize)
    def recomment(self,user_id,noramlize=True,n_recomment=None):
        pred_list = []
        recomment_list = []
        if self.uuCF:
            ids= np.where(self.data[:,0]==user_id)[0]
            item_rated_by_u = self.data[ids, 1]

            for item_id in range(self.n_item):
                if item_id not in item_rated_by_u:
                    pred=  self.predict(user_id,item_id,noramlize=noramlize)
                    if pred>0:
                        recomment_list.append(item_id)
                        pred_list.append(pred)
        else:
            ids= np.where(self.data[:,1]==user_id)[0]
            item_rated_by_u= self.data[ids,0]
            for item_id in range(self.n_user):
                if item_id not in item_rated_by_u:
                    pred= self.predict(user_id,item_id)
                    if pred>0:
                        recomment_list.append(item_id)
                        pred_list.append(pred)
        if n_recomment == None:
            arg = np.argsort(pred_list)
        else:
            arg = np.argsort(pred_list)[-n_recomment:]
        pred_list = np.array(pred_list)[arg]
        recomment_list = np.array(recomment_list)[arg]
        return recomment_list, pred_list
    def display_recommentdation(self,list_user_id,n_recomment=None, noramlize=True):
        str_="User-user Collaborative Filtering\n" if self.uuCF else "Item-item Collaborative Filtering\n"
        print("\nRECOMMENTDATION:",str_)
        for user_id in list_user_id:
            recomment_list, pred_list = self.recomment(user_id,n_recomment=n_recomment,noramlize=noramlize)
            print("{:^5}{:<20}{:<5} items:{:<15} predicted rating: {:<5}"
                  .format("","Recomment for user:",str([user_id]),str(recomment_list),str(pred_list)))

CF= CollaborativeFiltering(data,k_similarity=2,uuCF=True)
CF.fit()
print(CF.recomment(1,n_recomment=10))
CF.display_recommentdation([0,1,2,3,4,5,6],n_recomment=5,noramlize=False)

