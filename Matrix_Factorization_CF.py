import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
data=np.array([0, 0, 5, 0, 1, 4,0, 3, 2, 0, 4, 2, 1, 0, 5, 1,2, 4,1 ,3 ,2,1, 4, 0,2, 0, 2,2, 2, 1,2, 3, 3,
			   2, 4, 4,3, 0, 0,3, 1, 0,3, 3, 4,4 ,0 ,1,4 ,3, 4,5, 1 ,2,5 ,2, 1,6 ,2 ,1,6 ,3 ,4,6, 4 ,5])
data= data.reshape(22,3)
database_org=pd.read_csv("ml-100k\\ua.base",sep="\t").as_matrix()
datatest_org=pd.read_csv("ml-100k\\ua.test",sep="\t").as_matrix()
ml100k_base= database_org[:,:3]
ml100k_test= datatest_org[:,:3]
ml100k_base[:,:2]-=1
ml100k_test[:,:2]-=1
class Matrix_Factorization_CF(object):
    def __init__(self,data,k_feature=3,lr=2,lambda_=.1,limit_iter=100,user_base=True,W_init=None,X_init=None):
        self.user_base=user_base
        self.data = data if user_base else data[:,[1,0,2]]
        #print(self.data)
        self.n_user = int(np.max(self.data[:,0])+1)
        self.n_item = int(np.max(self.data[:,1])+1)
        self.bu = np.random.randn(self.n_user)
        self.bi = np.random.randn(self.n_item)
        self.limit_iter = int(limit_iter)
        self.lr = lr
        self.lambda_ = lambda_
        self.k_feature = k_feature
        self.W = np.random.randn(self.k_feature,self.n_user) if W_init is None else W_init
        self.X = np.random.randn(self.n_item,self.k_feature) if X_init is None else X_init
        self.n_rating = int(len(self.data[:,2]))
        self.normalized()
        self.fit()

    def normalized(self):
        self.data_nmlized= self.data.copy().astype(np.float32)
        self.mean= np.zeros(self.n_user)
        for u in range(self.n_user):
            ids = np.where(self.data[:,0]==u)[0]
            items_israted= self.data[ids,1]
            ratings= self.data[ids,2]
            mean = np.mean(ratings)
            if mean is np.nan:
                mean=0
            self.mean[u]=mean
            self.data_nmlized[ids,2]-=mean

    def add(self,newdata):
        self.data= np.concatenate((self.data,newdata),axis=0)

    def loss(self):
        loss=0
        for u in range(self.n_user):
            items,ratings = self.get_item_rated_by_user(u)
            loss += (np.sum((ratings - np.dot(self.X[items,:],self.W[:,u]))**2)*.5/self.n_rating)
        loss+= .5*self.lambda_*(np.linalg.norm(self.X,'fro') + np.linalg.norm(self.W,'fro'))
        return loss

    def get_item_rated_by_user(self,user_id):
        ids = np.where(self.data_nmlized[:,0]==user_id)[0]
        items= self.data_nmlized[ids,1].astype(np.int32)
        ratings= self.data_nmlized[ids,2]
        return items,ratings

    def get_user_who_rated_item(self,item_id):
        ids = np.where(self.data_nmlized[:,0]==item_id)[0]
        users= self.data_nmlized[ids,0].astype(np.int32)
        ratings= self.data_nmlized[ids,2]
        return users,ratings

    def updateW(self):
        for u in range(self.n_user):
            items,ratings = self.get_item_rated_by_user(u)
            error = np.dot(self.X[items,:],self.W[:,u]) + self.bu[u] + np.mean(self.bi[items]) - ratings
            self.W[:,u] -= (self.lr*np.dot(self.X[items, :].T,error)/self.n_rating + self.lambda_*self.W[:,u])
            self.bu[u] -= (self.lr*np.sum(error)/self.n_rating + self.lambda_*self.bu[u])

    def updateX(self):
        for i in range(self.n_item):
            users,ratings = self.get_user_who_rated_item(i)
            error = np.dot(self.X[i,:],self.W[:, users]) + self.bi[i] + np.mean(self.bu[users]) - ratings
            self.X[i,:] -= (self.lr*np.dot(error, self.W[:, users].T)/self.n_rating + self.lambda_*self.X[i,:])
            self.bi[i] -= (self.lr*np.sum(error)/self.n_rating + self.lambda_*self.bi[i])
    def fit(self):
        for iter_ in range(self.limit_iter):
            self.updateW()
            self.updateX()
            if iter_%5==0:
                loss=self.loss()
                print('Iter:',iter_,"Loss: %.4f"%loss)
                if loss<.01:
                    break

    def __predict__(self,user_id,item_id):
        pred = np.dot(self.X[item_id,:],self.W[:,user_id]) + self.bu[user_id] + self.bi[item_id]
        pred += self.mean[user_id]
        if pred <0: 
            return 0
        elif pred>5:
            return 5
        else:
            return pred

    def predict(self,user_id,item_id):
        if self.user_base:
            return self.__predict__(user_id,item_id)
        return self.__predict__(item_id,user_id)

    def recommend(self,user_id,n_recomment=None):
        pred_list=[]
        recommend_list=[]
        if self.user_base:
            items_israted,ratings= self.get_item_rated_by_user(user_id)
            for item in range(self.n_item):
                if item not in items_israted:
                    pred=self.__predict__(user_id,item)
                    if pred>self.mean[user_id]:
                        pred_list.append(pred)
                        recommend_list.append(item)
        else:
            users,ratings= self.get_user_who_rated_item(user_id)
            for user in range(self.n_user):
                if user not in users:
                    pred= self.__predict__(user,user_id)
                    if pred>self.mean[user]:
                        pred_list.append(pred)
                        recommend_list.append(user)
        if n_recomment is None:
            return np.array(recommend_list),np.array(pred_list)
        pred_list= np.array(pred_list)
        recommend_list= np.array(recommend_list)
        arg= np.argsort(pred_list)[-n_recomment:]
        return recommend_list[arg],pred_list[arg]

    def evaluate_RMSE(self,data_test):
        n_rating= int(len(data[:,2]))
        loss=0
        for r in range(n_rating):
            loss += (self.predict(data_test[r,0],data_test[r,1])-data_test[r,2])**2
        loss/=n_rating
        return np.sqrt(loss)
    def print_recommentdation(self,user_id_list,n_recomment=None):
        for user_id in user_id_list:
            recommend_list,pred_list= self.recommend(user_id,n_recomment)
            print('{:<10} Recomment for user: {:<5} items: {:<35} predicted rating: {:<10}'\
                .format('',str([user_id]),str(recommend_list),str(pred_list)))



CF= Matrix_Factorization_CF(ml100k_base, lr=1, limit_iter=100, lambda_=0.01,user_base=True)
print(CF.recommend(100,n_recomment=2))
print(CF.evaluate_RMSE(ml100k_test))
for i in range(0,100,10):
    print('i:',i)
    print(ml100k_test[i])
    print("mean: ", CF.mean[ml100k_test[i,0]])
    print(CF.predict(ml100k_test[i,0],ml100k_test[i,1]))
