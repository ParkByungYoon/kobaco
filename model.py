from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class DecisionTree:

    def __init__(self, pivot_df, embedding, k=15) -> None:
        self.df = pivot_df
        self.embedding = embedding

        self.alpha = 0.0
        self.max_depth = None
        self.k = k

        X = np.array(pivot_df.iloc[:,:])
        Y = np.array(self.kmeans(k))

        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(X, Y, test_size=0.1, random_state=42)
        self.X = self.x_train
        self.Y = self.y_train

        self.feature_names = self.df.columns.tolist()[:]
        self.class_names = [str(i) for i in list(self.Y)]


    def kmeans(self, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        embedding = self.embedding[self.df.index.values.tolist()].copy()
        kmeans.fit(embedding)
        return kmeans.predict(embedding)
    

    def kmeans_target(self, tgt_n): 
        self.tgt_n = tgt_n
        self.X = self.x_train
        self.Y = (self.y_train==self.tgt_n).astype(int)
    

    def make_dt(self, ccp_alpha=0.0, max_depth=None, random_state = 42):
        model = DecisionTreeClassifier(ccp_alpha = ccp_alpha, max_depth = max_depth, random_state=random_state)
        model = model.fit(self.X, self.Y)
        return model
    

    def get_valid_score(self, model, scoring, average='macro'):
        self.X = self.x_valid
        if model.n_classes_ == 2:
            self.Y = (self.y_valid==self.tgt_n).astype(int)
        else:
            self.Y = self.y_valid
        return self.get_score(model, scoring, average)


    def get_score(self, model, scoring, average='macro'):
        prediction = model.predict(self.X)
        if model.n_classes_ == 2:   
            average='binary'
        
        recall = recall_score(self.Y, prediction, average=average)
        precision = precision_score(self.Y, prediction, average=average)
        f1 = f1_score(self.Y, prediction, average=average)

        if scoring == 'recall': return recall
        elif scoring == 'precision': return precision
        elif scoring == 'f1_score': return f1
        else:   return recall, precision, f1


    def get_proper_depth(self, target_score, scoring='f1_score'):
        self.max_depth = self.make_dt().get_depth()
        
        passed_depths = []
        score_list = []
        i = 0

        for depth in range(self.max_depth, 1, -1): #이분탐색 개선가능
            if depth % 10 == 0:
                print(f'testing depth {depth}...', end='\r')
            model = self.make_dt(max_depth = depth)
            score = self.get_score(model, scoring)
            
            passed_depths.append(depth)
            score_list.append(score)

            if (i == 0) and (score < target_score):
                raise Exception('너무 높은 target f1 설정, 달성 불가능')

            if (score < target_score) and (i > 0): #목표 f1값 아래로 떨어지면
                # 이전 max depth가 임계치를 넘는 최소 depth
                self.max_depth = passed_depths[-2]
                self.max_score = score_list[-2]
                
                plt.plot(passed_depths,score_list)
                plt.xlabel('depth')
                plt.ylabel('fl-score')
                plt.hlines(y=target_score, xmin = passed_depths[-1], xmax=passed_depths[0],colors='r')
                plt.show()

                self.max_depth_dt = self.make_dt(max_depth = passed_depths[-2])
                return passed_depths[-2], score_list[-2]
                
            i += 1
        raise Exception('너무 작은 target f1')
    

    def get_all_depth(self, scoring='all'):
        self.max_depth = self.make_dt().get_depth()
    
        score_list = []

        for depth in range(1, self.max_depth):
            if depth % 10 == 0:
                print(f'testing depth {depth}...', end='\r')
            model = self.make_dt(max_depth = depth)
            score = self.get_score(model, scoring)
            score_list.append(score)
        
        return score_list