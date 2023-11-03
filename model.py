from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, _tree, export_graphviz, export_text
from sklearn import tree
import graphviz
import pandas as pd
import numpy as np
import pickle
import warnings   
import os 
import matplotlib.pyplot as plt
import re
from sklearn.metrics import recall_score, precision_score, f1_score

warnings.filterwarnings(action='ignore')


class DecisionTree:

    def __init__(self, pivot_df, embedding, k=15) -> None:
        self.df = pivot_df
        self.embedding = embedding

        self.alpha = 0.0
        self.max_depth = None
        self.k = k

        self.kmeans(k)

        self.X = np.array(pivot_df.iloc[:,:])
        self.Y = np.array(self.label)

        self.feature_names = self.df.columns.tolist()[:]
        self.class_names = [str(i) for i in list(self.Y)]


    def kmeans(self, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        embedding = self.embedding[self.df.index.values.tolist()].copy()
        kmeans.fit(embedding)
        self.label = kmeans.predict(embedding)
    

    def kmeans_target(self, tgt_n): 
        self.Y = (self.label==tgt_n).astype(int)
    

    def make_dt(self, ccp_alpha=0.0, max_depth=None, random_state = 42):
        model = DecisionTreeClassifier(ccp_alpha = ccp_alpha, max_depth = max_depth, random_state=random_state)
        model = model.fit(self.X, self.Y)
        return model


    def get_score(self, model, scoring):
        prediction = model.predict(self.X)
        
        if scoring == 'recall':
            score = recall_score(self.Y, prediction)
        elif scoring == 'precision':
            score = precision_score(self.Y, prediction)
        elif scoring == 'f1_score':
            score = f1_score(self.Y, prediction, average='micro')
        
        return score


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

                self.max_depth_dt = model
                return passed_depths[-2], score_list[-2]
                
            i += 1
        raise Exception('너무 작은 target f1')