from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split


class DecisionTree:

    def __init__(self, pivot_df, embedding, validation=False, k=15, hierarchical=False) -> None:
        self.df = pivot_df
        self.embedding = embedding

        self.alpha = 0.0
        self.max_depth = None
        self.k = k
        self.validation = validation

        X = np.array(pivot_df.iloc[:,:])
        if hierarchical:
            Y = np.array(self.agglomerative_clustering(k))
        else :
            Y = np.array(self.kmeans(k))

        if validation:
            self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(X, Y, test_size=0.1, random_state=42)
        else:
            self.x_train, self.y_train = X, Y
        
        self.X = self.x_train
        self.Y = self.y_train

        self.feature_names = self.df.columns.tolist()[:]
        self.class_names = [str(i) for i in list(self.Y)]


    def kmeans(self, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        embedding = self.embedding[self.df.index.values.tolist()].copy()
        kmeans.fit(embedding)
        return kmeans.predict(embedding)
    
    def agglomerative_clustering(self, k):
        agg_cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
        embedding = self.embedding[self.df.index.values.tolist()].copy()
        return agg_cluster.fit_predict(embedding)
    

    def kmeans_target(self, tgt_n): 
        self.tgt_n = tgt_n
        self.X = self.x_train
        self.Y = (self.y_train==self.tgt_n).astype(int)
    

    def make_dt(self, ccp_alpha=0.0, min_samples_split=2, max_depth=None, min_impurity_decrease=0, min_samples_leaf=1, random_state = 42):
        model = DecisionTreeClassifier(
            ccp_alpha = ccp_alpha, 
            max_depth = max_depth, 
            random_state=random_state, 
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_leaf=min_samples_leaf,
        )
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
    

    def get_all_depth(self, scoring='all', visualize=True):
        self.max_depth = self.make_dt().get_depth()
    
        score_list = []
        val_score_list = []


        for depth in range(1, self.max_depth):
            if self.validation:
                self.X = self.x_train
                if len(np.unique(self.Y)) == 2:
                    self.Y = (self.y_train==self.tgt_n).astype(int)
                else:
                    self.Y = self.y_train

            model = self.make_dt(max_depth = depth)

            score = self.get_score(model, scoring)
            score_list.append(score)

            if self.validation:
                val_score = self.get_valid_score(model, scoring)
                val_score_list.append(val_score)

            if visualize:   
                self.visualize_tree(model)
        
        if self.validation:
            return score_list, val_score_list
        else:
            return score_list
    

    def visualize_tree(self, model):
        plt.figure(figsize=(70, 50))
        plot_tree(model, feature_names=self.feature_names, class_names=self.class_names, filled=True)
        plt.show()



    def get_all_split(self, scoring='f1_score', visualize=True):
        self.max_depth = 30
        
        passed_mss = []
        score_list = []
        depth_list = []

        for mss in range(2, 30): #이분탐색 개선가능
            model = self.make_dt(max_depth = self.max_depth, min_samples_split=mss)
            score = self.get_score(model, scoring)
            depth = model.get_depth()
            
            passed_mss.append(mss)
            score_list.append(score)
            depth_list.append(depth)
        
            if visualize:   
                self.visualize_tree(model)

        return passed_mss, score_list, depth_list

    def get_all_impurity(self, scoring='f1_score'):
        self.max_depth = 30
        
        passed_mid = []
        score_list = []
        depth_list = []

        for mid in range(10):
            mid = mid/10000
            model = self.make_dt(max_depth = self.max_depth, min_impurity_decrease=mid)
            score = self.get_score(model, scoring)
            depth = model.get_depth()
            
            passed_mid.append(mid)
            score_list.append(score)
            depth_list.append(depth)

        return passed_mid, score_list, depth_list
    



    def get_proper_min_sample_split(self, target_score, scoring='f1_score'):
        self.max_depth = self.make_dt().get_depth()
        
        passed_mss = []
        score_list = []
        i = 0

        for mss in range(2, 30): #이분탐색 개선가능
            model = self.make_dt(max_depth = self.max_depth, min_samples_split=mss)
            score = self.get_score(model, scoring)
            
            passed_mss.append(mss)
            score_list.append(score)

            if (i == 0) and (score < target_score):
                raise Exception('너무 높은 target f1 설정, 달성 불가능')

            if (score < target_score) and (i > 0): #목표 f1값 아래로 떨어지면
                # 이전 max depth가 임계치를 넘는 최소 depth
                # self.max_depth = passed_mss[-2]
                self.max_score = score_list[-2]
                
                plt.plot(passed_mss,score_list)
                plt.xlabel('min sample split')
                plt.ylabel('fl-score')
                plt.hlines(y=target_score, xmin = passed_mss[-1], xmax=passed_mss[0],colors='r')
                plt.show()

                self.max_depth_dt = self.make_dt(max_depth = self.max_depth, min_samples_split=passed_mss[-2])
                return passed_mss[-2], score_list[-2]
                
            i += 1
        raise Exception('너무 작은 target f1')


    def get_proper_min_impurity_decrease(self, target_score, scoring='f1_score'):
        self.max_depth = self.make_dt().get_depth()
        
        passed_mid = []
        score_list = []
        i = 0

        for mid in range(10):
            mid = mid/10000
            model = self.make_dt(max_depth = self.max_depth, min_impurity_decrease=mid)
            score = self.get_score(model, scoring)
            
            passed_mid.append(mid)
            score_list.append(score)

            if (i == 0) and (score < target_score):
                raise Exception('너무 높은 target f1 설정, 달성 불가능')

            if (score < target_score) and (i > 0): #목표 f1값 아래로 떨어지면
                # 이전 max depth가 임계치를 넘는 최소 depth
                # self.max_depth = passed_mid[-2]
                self.max_score = score_list[-2]
                
                plt.plot(passed_mid,score_list)
                plt.xlabel('depth')
                plt.ylabel('min impurity decrease')
                plt.hlines(y=target_score, xmin = passed_mid[-1], xmax=passed_mid[0],colors='r')
                plt.show()

                self.max_depth_dt = self.make_dt(max_depth = self.max_depth, min_impurity_decrease=passed_mid[-2])
                return passed_mid[-2], score_list[-2]
                
            i += 1
        raise Exception('너무 작은 target f1')
    

    def get_proper_min_sample_split(self, target_score, scoring='f1_score'):
        self.max_depth = self.make_dt().get_depth()
        
        passed_mss = []
        score_list = []
        i = 0

        for mss in range(2, 30): #이분탐색 개선가능
            model = self.make_dt(max_depth = self.max_depth, min_samples_split=mss)
            score = self.get_score(model, scoring)
            
            passed_mss.append(mss)
            score_list.append(score)

            if (i == 0) and (score < target_score):
                raise Exception('너무 높은 target f1 설정, 달성 불가능')

            if (score < target_score) and (i > 0): #목표 f1값 아래로 떨어지면
                # 이전 max depth가 임계치를 넘는 최소 depth
                # self.max_depth = passed_mss[-2]
                self.max_score = score_list[-2]
                
                plt.plot(passed_mss,score_list)
                plt.xlabel('min sample split')
                plt.ylabel('fl-score')
                plt.hlines(y=target_score, xmin = passed_mss[-1], xmax=passed_mss[0],colors='r')
                plt.show()

                self.max_depth_dt = self.make_dt(max_depth = self.max_depth, min_samples_split=passed_mss[-2])
                return passed_mss[-2], score_list[-2]
                
            i += 1
        raise Exception('너무 작은 target f1')


    def get_proper_cost_complexity_pruning(self, target_score, scoring='f1_score'):
        self.max_depth = self.make_dt().get_depth()
        
        passed_ccp = []
        score_list = []
        i = 0

        for ccp in range(0, 10): #이분탐색 개선가능
            ccp = ccp/10000
            model = self.make_dt(max_depth = self.max_depth, ccp_alpha=ccp)
            score = self.get_score(model, scoring)
            
            passed_ccp.append(ccp)
            score_list.append(score)

            if (i == 0) and (score < target_score):
                raise Exception('너무 높은 target f1 설정, 달성 불가능')

            if (score < target_score) and (i > 0): #목표 f1값 아래로 떨어지면
                # 이전 max depth가 임계치를 넘는 최소 depth
                self.max_score = score_list[-2]
                
                plt.plot(passed_ccp,score_list)
                plt.xlabel('min sample split')
                plt.ylabel('fl-score')
                plt.hlines(y=target_score, xmin = passed_ccp[-1], xmax=passed_ccp[0],colors='r')
                plt.show()

                self.max_depth_dt = self.make_dt(max_depth = self.max_depth, ccp_alpha=passed_ccp[-2])
                return passed_ccp[-2], score_list[-2]
                
            i += 1
        raise Exception('너무 작은 target f1')
    


    def get_proper_min_samples_leaf(self, target_score, scoring='f1_score'):
        self.max_depth = self.make_dt().get_depth()
        
        passed_msl = []
        score_list = []
        i = 0

        for msl in range(1, 30): #이분탐색 개선가능
            model = self.make_dt(max_depth = self.max_depth, min_samples_leaf=msl)
            score = self.get_score(model, scoring)
            
            passed_msl.append(msl)
            score_list.append(score)

            if (i == 0) and (score < target_score):
                raise Exception('너무 높은 target f1 설정, 달성 불가능')

            if (score < target_score) and (i > 0): #목표 f1값 아래로 떨어지면
                # 이전 max depth가 임계치를 넘는 최소 depth
                self.max_score = score_list[-2]
                
                plt.plot(passed_msl,score_list)
                plt.xlabel('min sample split')
                plt.ylabel('fl-score')
                plt.hlines(y=target_score, xmin = passed_msl[-1], xmax=passed_msl[0],colors='r')
                plt.show()

                self.max_depth_dt = self.make_dt(max_depth = self.max_depth, min_samples_leaf=passed_msl[-2])
                return passed_msl[-2], score_list[-2]
                
            i += 1
        raise Exception('너무 작은 target f1')