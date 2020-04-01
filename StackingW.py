import numpy as np
from sklearn.model_selection import KFold
import copy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


class StackingW:
    def __init__(self, topLayer_model, base_model_list,
                 n_fold=5, use_probas=True, average_probas=False, val_weight_average=False, val_set=[]):
        self.topLayer_model = topLayer_model
        self.base_model_list = base_model_list #存储M个输入的未训练模型
        self.n_flod = n_fold  # 默认5折交叉
        self.use_probas=use_probas
        self.average_probas = average_probas
        self.val_weight_average = val_weight_average
        self.val_set = val_set
        self.weight_lst = []

    def fit(self, X_train, y_train):
        X_train, y_train = np.array(X_train), np.array(y_train)
        self.class_inter_dict = self.__build_class_inter_dict(y_train)
        print(self.class_inter_dict)

        self.had_train_models = []  # 存储训练好的(M * K)个模型
        for i, model in enumerate(self.base_model_list):
            #print('model_{}'.format(i))
            train_pred = []
            KFold_models = []
            loss_lst = []
            for j, (tra_idx, val_idx) in enumerate(KFold(n_splits=self.n_flod).split(X_train)):
                X_tra, X_val = X_train[tra_idx], X_train[val_idx]
                y_tra, y_val = y_train[tra_idx], y_train[val_idx]
                model.fit(X_tra, y_tra)
                if self.val_set==[]:
                    print('#直接用"构建特征的验证集"计算损失')
                    loss_lst.append(self.__cal_loss(model, X_val, y_val))  #直接用构建特征的验证集计算损失
                else:
                    print('#使用"外部验证集"计算损失')
                    loss_lst.append(self.__cal_loss(model, self.val_set[0], self.val_set[1]))  #使用外部验证集计算损失
                KFold_models.append(copy.deepcopy(model))
                if self.use_probas:
                    train_pred += model.predict_proba(X_val).tolist()
                else:
                    train_pred += [[e]for e in model.predict(X_val)]
            self.weight_lst.append(self.__cal_weight_lst(loss_lst))
            self.had_train_models.append(copy.deepcopy(KFold_models))  #存储训练好的K折模型，用于预测

            train_pred = np.array(train_pred)
            if i == 0:
                X_train_stack = train_pred
            else:
                if not self.average_probas:
                    X_train_stack = np.c_[X_train_stack, train_pred]
                else:
                    #将每个模型的预测，求平均
                    X_train_stack += train_pred
                    if i == len(self.base_model_list) - 1:
                        X_train_stack = X_train_stack / len(self.base_model_list)

        # 顶层模型的训练
        self.topLayer_model.fit(X_train_stack, y_train)

    def predict(self, X_test):
        return self.__predict_tmp(X_test, out_probas=False)

    def predict_proba(self, X_test):
        return self.__predict_tmp(X_test, out_probas=True)

    def __predict_tmp(self, X_test, out_probas=False):  # 测试集的数据是X_test_stack，而不是原来的X_test
        for i, KF_models in enumerate(self.had_train_models):
            test_pred = []
            for model in KF_models:
                if self.use_probas:
                    test_pred.append(model.predict_proba(X_test).tolist())
                else:
                    test_pred.append([[e] for e in model.predict(X_test)])
            if self.val_weight_average:  #每折加权平均
                test_pred = self.__cal_weight_average(self.weight_lst[i], np.array(test_pred))
            else: #每折直接平均
                test_pred = np.mean(np.array(test_pred), axis=0)
            if i == 0:
                X_test_stack = test_pred
            else:
                if not self.average_probas:
                    X_test_stack = np.c_[X_test_stack, test_pred]
                else:
                    X_test_stack += test_pred
                    if i == len(self.base_model_list) - 1:
                        X_test_stack = X_test_stack / len(self.base_model_list)
        # 顶层模型预测
        if out_probas:
            return self.topLayer_model.predict_proba(X_test_stack)
        else:
            return self.topLayer_model.predict(X_test_stack)

    def __cal_weight_average(self, kw_lst, test_pred):
        test_weight_average = []
        for kw, test_single in zip(kw_lst, test_pred):
            test_weight_average.append(kw * test_single)
        return np.sum(test_weight_average, axis=0)

    def __cal_weight_lst(self, loss_lst):
        print('每一折的损失:', loss_lst)
        Sk_sum = 0
        for sj in loss_lst:
            Sk_sum += (1 / sj)
        weight_lst = []
        for sk in loss_lst:
            weight_lst.append((1 / sk) / Sk_sum)

        print('每一折对应的模型权重:', weight_lst)
        print('所有权值加起来=', sum(weight_lst))
        return weight_lst

    def __cal_loss(self, model, X_val, y_val):
        n_class = len(set(y_val))
        y_pred_proba = model.predict_proba(X_val)
        y_val_oneHot = self.__oneHot(y_val)

        #计算损失
        sk = 0
        for i_sample in range(len(y_val)):
            for i_class in range(n_class):
                sk += abs(y_pred_proba[i_sample,i_class] - y_val_oneHot[i_sample,i_class])
        return sk / n_class

    def __oneHot(self, y_val):
        inter_encode = np.array([self.class_inter_dict[e] for e in y_val])
        onehot_encoder = OneHotEncoder(sparse=False)
        y_val_ontHot = onehot_encoder.fit_transform(inter_encode.reshape(-1, 1))
        return np.array(y_val_ontHot)

    def __build_class_inter_dict(self, y_train):
        y_train_set = set(y_train)
        class_inter_dict = {}
        for i, e in enumerate(y_train_set):
            class_inter_dict[e] = i
        return class_inter_dict

