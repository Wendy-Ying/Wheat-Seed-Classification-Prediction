import numpy as np

# 定义一个简单的决策树桩分类器
class DecisionStump:
    def fit(self, X, y, sample_weights):
        self.best_feature = None
        self.best_threshold = None
        self.best_polarity = None
        self.best_error = float('inf')
        m, n = X.shape
        
        for feature in range(n):
            # 获取当前特征的值，并排序
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # 分别尝试正负两种方向
                for polarity in [1, -1]:
                    # 预测结果：如果特征值小于阈值，则为1；否则为-1
                    predictions = np.ones(m)
                    predictions[polarity * X[:, feature] < polarity * threshold] = -1
                    
                    # 计算误差
                    errors = (predictions != y)
                    weighted_error = np.sum(sample_weights * errors)
                    
                    if weighted_error < self.best_error:
                        self.best_error = weighted_error
                        self.best_feature = feature
                        self.best_threshold = threshold
                        self.best_polarity = polarity

    def predict(self, X):
        m = X.shape[0]
        predictions = np.ones(m)
        feature_values = X[:, self.best_feature]
        threshold = self.best_threshold
        polarity = self.best_polarity
        
        predictions[polarity * feature_values < polarity * threshold] = -1
        return predictions

# 定义AdaBoost算法
class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        m, n = X.shape
        # 初始化样本权重为均等
        sample_weights = np.ones(m) / m
        # 将标签y转为 {1, -1}，因为AdaBoost假设标签是二分类的
        y = 2 * y - 1  # 假设标签为 {0, 1} -> 转为 { -1, 1 }

        for _ in range(self.n_estimators):
            # 训练一个弱分类器
            stump = DecisionStump()
            stump.fit(X, y, sample_weights)
            
            # 获取该弱分类器的预测结果
            predictions = stump.predict(X)
            
            # 计算弱分类器的错误率
            errors = (predictions != y)
            weighted_error = np.sum(sample_weights * errors) / np.sum(sample_weights)
            
            # 计算该分类器的权重alpha
            alpha = 0.5 * np.log((1 - weighted_error) / (weighted_error + 1e-10))
            
            # 更新样本权重
            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)  # 归一化权重
            
            # 保存弱分类器和其alpha值
            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        # 使用每个弱分类器做加权投票
        m = X.shape[0]
        final_predictions = np.zeros(m)
        
        for model, alpha in zip(self.models, self.alphas):
            final_predictions += alpha * model.predict(X)
        
        # 返回最终分类（+1 或 -1）
        return np.sign(final_predictions)

# 测试AdaBoost
if __name__ == '__main__':
    # 示例数据
    X_train = np.array([[1, 2], [2, 3], [3, 3], [4, 5], [5, 6], [6, 7]])
    y_train = np.array([0, 0, 1, 1, 1, 0])
    
    # 创建并训练AdaBoost模型
    model = AdaBoost(n_estimators=10)
    model.fit(X_train, y_train)
    
    # 进行预测
    predictions = model.predict(X_train)
    print("Predictions:", predictions)
