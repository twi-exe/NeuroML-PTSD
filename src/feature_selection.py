from sklearn.linear_model import ElasticNet
import numpy as np

def select_features(X_train, y_train, top_k=20):
    model = ElasticNet(random_state=42)
    model.fit(X_train, y_train)
    selected = X_train.columns[np.abs(model.coef_) > 0][:top_k]
    return selected
