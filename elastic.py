import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics as mt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

data = pd.read_csv("C:/Users/ilhan/Downloads/Ev.csv")
veri = data.copy()

veri = veri.drop(columns="Address", axis=1)
y = veri["Price"]
X = veri.drop(columns="Price", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def caprazdog(model):
    dogruluk = cross_val_score(model, X, y, cv=10)
    return dogruluk.mean()

def basari(gercek, tahmin):
    rmse = mt.mean_squared_error(gercek, tahmin, squared=True)
    r2 = mt.r2_score(gercek, tahmin)
    return [rmse, r2]

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_tahmin = lin_model.predict(X_test)

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)
ridge_tahmin = ridge_model.predict(X_test)

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_tahmin = lasso_model.predict(X_test)

elas_model = ElasticNet(alpha=0.1)
elas_model.fit(X_train, y_train)
elas_tahmin = elas_model.predict(X_test)

sonuclar = [["Linear Model", basari(y_test, lin_tahmin)[0], basari(y_test, lin_tahmin)[1], caprazdog(lin_model)],
            ["Ridge Model", basari(y_test, ridge_tahmin)[0], basari(y_test, ridge_tahmin)[1], caprazdog(ridge_model)],
            ["Lasso Model", basari(y_test, lasso_tahmin)[0], basari(y_test, lasso_tahmin)[1], caprazdog(lasso_model)],
            ["Elastic Net Model", basari(y_test, elas_tahmin)[0], basari(y_test, elas_tahmin)[1], caprazdog(elas_model)]]

pd.options.display.float_format = '{:.4f}'.format
sonuclar = pd.DataFrame(sonuclar, columns=["Model", "RMSE", "R2", "Doğrulama"])
print(sonuclar)
