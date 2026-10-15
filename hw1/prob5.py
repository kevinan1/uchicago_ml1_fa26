from sklearn.datasets import load_breast_cancer


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm, t
import pandas as pd
import optax

def prob5class(model):
    X, y = load_breast_cancer(as_frame=True, return_X_y=True)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = StandardScaler().fit(X_train)
    X_train, X_test = sc.transform(X_train), sc.transform(X_test)
    clf = LogisticRegression().fit(X_train,Y_train)
    print("baseline acc:", clf.score(X_test, Y_test))
    X_train, X_test = jnp.asarray(X_train, jnp.float32), jnp.asarray(X_test, jnp.float32)
    Y_train, Y_test = jnp.asarray(Y_train, jnp.float32), jnp.asarray(Y_test, jnp.float32)
    def init_params(n):
        return {"w": jnp.zeros(n), "b": jnp.array(0.0)}

    def cross_entropy(params, X, y, eps=1e-7):
        p = jnp.clip(model(params, X), eps, 1 - eps)
        return -jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))

    params = init_params(X_train.shape[1])
    opt = optax.sgd(learning_rate=0.01)
    state = opt.init(params)
    @jax.jit
    def step(params, state):
        l, grads = jax.value_and_grad(cross_entropy)(params, X_train, Y_train)
        updates, state = opt.update(grads, state)
        return optax.apply_updates(params, updates), state, l

    for i in range(10000):
        params, state, l = step(params, state)

    print("jax acc:", float(jnp.mean((model(params, X_test) > 0.5) == Y_test)))

def logistic(params, X):
    return jax.nn.sigmoid(X @ params['w'] + params['b'])

def probit(params, X):
    return norm.cdf(X @ params['w'] + params['b'])

def linear(params, X):
    return X @ params['w'] + params['b']

def prob5a_logistic_class():
    prob5class(logistic)
def prob5b_probit_class():
    prob5class(probit)

def prob5_regression(loss):
    df = pd.read_csv('ale.csv')
    X = df[['anchor_ratio', 'trans_range', 'node_density', 'iterations']]
    y = df['ale']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    sc = StandardScaler().fit(X_train)
    X_train, X_test = sc.transform(X_train), sc.transform(X_test)

    clf = LinearRegression().fit(X_train, y_train)
    print('linear regression error:', mean_absolute_error(y_test, clf.predict(X_test)))

    X_train, X_test = jnp.asarray(X_train, jnp.float32), jnp.asarray(X_test, jnp.float32)
    y_train, y_test = jnp.asarray(y_train, jnp.float32), jnp.asarray(y_test, jnp.float32)

    params = {'w': jnp.zeros(4), 'b': jnp.array(0.0)}

    opt = optax.sgd(learning_rate = 0.1)
    state = opt.init(params)

    @jax.jit
    def step(params, state):
        l, grads = jax.value_and_grad(loss)(params, X_train, y_train)
        updates, state = opt.update(grads, state)
        return optax.apply_updates(params, updates), state, l

    for i in range(5000):
        params, state, l = step(params, state)
        
    print('jax MAE:', jnp.mean(jnp.abs(y_test - linear(params, X_test))))
    print('avg test loss:', loss(params, X_test, y_test))

def mse(params, X, y):
    return jnp.mean((y - linear(params, X)) ** 2)

def t_loss(params, X, y, nu=5.0):
    return -jnp.mean(t.logpdf(y - linear(params, X), nu))

def prob5a_linear_reg():
    prob5_regression(mse)
def prob5b_t_reg():
    prob5_regression(t_loss)
    
