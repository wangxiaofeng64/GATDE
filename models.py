import sys

sys.path.append("your filepath/gat.py")
import gat
from gat import GATConv

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse as sp
import pandas as pd
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings("ignore")

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Reshape, GlobalMaxPool1D, MaxPool1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import activations, regularizers, constraints, initializers
import tensorflow.keras.backend as K

tf.keras.utils.set_random_seed(123)
tf.random.set_seed(123)

ppi = pd.read_csv('data/BRCA/ppi_network_of_BRCA.csv')
expression = pd.read_csv('data/BRCA/RPPA_data_of_BRCA.csv')
cancertype = pd.read_csv('data/BRCA/clinical_data_of_BRCA.csv')
cancertype = cancertype[['patient', 'BRCA_Subtype_PAM50']]
expression = expression.transpose()

expression.reset_index(inplace=True)
expression.columns = expression.iloc[0]
expression = expression[1:]
expression["Sample_description"] = expression["Sample_description"].apply(lambda x: x[:12])
expression

def Diffusion(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]

    A_loop = sp.eye(N) + A

    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

    # PPR-based diffusion
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)

    # Sparsify using threshold epsilon
    S_tilde = S.multiply(S >= eps)

    # Column-normalized transition matrix on graph S_tilde
    D_tilde_vec = S_tilde.sum(0).A1

    T_S = S_tilde / D_tilde_vec

    return T_S


data = cancertype.merge(expression, left_on='patient', right_on='Sample_description')
data.dropna()
columns = data.columns.tolist()[3:]
columns
','.join(columns)

data1 = data[columns]
corr = data1.corr()
adjoint = corr.applymap(lambda x: 0)

for i in range(0, len(ppi)):
    node1 = ppi.iloc[i, 0]
    node2 = ppi.iloc[i, 1]

    correlation = np.corrcoef(expression[node1].tolist(), expression[node2].tolist())[0, 1]
    correlation = abs(correlation)
    adjoint.loc[node1, node2] = correlation
    adjoint.loc[node2, node1] = correlation

adjoint.fillna(0, inplace=True)  # Ensure adjoint matrix has no NaN values

sparse_matrix1 = csr_matrix(adjoint.values)
x = Diffusion(sparse_matrix1, 0.1, 0.01)
dense_matrix = x.toarray()

df = pd.DataFrame(dense_matrix).fillna(0)
x2 = df.applymap(lambda x: 1 if x != 0 else 0)
x2

non_zero_count = (x2.fillna(0) != 0).sum().sum()

labels = data["BRCA_Subtype_PAM50"]
label_mapping = {'LumA': 0, 'Basal': 1, 'Her2': 2, 'LumB': 3, 'Normal': 4}
data['numeric_label'] = data['BRCA_Subtype_PAM50'].replace(label_mapping)
x = data[columns].astype(np.float32)
from sklearn.preprocessing import StandardScaler
def z_score_standardization(data, columns):
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns].astype(np.float32))
    return data[columns]

scaler = StandardScaler()
x= scaler.fit_transform(x)
x.shape

y = data['numeric_label']
from tensorflow.keras.utils import to_categorical

y_encoded = to_categorical(y)
y_encoded.shape

X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)
tf.keras.utils.set_random_seed(123)
tf.random.set_seed(123)


def build_model():
    X_in = Input(shape=(len(adjoint), 1))
    A_in = Input((len(adjoint), len(adjoint)), sparse=True)

    # GATConv expects two inputs: [features, adjacency matrix]
    X_1 = GATConv(1, activation='relu')([X_in, A_in])

    # Use Lambda to extract the features
    X_1_features = tf.keras.layers.Lambda(lambda x: x[0])(X_1)
    X_2 = Flatten()(X_1_features)

    output = Dense(5, activation='softmax')(X_2)
    model = Model(inputs=[X_in, A_in], outputs=output)
    return model


model = build_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

xtrain = np.expand_dims(X_train, axis=2)
print("x train", xtrain.shape)

xtest = np.expand_dims(X_test, axis=2)
print("x test", xtest.shape)

adjoint1 = np.array(x2)[np.newaxis, :, :]
adjoint2 = np.repeat(adjoint1, len(xtrain), 0)
adjoint3 = np.repeat(adjoint1, len(xtest), 0)
print("A", adjoint2.shape)

model.fit([xtrain, adjoint2], y_train, batch_size=32, epochs=100, validation_split=0.2)

y_pred = model.predict([xtest, adjoint3])

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

recall = recall_score(y_test, y_pred, average='macro')
print("Recall:", recall)

precision = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)

specificity = 1 - (sum((y_pred == 1) & (y_test == 0)) / sum(y_test == 0))
print("Specificity:", specificity)

f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)

mcc = matthews_corrcoef(y_test, y_pred)
print("MCC:", mcc)