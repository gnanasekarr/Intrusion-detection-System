import numpy as np
import pandas as pd
import pickle

# STEP 1: Load Input dataset

loadset = pd.read_csv(r'E:\Gnanasekar\Project\intrusion detection system\Deployment-flask-master\KDD Cup 1999 imbalanced Data with 10000 records - Copy.csv')
print("==============================\nKDD Cup 1999 imbalanced Data with 10000 records\n==============================\n",loadset)
print(loadset)

#STEP 2: Feature Selection
inp=["protocol_type","service","src_bytes","dst_bytes","logged_in","count","srv_count","srv_diff_host_rate","diff_srv_rate","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","class"]

load=pd.DataFrame(loadset[inp])

# STEP 3: Convert categorical to numerical data using Label Encoding technique

from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

cols_label_encoder = ['service', 'class','protocol_type']

for col in cols_label_encoder:
    load[col] = encoder.fit_transform(load[col])

print("\n\nback-->0\nbuffer_overflow-->1\nftp_write-->2\nguess_passwd-->3\nimap-->4\nipsweep-->5\nland-->6\nloadmodule-->7\nmultihop-->8\nneptune-->9\nnmap-->10\nnormal-->11\nperl-->12\nphf-->13\npod-->14\nportsweep-->15\nrootkit-->16\nsatan-->17\nsmurf-->18\nspy-->19\nteardrop-->20\nwarezclient-->21\nwarezmaster-->22\n")
print("\n=======================================================================\nDataset after categorical to numerical conversion\n=======================================================================\n",load)

# df = pd.get_dummies(load, columns=cols_onehot)

X = load.drop(labels='class', axis=1)
y = load['class']


# STEP 4: Split dataset into training and testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=4)

print("\n==========================================================================================\nTraining Dataset without class label\n==========================================================================================\n",X_train)
print("\n=============================================================================================\nTraining Dataset class label values only\n=============================================================================================\n",y_train)
print("\n==========================================================================================\nTesting Dataset without class label\n==========================================================================================\n",X_test)
print("\n=============================================================================================\nActual class label values for Testing Dataset\n=============================================================================================\n",y_test)


# STEP 5: Principal Component Analysis (PCA) for feature selection and dimensionlaity reduction

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
PCARF = RandomForestClassifier()
PCARF.fit(X_train, y_train)

y_pred = PCARF.predict(X_test)
print(y_pred)

# val1=np.array([load.iloc[534,0:-1].values])
# print(PCARF.predict(val1))

pickle.dump(PCARF, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

from sklearn.metrics import accuracy_score
PCARFAccuracy = accuracy_score(y_pred,y_test)*100
print("PCA-RF Accuracy: ",PCARFAccuracy," %\n")


from sklearn.svm import SVC
svm=SVC()
svm.fit(X_train, y_train)
Y_pred = svm.predict(X_test)
print("\n======================================================================================================\nSVM Predicted class label values for Testing Dataset\n======================================================================================================\n",Y_pred)

from sklearn.metrics import accuracy_score
svmAccuracy = accuracy_score(y_pred,y_test)*100
print("SVM Accuracy: ",svmAccuracy," %\n")


