from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("insurance_claim_data.txt")

# police_report_new = data.loc[(data.police_report=="Yes") | (data.police_report=="No"),]

data['police_report'] = data['police_report'].replace({'Yes': 1, 'No': 0, 'Unknown':2})
data["claim_type_numeric"] = data["claim_type"].replace({"Injury only":0,"Material only":1,"Material and injury":2})
data["claim_area_num"] = data["claim_area"].replace({"Auto" : 1 , "Home" : 0})
data.columns
features_var = list(['age','days_to_incident', 'claim_amount','claim_type_numeric', 'total_policy_claims'])

x = data[features_var]
y = data.claim_area_num

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.30)

clf_tree = DecisionTreeClassifier()
clf_reg = LogisticRegression()

clf_tree.fit(x_train, y_train)
clf_reg.fit(x_train, y_train)

s_tree = pd.DataFrame(clf_tree.predict_proba(x_test), columns=clf_tree.classes_)
s_reg = pd.DataFrame(clf_reg.predict_proba(x_test), columns=clf_reg.classes_)


y_score1 = clf_tree.predict_proba(x_test)[:,1]
y_score2 = clf_reg.predict_proba(x_test)[:,1]


false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)
false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, y_score2)


print('roc_auc_score for DecisionTree: ', roc_auc_score(y_test, y_score1))
print('roc_auc_score for Logistic Regression: ', roc_auc_score(y_test, y_score2))




plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - DecisionTree')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Logistic regression')
plt.plot(false_positive_rate2, true_positive_rate2)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

clf_tree.feature_importances_
