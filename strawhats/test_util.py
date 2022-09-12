# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
# from config import args

# %%
y_true=[]
y_pred=[]
from tqdm import tqdm
for pkl in tqdm(range(0,40)):
    if pkl==33:
        continue
    y_test_pred_batch=np.load("Array/y_test_pred"+str(pkl)+".npy")
    y_test_true_batch=np.load("Array/y_test_true"+str(pkl)+".npy")
    # print("pkl:",pkl,"len: ",len(y_test_pred_batch))
    y_true=y_true+list(y_test_true_batch)
    y_pred=y_pred+list(y_test_pred_batch)

# %%
len(y_pred)

# %%
accuracy = accuracy_score(y_true=np.array(y_true), y_pred=np.array(y_pred))
print(accuracy)

# %%
report = classification_report(y_true=np.array(y_true), y_pred=y_pred, target_names=['Real', 'Fake'])
conf_matrix = confusion_matrix(y_true=np.array(y_true), y_pred=y_pred)

# %%
report

# %%
conf_matrix

# %%



