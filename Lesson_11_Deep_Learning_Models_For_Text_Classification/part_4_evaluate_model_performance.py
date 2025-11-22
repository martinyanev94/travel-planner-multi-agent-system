from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

y_pred = (model.predict(x_test) > 0.5).astype("int32")

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
