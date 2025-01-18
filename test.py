# load_detect_correct.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from utils import *
from demo import *

# Load data
data_test = pd.read_csv("test_angle.csv")

# Prepare test data
X_test, Y_test = data_test.iloc[:, :-1].values, pd.get_dummies(data_test['target']).values

# Load the saved model
model = load_model("trained_angle_model.h5")
print("Model loaded successfully.")

# Test phase: predict on the test set
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)
true_classes = Y_test.argmax(axis=1)


# Call the feedback function
correct_feedback(model,'teacher_yoga/angle_teacher_yoga.csv')

cv2.destroyAllWindows()
