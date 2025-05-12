from xgboost import XGBClassifier
import numpy as np

# Step 1: Expanded training data (Age, Income)
X_train = np.array([
    [22, 25090], [25, 3000], [28, 3200], [30, 4000], [35, 5000],
    [40, 6000], [45, 7000], [50, 7500], [55, 8000], [60, 8500],
    [23, 2700], [26, 3500], [29, 3600], [32, 4100], [37, 5200],
    [42, 6100], [47, 7200], [52, 7700], [57, 8300], [62, 8800],
    [24, 260000], [27, 3400], [33, 4300], [36, 5100], [39, 5900],
    [44, 6800], [49, 7400], [53, 7900], [58, 8400], [65, 8900]
])

# Step 2: Labels (0 = No, 1 = Yes)
y_train = np.array([
    1, 0, 0, 1, 1,
    1, 1, 1, 1, 1,
    0, 0, 0, 1, 1,
    1, 1, 1, 1, 1,
    1, 0, 1, 1, 1,
    1, 1, 1, 1, 1
])

# Step 3: Create and train the model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Step 4: User input for prediction
age = int(input("Enter the age: "))
income = float(input("Enter the income: "))
if age <= 0 or income <= 0:
    print("Invalid input: Age and income must be positive.")
else:
    new_input = np.array([[age, income]])
    prediction = model.predict(new_input)

    # Step 5: Interpret result
    if prediction[0] == 1:
        print("Prediction: Likely to buy the product ✅")
    else:
        print("Prediction: Not likely to buy the product ❌")
