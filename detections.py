import pickle
import pandas as pd
import numpy as np

# Read the dataset which will be used for making predictions
testing_data = "TMD_Detection_Testing/TMD_testing_dataset.csv"
testing_dataset = pd.read_csv(testing_data, header=None)
col_names = ['Distance','Speed', 'Acceleration', 'Avg Speed', 'STD speed', 'Max Speed', '75% percentile speed' , '50% percentile speed','Avg Acc', 'STD Acc']
testing_dataset.columns = col_names

# Read the trained model pickle file
model_name = 'finalized_model.pkl'
loaded_model = pickle.load(open(model_name, 'rb'))

# Find and Calculate the confidence percentage for each row of data in the testing dataset
predictions_array = loaded_model.predict_proba(testing_dataset)
predictions_percentages = np.amax(predictions_array,axis=1)
predictions_percentages = predictions_percentages * 100

# Class prediction for each row of data in the testing dataset
predictions_classes = loaded_model.predict(testing_dataset)

# Save the predictions (class and confidence percentage) in a txt file
final_predictions = []
i = 0
while (i < len(predictions_classes) - 1):
    final_prediction = "Prediction is '" + predictions_classes[i] + "' " + str(predictions_percentages[i]) + "%"
    final_predictions.append(final_prediction)
    i += 1

np.savetxt("TMD_Detection_Testing/TMD_testing_predictions.txt" , final_predictions, delimiter=",", fmt='%s')