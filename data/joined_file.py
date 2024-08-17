import pandas as pd


file1 = "C:\\Users\\tomer\PycharmProjects\\NLP_Project\\HumanChoicePrediction\\data\\df_concat.csv"
file2 = "C:\\Users\\tomer\\PycharmProjects\\NLP_Project\\HumanChoicePrediction\\data\\games_clean_X_original.csv"
file3 = "C:\\Users\\tomer\\PycharmProjects\\NLP_Project\\HumanChoicePrediction\\data\\games_clean_Y_original.csv"

df = pd.read_csv(file1)
train = pd.read_csv(file2)
test = pd.read_csv(file3)

columns_to_select = ['reviewId', 'Staff (Pos)', 'Facilities (Pos)', 'Cleanliness (Pos)',
                     'Location (Pos)', 'Food (Pos)', 'Staff (Neg)', 'Facilities (Neg)',
                     'Cleanliness (Neg)', 'Location (Neg)', 'Food (Neg)']
df = df[columns_to_select]

# train_merged = pd.merge(train, df, on='reviewId')
train_merged = pd.merge(train, df, on='reviewId',how='left', sort=False)

original_order = train.index

train_merged = train_merged.reindex(original_order)



csv_file_path = "C:\\Users\\tomer\\PycharmProjects\\NLP_Project\\HumanChoicePrediction\\data\\games_clean_X.csv"
train_merged.to_csv(csv_file_path,index=False)

# test_merged = pd.merge(test, df, on='reviewId')
test_merged = pd.merge(test, df, on='reviewId',how='left', sort=False)

csv_file_path = "C:\\Users\\tomer\\PycharmProjects\\NLP_Project\\HumanChoicePrediction\\data\\games_clean_Y.csv"
test_merged.to_csv(csv_file_path,index=False)



file1 = "C:\\Users\\tomer\PycharmProjects\\NLP_Project\\HumanChoicePrediction\\data\\df_concat_NLP.csv"
file2 = "C:\\Users\\tomer\\PycharmProjects\\NLP_Project\\HumanChoicePrediction\\data\\games_clean_X.csv"
file3 = "C:\\Users\\tomer\\PycharmProjects\\NLP_Project\\HumanChoicePrediction\\data\\games_clean_Y.csv"

df = pd.read_csv(file1)
train = pd.read_csv(file2)
test = pd.read_csv(file3)

columns_to_select = ['reviewId', 'Staff (Pos)', 'Facilities (Pos)', 'Cleanliness (Pos)',
                     'Location (Pos)', 'Food (Pos)', 'Staff (Neg)', 'Facilities (Neg)',
                     'Cleanliness (Neg)', 'Location (Neg)', 'Food (Neg)']
df = df[columns_to_select]

train_merged = pd.merge(train, df, on='reviewId',how='left', sort=False)
csv_file_path = "C:\\Users\\tomer\\PycharmProjects\\NLP_Project\\HumanChoicePrediction\\data\\games_clean_X_trans_tagging.csv"
train_merged.to_csv(csv_file_path,index=False)

test_merged = pd.merge(test, df, on='reviewId',how='left', sort=False)
csv_file_path = "C:\\Users\\tomer\\PycharmProjects\\NLP_Project\\HumanChoicePrediction\\data\\games_clean_Y_trans_tagging.csv"
test_merged.to_csv(csv_file_path,index=False)

