import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import itertools
from sklearn.model_selection import KFold

pd.set_option('display.max_columns', None)
ds = pd.read_csv("mobiles.csv")
ds = ds.drop_duplicates(subset='Title', keep='first').reset_index(drop=True)
# Load the dataset
original_ds = pd.read_csv("mobiles.csv")

# Remove duplicate model numbers, keeping the first occurrence
original_ds = original_ds.drop_duplicates(subset='Title', keep='first').reset_index(drop=True)
ds['Height'] = ds['Height'].replace(r'mm', '', regex=True).astype(float)
ds['Width'] = ds['Width'].replace(r'mm', '', regex=True).astype(float)
ds['Depth'] = ds['Depth'].replace(r'mm', '', regex=True).astype(float)
ds['Weight'] = ds['Weight'].replace(r'g', '', regex=True).astype(float)
ds['Battery Capacity'] = ds['Battery Capacity'].replace(r'mAh', '', regex=True).astype(float)

# Calculate percentage of missing values
missing_percentage = ds.isnull().sum() / len(ds) * 100

# Set a threshold for dropping columns
threshold = 50  # Percentage

# Drop columns with missing values above the threshold
columns_to_drop = missing_percentage[missing_percentage > threshold].index
ds = ds.drop(columns=columns_to_drop)

# Display the columns dropped
list(columns_to_drop)
numeric_cols = ds.select_dtypes(include=['number']).columns
object_cols = ds.select_dtypes(include=['object']).columns

# Fill missing values in numeric columns with the mean
ds[numeric_cols] = ds[numeric_cols].fillna(ds[numeric_cols].mean())

# Fill missing values in non-numeric columns with the mode
for col in object_cols:
    ds[col] = ds[col].fillna(ds[col].mode()[0])

# Verify if there are any remaining missing values
print("Remaining missing values in the dataset:")
print(ds.isnull().sum().sum())
columns_to_drop = [
    'Secondary Camera Available', 'Primary Camera Available', 'Other Features','OTG Compatible','Unnamed: 0','Display_size_inches',
    'In The Box','Other Display Features','Operating Frequency'
]
ds = ds.drop(columns=columns_to_drop)

num_columns = ds.shape[1]
print(f"Number of remaining columns: {num_columns}")

# Drop rows where 'Internal Storage' contains unexpected formats
ds = ds[~ds['Internal Storage'].str.contains(r'\+|TB|other_unwanted_patterns', na=False)]

def convert_storage(value):
    # Handle values with 'GB'
    if 'GB' in value:
        return float(value.replace('GB', '').strip())
    # Handle values with 'MB'
    elif 'MB' in value:
        return float(value.replace('MB', '').strip()) / 1024
    # Return NaN for unexpected formats
    else:
        return None

# Apply the conversion function
ds['Internal Storage'] = ds['Internal Storage'].apply(convert_storage)

# Handle missing or inconsistent data
ds['Internal Storage'].fillna(ds['Internal Storage'].mean(), inplace=True)
original_ds = ds.copy()
yes_no_columns = [
    'Hybrid Sim Slot', 'Touchscreen', 'HD Recording', 'Full HD Recording',
    'Video Recording', 'Wi-Fi Hotspot', 'NFC', 'GPS Support', 'Smartphone',
    'Removable Battery', 'SMS','Bluetooth Support','Wi-Fi'
]

# Replace 'Yes' with 1 and 'No' with 0
for col in yes_no_columns:
    if col in ds.columns:
        ds[col] = ds[col].map({'Yes': 1, 'No': 0})

# Select the columns that need label encoding
columns_to_encode = ['Title','Brand', 'Color', 'Resolution', 'Resolution Type', 'Display Type',
                     'Operating System','Map Support' ,'Processor Type','Browse Type','SIM Type', 'Processor Core','Bluetooth Version','SIM Size']

# Create a copy of the DataFrame to avoid modifying the original data


# Apply label encoding to each column
label_encoders = {}  # To store encoders if you need to reverse the transformation
for column in columns_to_encode:
    le = LabelEncoder()
    ds[column] = le.fit_transform(ds[column])
    label_encoders[column] = le  # Save the encoder for future use

ds['primary_camera']=ds['Primary Camera'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x)
ds['Primary Camera'] = ds['primary_camera'].apply(lambda x: x[0] if x else None)
ds['Primary Camera'] = ds['Primary Camera'].apply(lambda x: float(x) if x else None)
ds.drop(columns=['primary_camera'],inplace=True)
# Calculate the mean of 'Primary Camera' column
mean_value = ds['Primary Camera'].mean()

# Fill NaN values with the calculated mean
ds['Primary Camera'].fillna(mean_value, inplace=True)

null_count = ds['Primary Camera'].isnull().sum()
print(f"Number of null values in 'Primary Camera': {null_count}")

ds['5G'] = ds['Network Type'].apply(lambda x: 1 if '5G' in x else 0)
ds.drop(columns=['Network Type'],inplace=True)
ds['SC']=ds['Secondary Camera'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x)
ds['Secondary Camera'] = ds['Secondary Camera'].apply(lambda x: x[0] if x else None)
ds['Secondary Camera'] = ds['Secondary Camera'].apply(lambda x: float(x) if x else None)
ds['Secondary Camera'].fillna(ds['Secondary Camera'].mean(),inplace=True)
ds.drop(columns=['SC'],inplace=True)
sensor_frequency = ds['Sensors'].value_counts().to_dict()
ds['Sensor_Encoded'] = ds['Sensors'].map(sensor_frequency)
ds = ds.drop(['Sensors'], axis = 1)
# Check for 'GPS' keyword in 'GPS Type' and encode accordingly
ds['GPS_Encoded'] = ds['GPS Type'].str.contains('GPS', na=False).astype(int)

# Now 'ds' DataFrame contains a new column 'GPS_Encoded' with 1 for rows where
# 'GPS Type' contains 'GPS', and 0 otherwise.

ds.drop(columns=['GPS Type'],inplace=True)
ds.drop(columns=['Video Formats'],inplace=True)
num_cols = len(ds.select_dtypes(include=[np.number]).columns)
rows = (num_cols + 1) // 2  # Use 2 columns instead of 3 for better spacing
cols = 2

def remove_outliers_iqr(df, column, factor=1.5):
    # Check if the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (factor * IQR)
    upper_bound = Q3 + (factor * IQR)

    # Identify outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    # Remove outliers from the DataFrame
    df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Extract titles of the removed outliers
    removed_titles = outliers['Title'].values.tolist()

    return df_cleaned, removed_titles

# Process numerical columns
removed_outliers_titles = []
for col in ds.select_dtypes(include=[np.number]).columns:
    cleaned_df, outliers = remove_outliers_iqr(ds, col)
    removed_outliers_titles.extend(outliers)

ds = cleaned_df

print("Outliers removed.")
# Drop the specified columns
ds = ds.drop(['Wi-Fi Version','Primary Camera Features', 'Secondary Camera Features','Model Number','Model Name','Flash','Dual Camera Lens','Supported Networks', 'Internet Connectivity'], axis=1)
# Define the columns for each scaler
features_to_scale_1 = ['Brand','Price','Processor Type', 'Primary Camera', 'Internal Storage', 'Battery Capacity', '5G']
features_to_scale_2 = [col for col in ds.columns if col not in features_to_scale_1 ]  # Remaining columns

# Initialize scalers
scaler_1 = StandardScaler()
scaler_2 = StandardScaler()

# Fit the scalers on the respective feature sets
scaler_1.fit(ds[features_to_scale_1])  # Fit on first set of features
scaler_2.fit(ds[features_to_scale_2])  # Fit on remaining features

# Now apply the scalers to the corresponding columns in the original dataset
ds_scaled_1 = scaler_1.transform(ds[features_to_scale_1])  # Transform first set of features
ds_scaled_2 = scaler_2.transform(ds[features_to_scale_2])  # Transform remaining features

# Combine the scaled features back into the original dataframe structure
scaled_features = np.hstack([ds_scaled_1, ds_scaled_2])

# Create a DataFrame with the scaled features (you can combine them with non-scaled features if needed)
ds = pd.DataFrame(scaled_features, columns=features_to_scale_1 + features_to_scale_2)

# Each row in normalized_data is now an embedded vector
embeddings = np.array(ds)

# Example: Accessing the first embedded vector
print("First embedded vector:", embeddings[0])

# Optional: Save the embedded vectors to a file
np.save('embedded_vectors.npy', embeddings)
# Applying PCA
n_components = 3  # Adjust as needed
pca = PCA(n_components=n_components)
pca_data = pca.fit_transform(ds)

# Measure accuracy retained
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

print("Explained Variance Ratio (per component):", explained_variance)
print("Cumulative Explained Variance Ratio:", cumulative_explained_variance)

# For example, if retaining 95% variance is your goal:
retained_variance = cumulative_explained_variance[-1]
print(f"Accuracy retained by PCA: {retained_variance * 100:.2f}%")
# Calculate cumulative explained variance for all components
pca = PCA()
pca.fit(ds)
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

# Print and plot results
print("Number of components and retained variance:")
for i, variance in enumerate(cumulative_explained_variance, 1):
    print(f"n_components={i}, Cumulative Explained Variance={variance * 100:.2f}%")

# Optimal n_components to retain at least 95% variance
optimal_components = np.argmax(cumulative_explained_variance >= 0.95) + 1
print(f"\nOptimal n_components to retain at least 95% variance: {optimal_components}")
pca = PCA(n_components=27)
reduced_vectors = pca.fit_transform(embeddings)

print("Reduced embedded vector (first two vectors):", reduced_vectors[0],reduced_vectors[1])
# Optional: Save the embedded vectors to a file
np.save('embedded_vectors.npy', reduced_vectors)
model_number_input="REDMI 12 (Moonstone Silver, 128 GB)"
def get_model_index(model_number_input, original_ds, reduced_vectors):
    """
    Retrieves the index of a specified model number in the dataset.

    Parameters:
    - model_number_input (str): The model number to search for.
    - dataset (DataFrame): The dataset containing the model information.
    - reduced_vectors (list or ndarray): The vector representation for validation.

    Returns:
    - int: The index of the chosen model number.

    Raises:
    - ValueError: If the model number is not found or the index is invalid.
    """
    # Search for the model number in the dataset
    model_row = original_ds[original_ds['Title'] == model_number_input]

    # Check if the model number exists in the dataset
    if model_row.empty:
        raise ValueError(f"Model number {model_number_input} not found in the dataset.")
    else:
        chosen_index = model_row.index[0]  # Get the index of the chosen model number

    # Ensure valid index
    if chosen_index < 0 or chosen_index >= len(reduced_vectors):
        raise ValueError(f"Invalid index: {chosen_index}. Please enter a valid index.")

    return chosen_index
# Example call to the function
chosen_index = get_model_index(
    model_number_input="APPLE iPhone 13 (Pink, 128 GB)",
    original_ds=original_ds,
    reduced_vectors=reduced_vectors
)


print(f"Chosen Index: {chosen_index}")

nds = ds
nds = nds[['Brand', 'Price', 'Primary Camera', 'Processor Type', 'Internal Storage', '5G', 'Battery Capacity' ]]
# Each row in normalized_data is now an embedded vector
embedded_vectors = np.array(nds)

# Example: Accessing the first embedded vector
print("First embedded vector:", embedded_vectors[0])

# # Optional: Save the embedded vectors to a file
# np.save('vectors.npy', embedded_vectors)

# data = np.load("vectors.npy")
# targets = np.load("embedded_vectors.npy")

# # Hyperparameter tuning function
# def tune_hyperparameters_with_validation(data, targets, input_dim, output_dim, hyperparams, k_folds=5, epochs=20):
#     results = []

#     # Hyperparameter ranges
#     learning_rates = hyperparams['learning_rate']
#     batch_sizes = hyperparams['batch_size']
#     hidden_layers = hyperparams['hidden_layers']

#     kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

#     for lr, batch_size, hidden_layer_sizes in itertools.product(learning_rates, batch_sizes, hidden_layers):
#         fold_metrics = []

#         for train_idx, val_idx in kf.split(data):
#             train_data = torch.tensor(data[train_idx], dtype=torch.float32)
#             val_data = torch.tensor(data[val_idx], dtype=torch.float32)
#             train_targets = torch.tensor(targets[train_idx], dtype=torch.float32)
#             val_targets = torch.tensor(targets[val_idx], dtype=torch.float32)

#             train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=batch_size, shuffle=True)
#             val_loader = DataLoader(TensorDataset(val_data, val_targets), batch_size=batch_size, shuffle=False)

#             # Define the model with customizable hidden layers
#             class CustomModel(nn.Module):
#                 def __init__(self, input_dim, hidden_layer_sizes, output_dim):
#                     super(CustomModel, self).__init__()
#                     layers = []
#                     layers.append(nn.Linear(input_dim, hidden_layer_sizes[0]))
#                     layers.append(nn.BatchNorm1d(hidden_layer_sizes[0]))
#                     layers.append(nn.ReLU())
#                     layers.append(nn.Dropout(0.2))
#                     for i in range(len(hidden_layer_sizes) - 1):
#                         layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
#                         layers.append(nn.BatchNorm1d(hidden_layer_sizes[i + 1]))
#                         layers.append(nn.ReLU())
#                         layers.append(nn.Dropout(0.2))
#                     layers.append(nn.Linear(hidden_layer_sizes[-1], output_dim))
#                     self.model = nn.Sequential(*layers)

#                 def forward(self, x):
#                     return self.model(x)

#             model = CustomModel(input_dim, hidden_layer_sizes, output_dim)
#             criterion = nn.SmoothL1Loss()
#             optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#             scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

#             # Training and validation
#             best_val_loss = float('inf')
#             for epoch in range(epochs):
#                 model.train()
#                 train_loss = 0
#                 for batch_inputs, batch_outputs in train_loader:
#                     optimizer.zero_grad()
#                     predictions = model(batch_inputs)
#                     loss = criterion(predictions, batch_outputs)
#                     loss.backward()
#                     optimizer.step()
#                     train_loss += loss.item()

#                 # Validation phase
#                 model.eval()
#                 val_loss = 0
#                 val_predictions = []
#                 val_targets_list = []
#                 with torch.no_grad():
#                     for batch_inputs, batch_outputs in val_loader:
#                         predictions = model(batch_inputs)
#                         val_loss += criterion(predictions, batch_outputs).item()
#                         val_predictions.append(predictions.numpy())
#                         val_targets_list.append(batch_outputs.numpy())

#                 val_loss /= len(val_loader)
#                 scheduler.step(val_loss)
#                 val_predictions = np.vstack(val_predictions)
#                 val_targets_list = np.vstack(val_targets_list)
#                 mae = mean_absolute_error(val_targets_list, val_predictions)

#                 # Update best validation loss
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss

#             # Record metrics
#             fold_metrics.append((val_loss, mae))

#         # Average metrics over folds
#         avg_val_loss = np.mean([m[0] for m in fold_metrics])
#         avg_mae = np.mean([m[1] for m in fold_metrics])
#         results.append((lr, batch_size, hidden_layer_sizes, avg_val_loss, avg_mae))

#     # Select the best hyperparameters
#     best_params = min(results, key=lambda x: x[3])  # Select based on lowest validation loss
#     print(f"Best Hyperparameters: Learning Rate={best_params[0]}, Batch Size={best_params[1]}, Hidden Layers={best_params[2]}")
#     print(f"Validation Loss: {best_params[3]:.4f}, MAE: {best_params[4]:.4f}")

#     return best_params

# # usage
# hyperparams = {
#     'learning_rate': [0.0005, 0.001, 0.005],
#     'batch_size': [16, 32, 64],
#     'hidden_layers': [[128, 256], [256, 512], [128, 256, 512]]
# }

# best_hyperparameters = tune_hyperparameters_with_validation(
#     data, targets, input_dim=7, output_dim=27, hyperparams=hyperparams, k_folds=5, epochs=20
# )



# Save the recommendations to a file
# top_recommendations.to_csv("final_top_5_recommendations.csv", index=False)
print("hello world")


ds1 = pd.read_csv('mobiles.csv')


# Title of the app
st.title("Mobile Phone Recommendation System")

# Option to choose recommendation method
option = st.radio("How would you like to get recommendations?", 
                  ("Recommend by Phone", "Recommend by Features"))

if option == "Recommend by Phone":
    st.header("Recommend by Phone Title")
    Brand = st.selectbox(
        "Select Brand:",
        ds1['Brand'].unique()
    )
    dataset = ds1[ds1['Brand'] == Brand]
    titles = dataset['Title'].unique()
    Title = st.selectbox(
        "Select Phone:",
        titles
    )
    if st.button("Recommend"):
        st.write("# The Recommended Mobile phone :")
        # Example call to the function
        chosen_index = get_model_index(
            model_number_input=Title,
            original_ds=original_ds,
            reduced_vectors=reduced_vectors
        )


        print(f"Chosen Index: {chosen_index}")
        # Ensure valid index
        if chosen_index < 0 or chosen_index >= len(reduced_vectors):
            raise ValueError(f"Invalid index: {chosen_index}. Please enter a valid index.")

        # Calculate the vector for the chosen mobile
        chosen_vector = reduced_vectors[chosen_index]
        user_preference_vector = chosen_vector.reshape(1, -1)

        # Compute cosine similarity (absolute similarity)
        cosine_similarities = cosine_similarity(reduced_vectors, user_preference_vector).flatten()
        absolute_cosine_similarities = np.abs(cosine_similarities)

        # Compute dot product similarity (absolute similarity)
        dot_product_similarities = np.dot(reduced_vectors, chosen_vector)

        # Normalize dot product similarities for fair comparison
        norm_chosen_vector = np.linalg.norm(chosen_vector)
        norm_vectors = np.linalg.norm(reduced_vectors, axis=1)
        normalized_dot_product_similarities = dot_product_similarities / (norm_chosen_vector * norm_vectors)

        # Apply absolute value for dot product similarity
        absolute_dot_product_similarities = np.abs(normalized_dot_product_similarities)

        # Exclude the chosen index from recommendations
        filtered_indices = [i for i in range(len(cosine_similarities)) if i != chosen_index]

        # Sort the filtered indices by absolute cosine similarity scores
        filtered_cosine_similarities = [(i, absolute_cosine_similarities[i]) for i in filtered_indices]
        sorted_cosine_indices = sorted(filtered_cosine_similarities, key=lambda x: x[1], reverse=True)

        # Sort the filtered indices by absolute dot product similarity scores
        filtered_dot_product_similarities = [(i, absolute_dot_product_similarities[i]) for i in filtered_indices]
        sorted_dot_product_indices = sorted(filtered_dot_product_similarities, key=lambda x: x[1], reverse=True)

        # Get the top recommendations based on absolute similarity
        cosine_top_indices = [idx for idx, _ in sorted_cosine_indices]
        dot_product_top_indices = [idx for idx, _ in sorted_dot_product_indices]

        # Calculate MRR based on dot product similarities (ground truth)
        relevant_found = False
        for rank, index in enumerate(cosine_top_indices):
            if index == dot_product_top_indices[0]:  # Assuming ground truth is the first item in dot product list
                mrr = 1 / (rank + 1)
                relevant_found = True
                break
        if not relevant_found:
            mrr = 0  # If no relevant item is found

        # Display MRR
        print(f"Mean Reciprocal Rank (MRR) based on dot product ground truth: {mrr}")

        # Optional: Display the top recommendations
        cosine_top_recommendations = original_ds.iloc[cosine_top_indices]
        final_recommendations = cosine_top_recommendations.drop_duplicates(subset='Model Name', keep='first').head(5)

        print("\nTop 5 Recommended Mobiles (Unique by 'Model Name'):")
        print(final_recommendations[['Brand', 'Model Name', 'Model Number']])

        # Save the final recommendations to a file
        # final_recommendations.to_csv("final_top_5_recommendations.csv", index=False)
        st.dataframe(final_recommendations)
    

elif option == "Recommend by Features":
    st.header("Recommend by Features")
    
    columns=['Brand', 'Price', 'Primary Camera', 'Processor Type', 'Internal Storage', '5G', 'Battery Capacity']

    Brand = st.selectbox(
        "Select Brand:",
        ['APPLE', 'POCO', 'OnePlus', 'realme', 'vivo', 'MOTOROLA', 'REDMI',
       'Infinix', 'Nokia', 'SAMSUNG', 'OPPO', 'Micromax', 'MarQ', 'LAVA',
       'Google', 'itel', 'Kechaoda', 'HOTLINE', 'Tecno', 'KARBONN', 'I',
       'GFive', 'DIZO', 'Snexian', 'Good', 'Eunity', 'Energizer', 'IAIR',
       'Cellecor', 'IQOO', 'Xiaomi', 'MTR', 'Nothing', 'Mi', 'SAREGAMA',
       'Peace', 'UiSmart', 'Itel']
    )
    Price = st.number_input(
    "Enter Price:", 
    min_value=0,  # Minimum value allowed
    max_value=1000000,  # Maximum value allowed
    step=1,  # Increment step size
    value=0  # Default value
    )
    Primary_Camera = st.number_input(
    "Enter Primary Camera:", 
    min_value=1,  # Minimum value allowed
    max_value=150,  # Maximum value allowed
    step=1,  # Increment step size
    value=1  # Default value
    )

    Processor_Type = st.selectbox(
        "Select Processor Type:",
        ['A15 Bionic Chip', 'Helio G36', 'Dimensity 6100+',
       'Mediatek Helio G85', 'Dimensity 6020', 'Snapdragon 695',
       'Helio G88', 'Unisoc Spreadtrum SC9863A1', 'Helio G85', 'G37',
       'Mediatek Dimensity 930', 'T612', 'Dimensity 7050',
       'Snapdragon 4 Gen 2', 'T616', 'Unisoc T606', 'Dimensity 8020',
       'SC6531E', 'Exynos 850', 'Mediatek Helio G37', 'Dimensity 6080',
       'Dimensity 7020', 'A15 Bionic Chip, 6 Core Processor',
       'A16 Bionic Chip, 6 Core Processor', 'A13 Bionic Chip',
       'NA 0 Single Core 208MHz', 'Spreadtrum \nSC9863A1',
       'Mediatek Helio G99', 'Qualcomm Snapdragon 778G',
       'Qualcomm Snapdragon 680', 'Mediatek Helio G96',
       'Dimensity 6020 5G', 'Mediatek Dimensity 1080',
       'Qualcomm Snapdragon 695', '6833', 'Qualcomm Snapdragon 695 5G',
       'Qualcomm Snapdragon 8+ Gen 1',
       'Mediatek Helio A22 Processor, Upto 2.0 GHz', 'Google Tensor',
       'Helio G99', 'Spreadtrum SC9863A1', 'MediaTek Helio P35',
       '0 0 0 0 Processor', 'Mediatek MT6769Z Helio G85',
       'Mediatek Dimensity 700', 'Exynos 1330, Octa Core',
       'Mediatek Helio A22', 'A17 Pro Chip, 6 Core Processor',
       '0 0 0 208MHz', 'Qualcomm Snapdragon 750G', 'Snapdragon 685',
       'A14 Bionic Chip with Next Generation Neural Engine',
       'Mediatek Helio G35', 'Mediatek Dimensity 920',
       'MediaTek Helio G85', 'Qualcomm Snapdragon 4 Gen 1',
       'MediaTek Helio G88', 'Qualcomm Snapdragon 888 +',
       '2?GHz, Quad Core Processor', 'Mediatek Helio G99 Octa Core',
       'MediaTek G37', 'Snapdragon 778G 5G', 'Exynos 1280',
       'Mediatek Dimensity 1080 5G', 'Snapdragon 680', 'Android 10.0',
       'SEC S5E8535 (Exynos 1330)', 'Dimesity 8050',
       'MediaTek Helio G37 Octa-core Processor', 'SC9863A1',
       'Unisoc 6531F', 'Helio G70 (MT6769)',
       'Qualcomm Snapdragon 7+ Gen 2 (4nm)',
       'Qualcomm Snapdragon 8 Gen 2', 'Unisoc SC9863A/ Unisoc SC9863A1',
       'Unisoc 6531E', 'Mediatek Dimensity 1200', 'T107',
       'Qualcomm Snapdragon 778G Plus', 'Mediatek Dimensity 7200 5G',
       'Mediatek Helio P35', 'Mediatek MT6261D', 'Dimensity 8100',
       'Qualcomm SM6225 Snapdragon 680 4G (6 nm)', 'Helio P35',
       'Spreadtrum', 'Exynos 9825', 'Dimensity 810', 'Unisoc',
       'Mediatek Helio G36', 'Snapdragon 8+ Gen 1', 'Unisoc T107',
       'Dimensity 1080, Octa Core', 'Unisoc 9863A Octa Core',
       'Qualcomm Snapdragon 888 Octa-Core', 'Octa Core',
       'Qualcomm® Snapdragon™ 750G', 'Mediatek G99',
       'Exynos 1380, Octa Core', 'MediaTek Helio A22',
       'Mediatek Dimensity 8050', 'Mediatek Dimensity 8200 (4 nm)',
       'Meditek Helio G37', 'Tensor G2', 'Mediatek Dimensity 8200',
       'SC6531C', 'Qualcomm Snapdragon 8 Gen 1 Mobile Platform',
       'Unisoc T612', 'Qualcomm Snapdragon 778G+', 'MediaTek',
       'Mediatek Helio G95', 'Qualcomm Snapdragon 888', 'Unisoc T616',
       'Mediatek Dimensity 810 5G', 'Qualcomm Snapdragon 662',
       'Exynos Octa Core Processor', 'Unisoc T117', 'Snapdragon 8 Gen 1',
       'Qualcomm Snapdragon 870', 'UNISOC T700', 'Snapdragon 8 Gen 2',
       'Mediatek MT6765 Helio G37', 'Exynos 850 Octa Core',
       'Mediatek Helio G95 Octa Core', 'Mediatek MT6781 Helio G96']
    )
    Internal_Storage = st.selectbox(
        "Select Internal Storage:",
            ['128GB', '64GB', '256GB', '32MB', '32GB', '24MB',
       '4MB', '153MB', '0GB', '64MB', '0.125GB', '128MB', '16MB',
       '32KB', '32+3GB', '1TB', '3MB', '512GB', '0MB', '8MB',
       '10MB', '20MB', '2GB', '56MB', '256MB', '6GB', 
       '8GB', '31MB']
        )
    
    G = st.selectbox(
        "5 G:",
        ['No','Yes']
        )
    Five_G = 0
    if G == 'Yes':
        Five_G = 1
    elif G == 'No':
        Five_G = 0
    
    Battery_Capacity = st.number_input(
    "Enter Battery Capacity:", 
    min_value=500,  # Minimum value allowed
    max_value=7500,  # Maximum value allowed
    step=1,  # Increment step size
    value=3500  # Default value
    )

    if st.button("Recommend"):
        st.write("# The Recommended Mobile phones are :")
        # Load the dataset
        data = np.load("vectors.npy")
        targets = np.load("embedded_vectors.npy")

        # Normalize the data
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        # Split into train and test sets
        train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.2, random_state=42)

        # Convert to PyTorch tensors
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_targets = torch.tensor(train_targets, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_targets = torch.tensor(test_targets, dtype=torch.float32)

        # Create DataLoader
        batch_size = 32
        train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_data, test_targets), batch_size=batch_size, shuffle=False)

        # Define the improved model
        class ImprovedEmbeddingModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(ImprovedEmbeddingModel, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, output_dim)
                )

            def forward(self, x):
                return self.model(x)

        # Initialize the model, loss, and optimizer
        input_dim = data.shape[1]
        output_dim = targets.shape[1]
        model = ImprovedEmbeddingModel(input_dim, output_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Training the model
        epochs = 50

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_inputs, batch_outputs in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_inputs)
                loss = criterion(predictions, batch_outputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

        # Save the trained model
        # torch.save(model.state_dict(), "improved_embedding_model.pth")

        # Evaluating the model
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_inputs, batch_outputs in test_loader:
                predictions = model(batch_inputs)
                all_predictions.append(predictions.numpy())
                all_targets.append(batch_outputs.numpy())

        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)

        # Compute evaluation metrics
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)

        # Calculate the range of the target values
        target_range = np.ptp(all_targets)

        # Calculate normalized errors
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        nrmse = rmse / target_range  # Normalized RMSE


        # Calculate Explained Variance Score (EVS) relative to range
        evs = 1 - np.sum((all_targets - all_predictions) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)
        evs_relative_range = evs / (target_range ** 2)  # EVS relative to range

        # Print results

        print(f"Normalized RMSE: {nrmse:.4f}")
        # print(f"Normalized MAE: {nmae:.4f}")

        # Print accuracy percentages for normalized metrics
        print(f"Accuracy Percentage for Normalized RMSE: {100 - nrmse * 100:.2f}%")
        # print(f"Accuracy Percentage for Normalized MAE: {100 - nmae * 100:.2f}%")

        def map_to_embedding(input_vector):
            """
            Maps a 7-dimensional input vector to a 27-dimensional embedded vector.

            Args:
                input_vector (list or numpy array): A 7-dimensional input vector.

            Returns:
                numpy array: A 27-dimensional embedded vector.
            """
            model.eval()
            input_tensor = torch.tensor(input_vector, dtype=torch.float32)
            with torch.no_grad():
                embedded_vector = model(input_tensor).numpy()
            return embedded_vector

        # Example usage
        new_input = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]  # Example input
        embedded_vector = map_to_embedding(new_input)
        print("Mapped embedded vector:", embedded_vector)

        original_ds = pd.read_csv('orignal_ds.csv')
        brand_encoder = LabelEncoder().fit(original_ds['Brand'])
        processor_encoder = LabelEncoder().fit(original_ds['Processor Type'])
        scaler_2 = MinMaxScaler()

        def preprocess_and_map_input(user_input):
            """
            Process user input for the recommendation system.

            Args:
                user_input (dict): Dictionary containing user inputs with the following keys:
                                - 'Brand'
                                - 'Price'
                                - 'Primary Camera'
                                - 'Processor Type'
                                - 'Internal Storage'
                                - '5G'
                                - 'Battery Capacity'

            Returns:
                np.ndarray: 27-dimensional embedded vector
            """
            # Validate inputs
            if user_input['Brand'] not in original_ds['Brand'].unique():
                raise ValueError(f"Invalid Brand: {user_input['Brand']}")

            if user_input['Processor Type'] not in original_ds['Processor Type'].unique():
                raise ValueError(f"Invalid Processor Type: {user_input['Processor Type']}")

            if not isinstance(user_input['Price'], (int, float)) or user_input['Price'] <= 0:
                raise ValueError("Price must be a positive number.")

            if not isinstance(user_input['Primary Camera'], (int, float)) or user_input['Primary Camera'] <= 0:
                raise ValueError("Primary Camera must be a positive number.")

            if not isinstance(user_input['Battery Capacity'], (int, float)) or user_input['Battery Capacity'] <= 0:
                raise ValueError("Battery Capacity must be a positive number.")

            if not isinstance(user_input['5G'], int) or user_input['5G'] not in [0, 1]:
                raise ValueError("5G must be binary (0 or 1).")

            # Convert internal storage to GB if needed
            if isinstance(user_input['Internal Storage'], str) and user_input['Internal Storage'].endswith('MB'):
                storage_value = float(user_input['Internal Storage'][:-2])
                user_input['Internal Storage'] = storage_value / 1024
            elif isinstance(user_input['Internal Storage'], str) and user_input['Internal Storage'].endswith('GB'):
                user_input['Internal Storage'] = float(user_input['Internal Storage'][:-2])
            elif isinstance(user_input['Internal Storage'], (int, float)):
                user_input['Internal Storage'] = float(user_input['Internal Storage'])
            else:
                raise ValueError("Internal Storage must be a number or a string ending in 'GB' or 'MB'.")

            # Label encode categorical features
            encoded_brand = brand_encoder.transform([user_input['Brand']])[0]
            encoded_processor = processor_encoder.transform([user_input['Processor Type']])[0]
            # Define the correct column names for the features used in scaling
            columns_to_scale = ['Brand','Price','Processor Type', 'Primary Camera', 'Internal Storage', 'Battery Capacity', '5G']

            # Ensure the input features DataFrame has the same column names as the ones used for fitting the scaler
            features = pd.DataFrame([{
                'Brand': encoded_brand,
                'Price': user_input['Price'],
                'Primary Camera': user_input['Primary Camera'],
                'Processor Type': encoded_processor,
                'Internal Storage': user_input['Internal Storage'],
                '5G': user_input['5G'],
                'Battery Capacity': user_input['Battery Capacity']
            }], columns=['Brand', 'Price', 'Primary Camera', 'Processor Type', 'Internal Storage', '5G', 'Battery Capacity'])

            # Standardize the features, passing the DataFrame with feature names
            features_scaled = scaler_1.transform(features[columns_to_scale])
            # Simple Normalization (Unit Norm)
            norm_input_vector = np.linalg.norm(features_scaled)  # Compute the norm of the vector
            features_normalized = features_scaled / norm_input_vector  # Normalize the vector to unit norm


            # Map to 27-dimensional embedding (Assuming you have a function 'map_to_embedding' already)
            embedded_vector = map_to_embedding(features_normalized)

            return embedded_vector

        # Example Usage
        # user_input = {
        #     'Brand': 'SAMSUNG',
        #     'Price': 10999,
        #     'Primary Camera': 50,
        #     'Processor Type': 'Exynos 850',
        #     'Internal Storage': '64GB',
        #     '5G': 1,
        #     'Battery Capacity': 6000
        # }
        user_input = {
            'Brand': Brand,
            'Price': Price,
            'Primary Camera': Primary_Camera,
            'Processor Type': Processor_Type,
            'Internal Storage': Internal_Storage,
            '5G': Five_G,
            'Battery Capacity': Battery_Capacity
        }
        embedded_vector = preprocess_and_map_input(user_input)
        print(embedded_vector)
        embedded_vector=embedded_vector.flatten()
        def recommendations(chosen_vector, embedded_vectors, original_ds, top_n=5):


            # Calculate the vector for the chosen mobile
            user_preference_vector = chosen_vector.reshape(1, -1)

            # Compute cosine similarity (absolute similarity)
            cosine_similarities = cosine_similarity(embedded_vectors, user_preference_vector).flatten()
            absolute_cosine_similarities = np.abs(cosine_similarities)

            # Compute dot product similarity (absolute similarity)
            dot_product_similarities = np.dot(embedded_vectors, chosen_vector)

            # Normalize dot product similarities for fair comparison
            norm_chosen_vector = np.linalg.norm(chosen_vector)
            norm_vectors = np.linalg.norm(embedded_vectors, axis=1)
            normalized_dot_product_similarities = dot_product_similarities / (norm_chosen_vector * norm_vectors)

            # Apply absolute value for dot product similarity
            absolute_dot_product_similarities = np.abs(normalized_dot_product_similarities)

            # Exclude the chosen index from recommendations
            filtered_indices = [i for i in range(len(cosine_similarities)) if i != chosen_index]

            # Sort the filtered indices by absolute cosine similarity scores
            filtered_cosine_similarities = [(i, absolute_cosine_similarities[i]) for i in filtered_indices]
            sorted_cosine_indices = sorted(filtered_cosine_similarities, key=lambda x: x[1], reverse=True)

            # Sort the filtered indices by absolute dot product similarity scores
            filtered_dot_product_similarities = [(i, absolute_dot_product_similarities[i]) for i in filtered_indices]
            sorted_dot_product_indices = sorted(filtered_dot_product_similarities, key=lambda x: x[1], reverse=True)

            # Get the top recommendations based on absolute similarity
            cosine_top_indices = [idx for idx, _ in sorted_cosine_indices]

            # Select the top N recommendations, ensuring they are unique by 'Model Name'
            cosine_top_recommendations = original_ds.iloc[cosine_top_indices]
            final_recommendations = cosine_top_recommendations.drop_duplicates(subset='Model Name', keep='first').head(top_n)

            # Calculate MRR based on dot product similarities (ground truth)
            for rank, index in enumerate(cosine_top_indices):
                if index == sorted_dot_product_indices[0][0]:  # The most "relevant" item based on dot product
                    mrr = 1 / (rank + 1)
                    break
            else:
                mrr = 0  # If the relevant item is not found

            # Novelty: Measures the "novelty" of recommended items, using cosine similarity as a proxy
            novelty = np.mean([np.log2(similarity + 1) for _, similarity in sorted_dot_product_indices[:top_n]])

            # Diversity: Measures the diversity of the recommendations based on embeddings
            diversity = np.mean([1 - cosine_similarity([embedded_vectors[i]], [embedded_vectors[j]])[0][0]
                                for i in range(top_n) for j in range(i + 1, top_n)])

            # Displaying the metrics
            print("\nAccuracy Metrics:")
            print(f"Mean Reciprocal Rank (MRR): {mrr}")
            print(f"Novelty: {novelty}")
            print(f"Diversity: {diversity}")

            return final_recommendations

        top_recommendations = recommendations(embedded_vector, reduced_vectors, original_ds, top_n=5)

        # Display the recommendations
        print("\nTop 5 Recommended Mobiles (Unique by 'Model Name'):")
        print(top_recommendations[['Brand', 'Model Name', 'Model Number']])
        st.dataframe(top_recommendations)


    # # Filter recommendations based on features
    # if selected_features:
    #     feature_mask = phones["Features"].apply(
    #         lambda x: all(feature in x for feature in selected_features)
    #     )
    #     recommendations = phones[feature_mask]
    #     if not recommendations.empty:
    #         st.write("Phones matching your selected features:")
    #         st.table(recommendations)
    #     else:
    #         st.write("No phones match your selected features.")
    # else:
    #     st.write("Please select at least one feature.")



# Footer or additional information
st.sidebar.write("Mobile Phone Recommendation System")