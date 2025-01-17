import pandas as pd
import random

def load_data():
    # define the column names
    column_names = ['area', 'perimeter', 'compactness', 'length_of_kernel', 'width_of_kernel', 'asymmetry_coefficient', 'length_of_kernel_groove', 'class']
    # read the dataset
    df = pd.read_csv('seeds_dataset.txt', sep=r'\s+', names=column_names)
    return shuffle_data(df)

def shuffle_data(df):
    # shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def split_data_into_features_and_labels():
    # load the dataset
    df = load_data()
    # split the dataset into features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def split_data():
    # load the dataset
    df = load_data()
    # split the dataset into training and testing sets
    train_size = int(0.8 * len(df))
    train_data = df[:train_size]
    test_data = df[train_size:]
    # split the samples into features and labels
    train_features = train_data.iloc[:, :-1].values
    train_labels = train_data.iloc[:, -1].values
    test_features = test_data.iloc[:, :-1].values
    test_labels = test_data.iloc[:, -1].values
    return train_features, train_labels, test_features, test_labels

def remove_random_class(df, class_to_remove):
    # Get the unique classes in the dataset
    classes = df['class'].unique()
    # Select a random class
    # class_to_remove = random.choice(classes)
    print(f"Removing class: {class_to_remove}")
    # Remove rows where the 'class' column equals the randomly selected class
    df = df[df['class'] != class_to_remove].reset_index(drop=True)
    # Get the remaining classes (after removal)
    remaining_classes = df['class'].unique()
    # Create a mapping for the remaining classes to 1 and 2
    class_mapping = {remaining_classes[0]: 1, remaining_classes[1]: 2}
    # Apply the mapping to relabel the classes
    df['class'] = df['class'].map(class_mapping)
    return df

def binary_dataset_split(class_to_remove):
    df = load_data()
    df = remove_random_class(df, class_to_remove)
    df = shuffle_data(df)
    # split the dataset into training and testing sets
    train_size = int(0.8 * len(df))
    train_data = df[:train_size]
    test_data = df[train_size:]
    # split the samples into features and labels
    train_features = train_data.iloc[:, :-1].values
    train_labels = train_data.iloc[:, -1].values
    test_features = test_data.iloc[:, :-1].values
    test_labels = test_data.iloc[:, -1].values
    return train_features, train_labels, test_features, test_labels