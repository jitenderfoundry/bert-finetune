import shutil
import os
from truefoundry.ml import get_client
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from data_validation import validate_classification_data


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' has been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")


def create_folder(folder_path):
    try:
        os.mkdir(folder_path)
        print(f"Folder '{folder_path}' has been created.")
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")
    except Exception as e:
        print(f"Error: {e}")


def download_training_data(fqn):
    if not os.path.exists("train_data"):
        create_folder("train_data")
    try:
        client = get_client()
        train_artifact_version = client.get_artifact_version_by_fqn(fqn=fqn)
        train_download_path = train_artifact_version.download(path="train_data", overwrite=True)
        
        # Check and rename the downloaded file to train_set.csv if necessary
        files_dir = os.path.join("train_data", "files")
        downloaded_files = os.listdir(files_dir)
        if downloaded_files:
            downloaded_file = os.path.join(files_dir, downloaded_files[0])
            if os.path.basename(downloaded_file) != "train_set.csv":
                new_file_name = os.path.join(files_dir, "train_set.csv")
                os.rename(downloaded_file, new_file_name)
                print(f"Downloaded file renamed to train_set.csv")
            else:
                print("File is already named train_set.csv")
        else:
            print("No files found in the download directory")
        
    except Exception as e:
        print(f"Error in downloading or processing the training data: {e}")


def download_testing_data(fqn):
    if not os.path.exists("test_data"):
        create_folder("test_data")
    try:
        client = get_client()
        test_artifact_version = client.get_artifact_version_by_fqn(fqn=fqn)
        test_download_path = test_artifact_version.download(path="test_data",overwrite=True)
    except Exception as e:
        print(f"Error in downloading the testing data: {e}")
        

def preprocess_function(example,tokenizer,classes):
    class2id = {class_:id for id, class_ in enumerate(classes)}
    all_labels = ast.literal_eval(example["categories"])
    labels = [0. for i in range(len(classes))]
    for label in all_labels:
        label_id = class2id[label]
        labels[label_id] = 1
    example = tokenizer(example['query'], truncation=True, max_length=512)
    example['labels'] = labels
    return example

def process_data(train_path,test_path,tokenizer,text_size=0.2):

    download_training_data(train_path)
    train_data = pd.read_csv("train_data/files/train_set.csv")
    
    #validating the data format
    validate_classification_data(train_data)
    if test_path is None or test_path == "None":
        df = pd.read_csv("train_data/files/train_set.csv")
        train_df, test_df = train_test_split(df, test_size=text_size, random_state=42)
        train_df.to_csv('train_data/files/train_set.csv', index=False)
        if not os.path.exists("test_data"):
            create_folder("test_data")
        create_folder("test_data/files")
        test_df.to_csv('test_data/files/test_set.csv', index=False)
        df.to_csv('train_data/files/complete_data.csv', index=False)
    else:
        download_testing_data(test_path)
        test_data = pd.read_csv("test_data/files/test_set.csv")
        validate_classification_data(test_data)

    classes = list(set([item for sublist in train_data['categories'].apply(ast.literal_eval) for item in sublist]))

    dataset = load_dataset(
                        'csv',
                        data_files = {
                           'train':f"train_data/files/train_set.csv",
                            'test': f"test_data/files/test_set.csv"
                            },
                      )
    
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer=tokenizer, classes=classes))
    return tokenized_dataset , classes


def download_model_file(fqn):
    if not os.path.exists("model"):
        create_folder("model")
    try:
        client = get_client()
        model_artifact_version = client.get_artifact_version_by_fqn(fqn=fqn)
        model_download_path = model_artifact_version.download(path="model", overwrite=True)
        
        # Rename the downloaded folder to "downloaded_model"
        files_dir = os.path.join("model", "files")
        if os.path.exists(files_dir):
            new_folder_name = os.path.join("model", "downloaded_model")
            os.rename(files_dir, new_folder_name)
            print(f"Downloaded folder renamed to 'downloaded_model'")
        else:
            print("No 'files' folder found in the download directory")
    except Exception as e:
        print(f"Error in downloading or processing the existing Model: {e}")