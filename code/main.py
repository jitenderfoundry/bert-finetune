from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import evaluate
import argparse
from utility import process_data , delete_folder , download_model_file
import os
import truefoundry.ml as tfm
from truefoundry.ml import get_client ,ModelFramework
from datetime import datetime, timezone
from mlfoundry_utils import sanitize_name , MLFoundryCallback
# from config import *


TFY_INTERNAL_JOB_RUN_NAME = os.getenv("TFY_INTERNAL_JOB_RUN_NAME")
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.6).astype(int).reshape(-1)
    return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))


def train_model(train_path : str, test_path : str , model_path : str, job_type : str, model_name : str, ml_repo : str, epochs : int = 5, lr : float = 2e-5, existing_model : str = None, per_device_train_batch_size : int = 32, per_device_eval_batch_size : int = 32):
    # print(epochs)
    # tfm.login(tracking_uri = URL,relogin=True ,api_key=API_KEY)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
     

    tokenized_dataset , classes = process_data(train_path,
                                               test_path,
                                               tokenizer=tokenizer)
    
    class2id = {class_:id for id, class_ in enumerate(classes)} 
    id2class = {id:class_ for id, class_ in enumerate(classes)} 

    
    if existing_model and existing_model != "None":
        download_model_file(existing_model)
        model = AutoModelForSequenceClassification.from_pretrained(f"model/files/downloaded_model/")
    else:
        if job_type == "multi_label_classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                                                                    model_path, 
                                                                    num_labels=len(classes), 
                                                                    id2label=id2class, 
                                                                    label2id=class2id, 
                                                                    problem_type = "multi_label_classification"
                                                                )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                                                                    model_path, 
                                                                    num_labels=len(classes), 
                                                                    id2label=id2class, 
                                                                    label2id=class2id
                                                                )
    client = get_client()
    if TFY_INTERNAL_JOB_RUN_NAME:
        fallback_run_name = f"finetune-{sanitize_name(TFY_INTERNAL_JOB_RUN_NAME)}"
    else:
        fallback_run_name = f"finetune-{timestamp}"
    # create a run
    

    training_args = TrainingArguments(
    output_dir="checkpoints",
    learning_rate=lr,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=epochs,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to='none',
    load_best_model_at_end=True
    )
    run = client.create_run(ml_repo=ml_repo, run_name=fallback_run_name)
    run.log_params({
        "learning_rate": lr,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
    })
  
    


    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_dataset["train"],
                        eval_dataset=tokenized_dataset["test"],
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics,
                        callbacks=[MLFoundryCallback(run)]
                    )
    
    trainer.train()

    trainer.save_model("trained-model")
    print("Model saved")
    model_version = run.log_model(
        name=model_name,
        model_file_or_folder="trained-model/",
        framework=ModelFramework.TRANSFORMERS 
        )
    
    # model_version = client.log_model(
    #     ml_repo=ml_repo,
    #     name=model_name,
    #     model_file_or_folder="trained-model/",
    #     framework=ModelFramework.TRANSFORMERS
    # )
    
    print("Model logged")
    delete_folder("trained-model")
    if  os.path.exists("model"):
        delete_folder("model")

    


if __name__ == "__main__":
    
    # Setup the argument parser by instantiating `ArgumentParser` class
    parser = argparse.ArgumentParser()
    # Add the parameters as arguments
    parser.add_argument(
        '--train_path', 
        type=str,
        required=True, 
        help='Path to the train CSV file / FQN of the train CSV file.Also name the file as train_set.csv'
    )
    parser.add_argument(
        '--test_path',
        type=str, 
        help='Path to save the processed file.Also name the file as train_set.csv'
    )
    parser.add_argument(
        '--model_path', 
        type=str,
        required=True,  
        help='Path of hugging face model'
    )
    
    parser.add_argument(
        '--job_type', 
        type=str, 
        required=True, 
        help='Specify what kind of job you want to run like classification, multi-class classification, ner etc.'
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        help='Name of model in which it will be saved in truefoundry'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        help='Numbers of times you want to train your model on the complete dataset'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        help='Learning rate in the model'
    )
    parser.add_argument(
        "--ml_repo",
        type=str,
        required=True,
        help="""\
            The name of the ML Repo to track metrics and models.
            You can create one from the ML Repos Tab on the UI.
            Docs: https://docs.truefoundry.com/docs/key-concepts#creating-an-ml-repo,
        """
    )
    parser.add_argument(
        "--existing_model",
        type=str,
        help= """
            The artifact url of the existing model to be used for training.
        """
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        help="Per device train batch size"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        help="Per device eval batch size"
    )
    
    args = parser.parse_args()

    # Train the model
    train_model(**vars(args))
    

    

###
## python main.py --train_path artifact:truefoundry/CVS/data:latest --model_path roberta-base --job_type multi_label_classification --model_name mlclassifier  --ml_repo CVS-IVR --epochs 5 --lr 2e-5