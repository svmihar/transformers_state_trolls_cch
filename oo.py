from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from simpletransformers.classification import ClassificationModel, ClassificationArgs

import pandas as pd

sampling = False

df = pd.read_csv("./data/train_raw.csv")
if sampling:
    df = df.sample(20000)

val = pd.read_csv("./data/validate.csv")
val = val[["clean_text", "troll_or_not"]]
val.columns = ["text", "labels"]
data = df[["clean_text", "troll_or_not"]]
data.columns = ["text", "labels"]

val["labels"] = val["labels"].apply(lambda x: int(x))
data["labels"] = data["labels"].apply(lambda x: int(x))
data.dropna(inplace=True)
val.dropna(inplace=True)

train_df, test = train_test_split(data, test_size=0.3)


model_to_test = {
    "bert": "bert-base-multilingual-cased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-multilingual-cased",
    "electra": "google/electra-base-discriminator",
}


def train(arch, model_name,):
    model_args = ClassificationArgs(
        num_train_epochs=5,
        output_dir="./models",
        evaluate_during_training_steps=1000,
        train_batch_size=64,
        reprocess_input_data=True,
        evaluate_during_training=True,
        eval_batch_size=32,
        save_model_every_epoch=False,
        overwrite_output_dir=True,
        learning_rate=7e-5,
        save_eval_checkpoints=False,
        best_model_dir=f"./models/{model_name}/best_model",
        use_early_stopping=True,
        early_stopping_delta=1e-2,
        early_stopping_metric="mcc",
        tensorboard_dir='./runs/',
        early_stopping_metric_minimize=False,
        wandb_project='my_roberta',
        manual_seed=69,
        early_stopping_patience=5,
    )
    model = ClassificationModel(arch, model_name, args=model_args, use_cuda=True)

    model.train_model(
        train_df,
        eval_df=test,
        accuracy=lambda x, y: accuracy_score(x, [round(a) for a in y]),
    )

    result, model_output, top_loss = model.eval_model(test)
    print(result)
    print(top_loss)

    pred, _ = model.predict(["thanks for bearing with us"])
    print(pred)


if __name__ == "__main__":

    m = "electra"

    train(m, model_to_test[m])
