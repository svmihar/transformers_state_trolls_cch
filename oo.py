from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel, ClassificationArgs

import pandas as pd
df = pd.read_csv("./data/train_raw.csv")

data = df[["clean_text", "troll_or_not"]]
data.columns= ["text", "labels"]
data['labels']=data['labels'].apply(lambda x: int(x))
data.dropna(inplace=True)

train_df, test = train_test_split(data, test_size=0.3)


model_to_test = {
    'bert': 'bert-base-cased', 
    'distilroberta': 'distilroberta-base',
    'distilbert': 'distilbert-base-multilingual-cased', 
    'electra':'google/electra-small-discriminator'
}
def train(arch, model_name): 
    model_args = ClassificationArgs(
        num_train_epochs=3,
        output_dir='./models',
        evaluate_during_training_steps=1000,
        train_batch_size=128, 
        reprocess_input_data=True,
        evaluate_during_training=True, 
        eval_batch_size=64,
        save_model_every_epoch=False,
        overwrite_output_dir=True,
        save_eval_checkpoints=False, 
        best_model_dir=f'./models/{model_name}/best_model',
        wandb_kwargs={'name': model_name}
    )

    model = ClassificationModel(arch, model_name, args=model_args)

    model.train_model(train_df, eval_df=test)

    result, model_output, top_loss = model.eval_model(test)
    print(result)

    pred, raw_output = model.predict(["thanks for bearing with us"])
    print(pred)

if __name__ == "__main__":
    m = 'distilbert'
    train(m,model_to_test[m])
