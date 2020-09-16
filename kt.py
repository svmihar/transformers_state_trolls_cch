from oo import train_df,test
import ktrain 
from ktrain import text
MODEL_NAME = 'distilbert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=500, class_names=[1,0])
trn = t.preprocess_train(train_df['text'].values, train_df['labels'].values)
val = t.preprocess_test(test['text'].values, test['labels'].values)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)

learner.lr_find(show_plot=True, max_epochs=2, suggest=True)
# mg, ml = learner.estimate_lr()
learner.fit_onecycle(8e-5, 2)
learner.validate(class_names=t.get_classes())

print(learner.view_top_losses(n=10, preproc=t))