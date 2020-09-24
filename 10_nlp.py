#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from fastai.text.all import *
path = untar_data(URLs.IMDB)


# In[ ]:


files = get_text_files(path, folders = ['train', 'test', 'unsup'])


# In[ ]:


txt = files[0].open().read(); txt[:75]


# In[ ]:


spacy = WordTokenizer()
toks = first(spacy([txt]))
print(coll_repr(toks, 30))


# In[ ]:


first(spacy(['The U.S. dollar $1 is $1.00.']))


# In[ ]:


tkn = Tokenizer(spacy)
print(coll_repr(tkn(txt), 31))


# In[ ]:



txts = L(o.open().read() for o in files[:2000])


# In[ ]:


def subword(sz):
    sp = SubwordTokenizer(vocab_sz=sz)
    sp.setup(txts)
    return ' '.join(first(sp([txt]))[:40])


# In[ ]:


subword(1000)


# In[ ]:


subword(200)


# In[ ]:


toks200 = txts[:200].map(tkn)
toks200[0]


# In[ ]:


num = Numericalize()
num.setup(toks200)
coll_repr(num.vocab,20)


# In[ ]:


nums = num(toks)[:20]; nums


# In[ ]:


#hide_input
bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15:i*15+seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))


# In[ ]:


#hide_input
bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15+seq_len:i*15+2*seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))


# In[ ]:


#hide_input
bs,seq_len = 6,5
d_tokens = np.array([tokens[i*15+10:i*15+15] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))


# In[ ]:


nums200 = toks200.map(num)


# and then passing that to `LMDataLoader`:

# In[ ]:


dl = LMDataLoader(nums200)


# In[ ]:


x,y = first(dl)
x.shape,y.shape


# In[ ]:


get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup'])

dls_lm = DataBlock(
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_imdb, splitter=RandomSplitter(0.1)
).dataloaders(path, path=path, bs=128, seq_len=80)


# In[ ]:


dls_lm.show_batch(max_n=2)


# In[ ]:


learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3, 
    metrics=[accuracy, Perplexity()]).to_fp16()


# In[ ]:


learn.fit_one_cycle(1, 2e-2)


# ### Saving and Loading Models

# You can easily save the state of your model like so:

# In[ ]:


learn.save('1epoch')


# In[ ]:


learn = learn.load('1epoch')


# Once the initial training has completed, we can continue fine-tuning the model after unfreezing:

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(10, 2e-3)


# Once this is done, we save all of our model except the final layer that converts activations to probabilities of picking each token in our vocabulary. The model not including the final layer is called the *encoder*. We can save it with `save_encoder`:

# In[ ]:


learn.save_encoder('finetuned')


# > jargon: Encoder: The model not including the task-specific final layer(s). This term means much the same thing as _body_ when applied to vision CNNs, but "encoder" tends to be more used for NLP and generative models.

# This completes the second stage of the text classification process: fine-tuning the language model. We can now use it to fine-tune a classifier using the IMDb sentiment labels.

# ### Text Generation

# Before we move on to fine-tuning the classifier, let's quickly try something different: using our model to generate random reviews. Since it's trained to guess what the next word of the sentence is, we can use the model to write new reviews:

# In[ ]:


TEXT = "I liked this movie because"
N_WORDS = 40
N_SENTENCES = 2
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) 
         for _ in range(N_SENTENCES)]


# In[ ]:


print("\n".join(preds))


# As you can see, we add some randomness (we pick a random word based on the probabilities returned by the model) so we don't get exactly the same review twice. Our model doesn't have any programmed knowledge of the structure of a sentence or grammar rules, yet it has clearly learned a lot about English sentences: we can see it capitalizes properly (*I* is just transformed to *i* because our rules require two characters or more to consider a word as capitalized, so it's normal to see it lowercased) and is using consistent tense. The general review makes sense at first glance, and it's only if you read carefully that you can notice something is a bit off. Not bad for a model trained in a couple of hours! 
# 
# But our end goal wasn't to train a model to generate reviews, but to classify them... so let's use this model to do just that.

# ### Creating the Classifier DataLoaders

# In[ ]:


dls_clas = DataBlock(
    blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab),CategoryBlock),
    get_y = parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, path=path, bs=128, seq_len=72)


# In[ ]:


dls_clas.show_batch(max_n=3)


# Let's now look at how many tokens each of these 10 movie reviews have:

# In[ ]:


learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, 
                                metrics=accuracy).to_fp16()


# In[ ]:


learn = learn.load_encoder('finetuned')


# In[ ]:


learn.fit_one_cycle(1, 2e-2)


# In just one epoch we get the same result as our training in <<chapter_intro>>: not too bad! We can pass `-2` to `freeze_to` to freeze all except the last two parameter groups:

# In[ ]:


learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))


# Then we can unfreeze a bit more, and continue training:

# In[ ]:


learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))


# And finally, the whole model!

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))


# We reached 94.3% accuracy, which was state-of-the-art performance just three years ago. By training another model on all the texts read backwards and averaging the predictions of those two models, we can even get to 95.1% accuracy, which was the state of the art introduced by the ULMFiT paper. It was only beaten a few months ago, by fine-tuning a much bigger model and using expensive data augmentation techniques (translating sentences in another language and back, using another model for translation).
# 
# Using a pretrained model let us build a fine-tuned language model that was pretty powerful, to either generate fake reviews or help classify them. This is exciting stuff, but it's good to remember that this technology can also be used for malign purposes.

# 1. See what you can learn about language models and disinformation. What are the best language models today? Take a look at some of their outputs. Do you find them convincing? How could a bad actor best use such a model to create conflict and uncertainty?
# 1. Given the limitation that models are unlikely to be able to consistently recognize machine-generated texts, what other approaches may be needed to handle large-scale disinformation campaigns that leverage deep learning?

# In[ ]:




