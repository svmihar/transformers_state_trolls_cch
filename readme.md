---
title: troll or not troll with transformers
date: 2020-09-25
draft: false
categories : [dev]
description: "comparing bunch transformer models"
---

# Troll or Not Troll?
- [Troll or Not Troll?](#troll-or-not-troll)
  - [- reference](#--referencereference)
  - [Intro](#intro)
  - [Motivation](#motivation)
  - [Approach](#approach)
  - [Methods](#methods)
    - [preprocessing](#preprocessing)
    - [Dataset Exploration](#dataset-exploration)
      - [word length](#word-length)
    - [hyperparameters](#hyperparameters)
  - [Results and conclusion](#results-and-conclusion)
    - [accuracy score](#accuracy-score)
    - [mcc](#mcc)
    - [training time](#training-time)
    - [conclusion](#conclusion)
  - [reference](#reference)
---
## Intro
The field of NLP was revolutionized in the year 2018 by introduction of BERT and his Transformer friends(RoBerta, XLM etc.).

These novel transformer based neural network architectures and new ways to training a neural network on natural language data introduced transfer learning to NLP problems. Transfer learning had been giving out state of the art results in the Computer Vision domain for a few years now and introduction of transformer models for NLP brought about the same paradigm change in NLP.

## Motivation
Despite these amazing technological advancements applying these solutions to business problems is still a challenge given the niche knowledge required to understand and apply these method on specific problem statements. Hence, In the experiment I will be testing a few transformer architecture,  on a relatively big dataset, and how it performs on shorter text with a humongous corpus.

Before i proceed i would like to mention the following groups for the fantastic work they are doing and sharing which have made these experiments possible.

Please review these amazing sources of information:
- [Hugging Face Team](https://huggingface.co/)
- [simpletransformers](https://simpletransformers.ai/)

## Approach
I'll be looking at a few models, and compare it with the standard models along the way. Then all of the models will be evaluated with `accuracy_score` and `matthews_corrcoef`. The models that are tested are:
- distilbert-base-multilingual
- bert-base-multilingual
- roberta-base
- electra-small
- electra-base

## Methods
### preprocessing
the dataset below has been preprocessed by removing:
  - punctuations
  - non alphanum character (numbers excluded)
see [this script](preprocess.py)
### Dataset Exploration
all the these methods are inside the [this notebook][notebook/eda.ipynb]
with the total of 89k tweets, which are splitted in to train and test, and 2 classes, we can see the distribution between the classes are balanced.
![](https://i.imgur.com/P02wbfQ.png)
whereas class 0 has 44939 tweets, and class 1 has 45009 tweets
#### word length
by eyeballing on the length of each tweet, we can see that the tweet legnth on each class has a distinguishable differences.
![](https://i.imgur.com/l32qjan.png)
### hyperparameters
## Results and conclusion
All the training performance, losses and metrics are recorded to wandb. Results are [here](https://wandb.ai/yessyvita/my_roberta/reports/bert-base-multilingual--VmlldzoyNTM2MDg) if you prefer to see it for yourself.
### accuracy score
![](https://i.imgur.com/pjZW3M2.png)

baseline metric for prediction, by comparing its true label and the prediction result
### mcc
![](https://i.imgur.com/SLEbOEs.png)

Matthews correlation coeffienct takes account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.

The MCC is in essence a correlation coefficient value between -1 and +1. A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction. The statistic is also known as the phi coefficient.
### training time
![](https://i.imgur.com/GFqeBa6.png)

as you can see roberta, takes the longest to train (~ 9 hours) while bert and electra are way faster (~1 hour).
### conclusion
in this experiment we can see that `roberta-base` almost outpeforms every other model although ELECTRA came pretty close despite almost half the parameters trained. while the GLUE Benchmark results has `electra-base` as the state of the art, it took a while to find a good hyperparameters by using a good annealing factor. `roberta-base` gain advantage on shorter text, with a big number of corpus. Meanwhile `electra-base` came in pretty close in the second place, with a good hyperparameter optimization, it could surpass `roberta-base` for sure. While `roberta-base` excels in this kind of task, `roberta-base` is really resource intensive compared to `electra-base`

## reference
- [electra-base](https://openreview.net/pdf?id=r1xMH1BtvB)
- [Transformers Documentation](https://huggingface.co/transformers/pretrained_models.html)
- [Pytorch documentation](https://pytorch.org/docs/stable/index.html)
- [BERT Research Series from ChrisMcCormickAI](https://www.youtube.com/watch?v=FKlPCK1uFrc)
