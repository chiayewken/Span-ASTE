## Span-ASTE: Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction

**\*\*\*\*\* New August 30th, 2022: Featured on YouTube video by Xiaoqing Wan [![YT](https://img.shields.io/youtube/views/rRTvsuGRnJ0?style=social)](https://www.youtube.com/watch?v=rRTvsuGRnJ0) \*\*\*\*\***

**\*\*\*\*\* New March 31th, 2022: Scikit-Style API for Easy Usage \*\*\*\*\***

[![PWC](https://img.shields.io/badge/PapersWithCode-Benchmark-%232cafb1)](https://paperswithcode.com/sota/aspect-sentiment-triplet-extraction-on-aste)
[![Colab](https://img.shields.io/badge/Colab-Code%20Demo-%23fe9f00)](https://colab.research.google.com/drive/1F9zW_nVkwfwIVXTOA_juFDrlPz5TLjpK?usp=sharing)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook%20Demo-important)](https://github.com/chiayewken/Span-ASTE/blob/main/demo.ipynb)

This repository implements our ACL 2021 research paper [Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction](https://aclanthology.org/2021.acl-long.367/). 
Our goal is to extract sentiment triplets of the format `(aspect target, opinion expression and sentiment polarity)`, as shown in the diagram below. 

<img src="https://github.com/chiayewken/Span-ASTE/blob/13a851b166998210a7cd2def5fa4aff20819b54d/assets/task_image.png" width="450" height="150" alt="">

### Installation

- Tested on Python 3.7 (recommended to use a virtual environment such as [Conda](https://docs.conda.io/en/latest/miniconda.html))
- Install data and requirements: `bash setup.sh`
- Training config: [training_config/config.jsonnet](training_config/config.jsonnet)
- Modeling code: [span_model/models/span_model.py](span_model/models/span_model.py)

### Data Format

Our span-based model uses data files where the format for each line contains one input sentence and a list of output triplets.
The following data format is demonstrated in the [sample data file](sample_data.txt):

> sentence#### #### ####[triplet_0, ..., triplet_n]

Each triplet is a tuple that consists of `(span_a, span_b, label)`. Each span is a list. If the span covers a single word, the list will contain only the word index. If the span covers multiple words, the list will contain the index of the first word and last word. For example:

> It also has lots of other Korean dishes that are affordable and just as yummy .#### #### ####[([6, 7], [10], 'POS'), ([6, 7], [14], 'POS')]

For prediction, the data can contain the input sentence only, with an empty list for triplets:

> sentence#### #### ####[]

### Predict Using Model Weights

- First, download and extract [pre-trained weights](https://github.com/chiayewken/Span-ASTE/releases) to `pretrained_dir`
- The input data file `path_in` and output data file `path_out` have the same [data format](#data-format).

```
from wrapper import SpanModel

model = SpanModel(save_dir=pretrained_dir, random_seed=0)
model.predict(path_in, path_out)
```

### Model Training

- Configure the model with save directory and random seed.
- Start training based on the training and validation data which have the same [data format](#data-format).

```
model = SpanModel(save_dir=save_dir, random_seed=random_seed)
model.fit(path_train, path_dev)
```

- To train with multiple random seeds from the command-line, you can use the following command:
- Replace `14lap` for other datasets (eg `14res`, `15res`, `16res`)

```
python aste/wrapper.py run_train_many \
--save_dir_template "outputs/14lap/seed_{}" \
--random_seeds [0,1,2,3,4] \
--path_train data/triplet_data/14lap/train.txt \
--path_dev data/triplet_data/14lap/dev.txt
```

### Model Evaluation

- From the trained model, predict triplets from the test sentences and output into `path_pred`.
- The model includes a scoring function which will provide F1 metric scores for triplet extraction.

```
model.predict(path_in=path_test, path_out=path_pred)
results = model.score(path_pred, path_test)
```

- To evaluate with multiple random seeds from the command-line, you can use the following command:
- Replace `14lap` for other datasets (eg `14res`, `15res`, `16res`)

```
python aste/wrapper.py run_eval_many \
--save_dir_template "outputs/14lap/seed_{}" \
--random_seeds [0,1,2,3,4] \
--path_test data/triplet_data/14lap/test.txt
```

- To run scoring on your own prediction file from command-line, you can use the following command:
- Replace `14lap` for other datasets (eg `14res`, `15res`, `16res`)

```
python aste/wrapper.py run_score --path_pred your_file --path_gold data/triplet_data/14lap/test.txt
```

### Research Citation
If the code is useful for your research project, we appreciate if you cite the following paper:
```
@inproceedings{xu-etal-2021-learning,
    title = "Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction",
    author = "Xu, Lu  and
      Chia, Yew Ken  and
      Bing, Lidong",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.367",
    doi = "10.18653/v1/2021.acl-long.367",
    pages = "4755--4766",
    abstract = "Aspect Sentiment Triplet Extraction (ASTE) is the most recent subtask of ABSA which outputs triplets of an aspect target, its associated sentiment, and the corresponding opinion term. Recent models perform the triplet extraction in an end-to-end manner but heavily rely on the interactions between each target word and opinion word. Thereby, they cannot perform well on targets and opinions which contain multiple words. Our proposed span-level approach explicitly considers the interaction between the whole spans of targets and opinions when predicting their sentiment relation. Thus, it can make predictions with the semantics of whole spans, ensuring better sentiment consistency. To ease the high computational cost caused by span enumeration, we propose a dual-channel span pruning strategy by incorporating supervision from the Aspect Term Extraction (ATE) and Opinion Term Extraction (OTE) tasks. This strategy not only improves computational efficiency but also distinguishes the opinion and target spans more properly. Our framework simultaneously achieves strong performance for the ASTE as well as ATE and OTE tasks. In particular, our analysis shows that our span-level approach achieves more significant improvements over the baselines on triplets with multi-word targets or opinions.",
}
```
