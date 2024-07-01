This is the input data used to train and/or evaluate the models in the publication Fine-Tuning with Divergent Chains of Thought Boosts Reasoning Through Self-Correction in Language Models.

# In-Domain Datasets

We used the following datasets for training and evaluation.

| Dataset   |         Task         | Train |  Dev | Test |    License   |                                       Source                                      |
|-----------|:--------------------:|:-----:|:----:|:----:|:------------:|:---------------------------------------------------------------------------------:|
| ARC       |    Multiple choice   |  1033 |  294 | 1150 | CC BY-SA 4.0 |            https://huggingface.co/datasets/allenai/ai2_arc           |
| BGQA      |    Multiple choice   |  716  |  500 | 1000 |     CC BY    | https://storage.googleapis.com/gresearch/BoardgameQA/BoardgameQA.zip |
| Coin Flip |    Multiple choice   |  1000 | 1333 | 3333 |      mit     |          https://huggingface.co/datasets/skrishna/coin_flip          |
| CQA       |    Span extraction   |  958  |  285 |  804 | CC BY-SA 4.0 |             https://haitian-sun.github.io/conditionalqa/             |
| GSM8K     | Generation (numbers) |  1000 |  500 | 1319 |      mit     |             https://huggingface.co/datasets/openai/gsm8k             |
| HQA       |    Span extraction   |  1000 |  500 | 7405 | CC BY-SA 4.0 |                      https://hotpotqa.github.io/                     |
| LLC       |      Generation      |  350  |  50  |  100 |      N/A     |       https://huggingface.co/datasets/ChilleD/LastLetterConcat       |
| Quartz    |    Multiple choice   |  953  |  384 |  784 | CC BY-SA 4.0 |            https://huggingface.co/datasets/allenai/quartz            |
| StrQA     |      Boolean QA      |  998  |  343 |  344 |      mit     |          https://huggingface.co/datasets/ChilleD/StrategyQA          |


For each of these datasets, we generate a `cot_dataset.json`, which contains a list of (question, cot1, cot2, cot3, cot4). The number of CoTs/question can vary between 1 and 4.


- The folder `dcot_collection` contains 2 files:
    - `cot9_dataset.json`: The final version of our DCoT instruction tuning dataset. It aggregates the `cot_dataset.json` of each dataset. This is the training set we used for our models.
    - `cot9_dataset_900.json`: Is a subset of the previous one we used for training LLaMA 70B.


# Out-of-Domain Evaluation Dataset

We used the following datasets for evaluation in out-of-domain scenarios.

| Dataset        |              Task             |  Dev |   License  |                               Source                              |
|----------------|:-----------------------------:|:----:|:----------:|:-----------------------------------------------------------------:|
| AQuA           |        Multiple choice        |  254 | Apache 2.0 |                 https://github.com/google-deepmind/AQuA       |
| CSQA           |        Multiple choice        | 1220 |     mit    |           https://huggingface.co/datasets/tau/commonsense_qa/ |
| SVAMP          |      Generation (numbers)     |  100 |     mit    |                   https://github.com/arkilpatel/SVAMP         |
| Big Bench Hard | Multiple choice \& Generation | 6511 |     mit    |          https://huggingface.co/datasets/maveriq/bigbenchhard |


# Downloading the Datasets

- Running the script `../download_eval_data.sh` will download the datasets in each folder.
