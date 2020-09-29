# IndoNLU

IndoNLU is a collection of Natural Language Understanding (NLU) resources for Bahasa Indonesia.

## 12 Downstream Tasks
- You can check [[Link]](https://github.com/indobenchmark/indonlu/tree/master/dataset)
- We provide train, valid, and test set (with masked labels, no true labels). We are currently preparing a platform for auto-evaluation using Codalab. Please stay tuned!

## Examples
- A guide to load IndoBERT model and finetune the model on Sequence Classification and Sequence Tagging task.
- You can check [[Link]](https://github.com/indobenchmark/indonlu/tree/master/examples)

## Indo4B
- 23GB Indo4B Pretraining Dataset [[Link]](https://storage.googleapis.com/babert-pretraining/IndoNLU_finals/dataset/preprocessed/dataset_all_uncased_blankline.txt.xz)

## IndoBERT models
- 8 IndoBERT Pretrained Language Model [[Link]](https://huggingface.co/indobenchmark)
  - IndoBERT-base
    - Phase 1  [[Link]](https://huggingface.co/indobenchmark/indobert-base-p1)
    - Phase 2  [[Link]](https://huggingface.co/indobenchmark/indobert-base-p2)
  - IndoBERT-large
    - Phase 1  [[Link]](https://huggingface.co/indobenchmark/indobert-large-p1)
    - Phase 2  [[Link]](https://huggingface.co/indobenchmark/indobert-large-p2)
  - IndoBERT-lite-base
    - Phase 1  [[Link]](https://huggingface.co/indobenchmark/indobert-lite-base-p1)
    - Phase 2  [[Link]](https://huggingface.co/indobenchmark/indobert-lite-base-p2)
  - IndoBERT-lite-large
    - Phase 1  [[Link]](https://huggingface.co/indobenchmark/indobert-lite-large-p1)
    - Phase 2  [[Link]](https://huggingface.co/indobenchmark/indobert-lite-large-p2)

## Leaderboard
- Community Portal and Public Leaderboard [[Link]](https://www.indobenchmark.com/leaderboard.html)
- Submission Portal https://competitions.codalab.org/competitions/26537

### Submission Format
Please kindly check [[Link]](https://github.com/indobenchmark/indonlu/tree/master/submission_examples). For each task, there is different format. Every submission file always start with the `index` column (the id of the test sample following the order of the masked test set). 

First you need to rename your prediction into 'pred.txt', then zip the file.

## Paper
IndoNLU has been accepted on AACL 2020 and you can find the detail on https://arxiv.org/abs/2009.05387
If you are using any component on IndoNLU for research purposes, please cite the following paper:
```
@inproceedings{wilie2020indonlu,
  title={IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding},
  author={Bryan Wilie and Karissa Vincentio and Genta Indra Winata and Samuel Cahyawijaya and X. Li and Zhi Yuan Lim and S. Soleman and R. Mahendra and Pascale Fung and Syafri Bahar and A. Purwarianti},
  booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
  year={2020}
}
```
