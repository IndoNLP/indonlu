fork-setup:
	git remote add upstream https://github.com/indobenchmark/indonlu.git
	git remote -v

HYPERPARAMETER ?= default
EARLY_STOP ?= 15
BATCH_SIZE ?= 16

.PHONY : train

train:
	python3 scripts/trainer.py $(DATASET) $(EARLY_STOP) $(BATCH_SIZE) $(HYPERPARAMETER)

train_all_1:
	python3 scripts/trainer.py absa-airy 15 $(BATCH_SIZE) $(HYPERPARAMETER)
	python3 scripts/trainer.py absa-prosa 15 $(BATCH_SIZE) $(HYPERPARAMETER)
	python3 scripts/trainer.py doc-sentiment-prosa 15 $(BATCH_SIZE) $(HYPERPARAMETER)

train_all_2:
	python3 scripts/trainer.py emotion-twitter 15 $(BATCH_SIZE) $(HYPERPARAMETER)
	python3 scripts/trainer.py entailment-ui 15 $(BATCH_SIZE) $(HYPERPARAMETER)
	python3 scripts/trainer.py keyword-extraction-prosa 15 $(BATCH_SIZE) $(HYPERPARAMETER)

train_all_3:
	python3 scripts/trainer.py qa-factoid-itb 15 $(BATCH_SIZE) $(HYPERPARAMETER)

train_all_4:
	python3 scripts/trainer.py ner-grit 15 $(BATCH_SIZE) $(HYPERPARAMETER)
	python3 scripts/trainer.py ner-prosa 15 $(BATCH_SIZE) $(HYPERPARAMETER)

train_all_5:
	python3 scripts/trainer.py pos-idn 15 $(BATCH_SIZE) $(HYPERPARAMETER)

train_all_6:
	python3 scripts/trainer.py term-extraction-airy 15 $(BATCH_SIZE) $(HYPERPARAMETER)
	python3 scripts/trainer.py pos-prosa 15 $(BATCH_SIZE) $(HYPERPARAMETER)

run_non_pretrained_no_special_token:
	python3 scripts/trainer_non_pretrained.py $(DATASET) $(EARLY_STOP) $(BATCH_SIZE)

run_non_pretrained_no_special_token_all:
	python3 scripts/trainer_non_pretrained.py emotion-twitter 10 16
	python3 scripts/trainer_non_pretrained.py pos-idn 10 16
	python3 scripts/trainer_non_pretrained.py ner-grit 10 16
	python3 scripts/trainer_non_pretrained.py absa-airy 10 16
	python3 scripts/trainer_non_pretrained.py term-extraction-airy 10 16
	python3 scripts/trainer_non_pretrained.py entailment-ui 10 16
	python3 scripts/trainer_non_pretrained.py doc-sentiment-prosa 10 16
	python3 scripts/trainer_non_pretrained.py keyword-extraction-prosa 10 16
