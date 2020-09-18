if [[ $2 == 1 ]]; then
    bash run_single_task.sh $1 absa-airy 15 $3
    bash run_single_task.sh $1 absa-prosa 15 $3
    bash run_single_task.sh $1 doc-sentiment-prosa 15 $3
elif [[ $2 == 2 ]]; then
    bash run_single_task.sh $1 emotion-twitter 15 $3
    bash run_single_task.sh $1 entailment-ui 15 $3
    bash run_single_task.sh $1 keyword-extraction-prosa 15 $3
elif [[ $2 == 3 ]]; then 
    bash run_single_task.sh $1 qa-factoid-itb 15 $3
elif [[ $2 == 4 ]]; then
    bash run_single_task.sh $1 ner-grit 15 $3
    bash run_single_task.sh $1 ner-prosa 15 $3
elif [[ $2 == 5 ]]; then
    bash run_single_task.sh $1 pos-idn 15 $3
else
    bash run_single_task.sh $1 term-extraction-airy 15 $3
    bash run_single_task.sh $1 pos-prosa 15 $3
fi