for dset in ./save/*; do
  if [ -d $dset ]; then
    for exp in $dset/*; do
      if [ -d $exp ]; then
        dset=$(echo "$dset" | rev | cut -d/ -f1 | rev)
        exp=$(echo "$exp" | rev | cut -d/ -f1 | rev)
        type=$(echo "$exp" | cut -d_ -f1)
        echo "$dset ~ $exp ~ $type"
        
        CUDA_VISIBLE_DEVICES=$1 python3 predict.py --experiment_name $exp --dataset $dset --model_type $type --batch_size 12 --lower
      fi
    done
    break
  fi
done
