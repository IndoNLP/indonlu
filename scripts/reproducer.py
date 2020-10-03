import os
import sys
# import subprocess
import yaml

CUDA = os.getenv("CUDA_VISIBLE_DEVICES", "0")

path = "scripts/config/model/train.yaml"
with open(path, "r") as f:
    model_configs = yaml.safe_load(f)

hyperparams_config = sys.argv[4]
path = f"scripts/config/hyperparameter/{hyperparams_config}.yaml"
with open(path, "r") as f:
    hyperparams = yaml.safe_load(f)
hyperparams["dataset"] = sys.argv[1]
hyperparams["early_stop"] = sys.argv[2]
hyperparams["train_batch_size"] = sys.argv[3]

hyp_list = [
    "n_epochs",
    "train_batch_size",
    "model_checkpoint",
    "step_size",
    "gamma",
    "experiment_name",
    "lr",
    "early_stop",
    "dataset",
]
for m in model_configs:
    hyperparams["model_checkpoint"] = m["model_checkpoint"]
    for layer in m["num_layers"]:
        exp = [
            hyperparams["model_checkpoint"],
            f"b{hyperparams['train_batch_size']}",
            f"step{hyperparams['step_size']}",
            f"gamma{hyperparams['gamma']}",
            f"lr{hyperparams['lr']}",
            f"early{hyperparams['early_stop']}",
            f"layer{layer}",
            f"lower{m['lower']}"
        ]
        hyperparams["experiment_name"] = "_".join(exp)
        
        cmd = f"CUDA_VISIBLE_DEVICES={CUDA} python3 main.py"
        for hl in hyp_list:
            cmd += f" --{hl} {hyperparams[hl]}"
        if m["lower"]:
            cmd += " --lower"
        cmd += f" --num_layers {layer}"
        for o in hyperparams["options"]:
            cmd += f" {o}"

        # # debug
        # print(cmd)

        # # run in parallel
        # results = subprocess.run(
        #     cmd, shell=True, universal_newlines=True, check=True, text=True)

        os.system(cmd)
