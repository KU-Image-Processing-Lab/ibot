import os


def save_parameter(args):
    # ---- save hyperparameters ----
    os.makedirs(f"snapshots/{opt.train_save}", exist_ok=True)
    with open(f"snapshots/{opt.train_save}/hyperparameters.txt", "w") as f:
        for k, v in sorted(vars(opt).items()):
            print("{0}: {1}".format(k, v))
            f.write("{0}: {1}\n".format(k, v))
