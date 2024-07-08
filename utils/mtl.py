from collections import defaultdict
from utils.weight_methods import METHODS

def extract_weight_method_parameters_from_args():
    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=1,
                optim_niter=20,
            ),
            stl=dict(main_task=1),
            cagrad=dict(c=0.4),
            dwa=dict(temp=2.0),

        )
    )
    return weight_methods_parameters