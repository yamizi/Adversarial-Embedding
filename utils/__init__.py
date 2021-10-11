from config import COMET_APIKEY
from comet_ml import Experiment
import time

def init_comet(args, project_name="stegano-draft"):
    timestamp = time.time()
    args["timestamp"] = timestamp
    experiment_name = "{}_{}_{}_{}".format(args["algorithm"], args["bpp"], args["use_hidden"], timestamp)
    experiment = Experiment(api_key=COMET_APIKEY,
                            project_name=project_name,
                            workspace="yamizi",
                            auto_param_logging=False, auto_metric_logging=False,
                            parse_args=False, display_summary=False, disabled=False)

    experiment.set_name(experiment_name)
    experiment.log_parameters(args)

    return experiment
