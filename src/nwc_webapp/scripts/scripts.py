import os

import numpy as np
from tqdm import tqdm

from nwc_webapp.logging_config import setup_logger

# Set up logger
logger = setup_logger(__name__)


def create_single_prediction_array(folder_path, out_path):
    npz_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".npz")]

    npz_files = sorted(npz_files)
    data = np.load(npz_files[0])
    logger.info(list(data.keys()))
    logger.debug(data["prediction"].shape)
    all_predictions = []

    all_predictions = []
    for file in tqdm(npz_files):
        data = np.load(file)
        if "prediction" in data:
            all_predictions.append(data["prediction"])
        else:
            logger.warning("Manca la chiave")
    if all_predictions:
        merged_array = np.stack(all_predictions, axis=0)
    else:
        logger.error("Nessun file")

    np.save("single_array_predictions.npy", merged_array)

    np.save(os.path.join(out_path, "predictions.npy"), merged_array)


# folder_path = "/davinci-1/work/protezionecivile/sole24/convlstmheist_pred"
# out_path = "/davinci-1/work/protezionecivile/sole24/pred_teo/ConvLSTMHeist"

folder_path = "/davinci-1/work/protezionecivile/sole24/pystep_pred/sprog_2025-02-11T16:43:24/npz"
out_path = "/davinci-1/work/protezionecivile/sole24/pred_teo/pystep"
create_single_prediction_array(folder_path, out_path)
pass
