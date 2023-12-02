from flask import session
import numpy as np
from src.main.tools.db_write import DB_manager
from src.models.calculation import Calculation
from src.main.tools.cache_check import data_in_db

from src.main.analysis.coherence import coherence_time_frame
from scipy.io import savemat


def export_connectivity_to_mat(conn_func, name, data, sfreq, start, end, meta_data):
    db_manager = DB_manager()
    file_name = session["user_data_dir"]
    url = (
        "http://localhost:5000/connectivity?connectivity="
        + meta_data["connectivity_measure"]
        + "&start="
        + str(start)
        + "&end="
        + str(end)
    )
    cal = data_in_db(file_name=file_name, url=url, table=Calculation.query)
    if cal:
        print("Cached")
        data = cal.data
        f = data["f"]
        CM = data["CM"]
    else:
        f, CM = conn_func(data, sfreq)  # the actual calculation
        db_manager.write_calculation(
            file_name=file_name,
            url=url,
            data={"f": f.tolist(), "CM": CM.tolist()},
            created_by=session["username"],
        )
    # write results to mat file
    full_name = name + ".mat"
    vars_dict = {
        "frequencies": f,
        "connectivity_matrices": CM,
        "start": start,
        "end": end,
    }
    # add the meta data to the dictionary
    vars_dict.update(meta_data)
    savemat(
        full_name,
        vars_dict,
    )
    return
