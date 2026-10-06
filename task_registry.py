TASK_TO_COLNAME = {
    "hiv_b": "hiv_b",
    "bace_b": "bace_b",
    "bbbp_b": "bbbp_b",
    "O2": "O2",
    "N2": "N2",
    "CO2": "CO2",
    "Egb": "Egb",
    "Eea": "Eea",
    "Egc": "Egc",
    "Ei": "Ei",
    "freesolv": "freesolv",
    "esol": "esol",
    "mu": "mu",
    "alpha": "alpha",
    "homo": "homo",
    "lumo": "lumo",
    "cv": "cv",
}

TASK_TYPE_BY_NAME = {
    "hiv_b": "classification",
    "bace_b": "classification",
    "bbbp_b": "classification",
    "O2": "regression",
    "N2": "regression",
    "CO2": "regression",
    "Egb": "regression",
    "Eea": "regression",
    "Egc": "regression",
    "Ei": "regression",
    "freesolv": "regression",
    "esol": "regression",
    "mu": "regression",
    "alpha": "regression",
    "homo": "regression",
    "lumo": "regression",
    "cv": "regression",
}


def get_task_type(task_name):
    return TASK_TYPE_BY_NAME[str(task_name).strip()]


def is_regression_task(task_name):
    return get_task_type(task_name) == "regression"


def is_classification_task(task_name):
    return get_task_type(task_name) == "classification"
