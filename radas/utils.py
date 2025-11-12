import os
import uuid
import logging
import pandas as pd

from datetime import datetime

import random
import numpy as np
import torch

import itertools
from copy import deepcopy


def getLogger(logger_name):
    logging.basicConfig(
        level=logging.WARNING,
        format="\n# %(levelname)s (%(name)s): %(message)s\n",
    )

    logger = logging.getLogger(logger_name)

    return logger


logger = getLogger(__name__)

GIT_ACTION_USER_NAME_PREFIX = "git_action_"


def get_random_git_action_user_name():
    return f"{GIT_ACTION_USER_NAME_PREFIX}{get_rand_str()}"


def assert_user_name(user_name):
    # user_list = get_user_list()
    # assert (
    #     user_name in user_list
    # ), f"`user_name={user_name}` not in {user_list}, use your real user name to avoid confusion."
    assert_type(user_name, str)


def assert_msg(x, expect_type):
    return f"{x} is not a {expect_type}, but a {type(x)}"


def assert_type(x, expect_type):
    assert isinstance(x, expect_type), assert_msg(x, expect_type)


def assert_int(x):
    assert_type(x, int)


def assert_int_non_negative(x):
    assert_int(x)
    assert x >= 0, f"{x} not >= 0"


def assert_int_positive(x):
    assert_int(x)
    assert x > 0, f"{x} not > 0"


def assert_number(x):
    assert_type(x, (int, float))


def assert_number_non_negative(x):
    assert_number(x)
    assert x >= 0, f"{x} not >= 0"


def assert_number_positive(x):
    assert_number(x)
    assert x > 0, f"{x} not > 0"


def assert_str(x):
    assert_type(x, str)


def assert_dict(x):
    assert_type(x, dict)


def assert_callable(x):
    assert callable(x), f"{x} not callable"


def assert_bool(x):
    assert_type(x, bool)


def assert_list(x):
    assert_type(x, list)




RESET = "\033[0m"

BLUE = "\033[34m"
GREEN = "\033[32m"
RED = "\033[31m"
GREY = "\033[90m"
YELLOW = "\033[33m"
BLUE_BOLD = "\033[1;34m"
GREEN_BOLD = "\033[1;32m"
RED_BOLD = "\033[1;31m"
GREY_BOLD = "\033[1;90m"
YELLOW_BOLD = "\033[1;33m"


def string_with_color(s, color):
    return f"{color}{s}{RESET}"


def string_blue(s):
    return string_with_color(s, BLUE)


def string_blue_bold(s):
    return string_with_color(s, BLUE_BOLD)


def string_green(s):
    return string_with_color(s, GREEN)


def string_green_bold(s):
    return string_with_color(s, GREEN_BOLD)


def string_red(s):
    return string_with_color(s, RED)


def string_red_bold(s):
    return string_with_color(s, RED_BOLD)


def string_grey(s):
    return string_with_color(s, GREY)


def string_grey_bold(s):
    return string_with_color(s, GREY_BOLD)


def string_yellow(s):
    return string_with_color(s, YELLOW)


def string_yellow_bold(s):
    return string_with_color(s, YELLOW_BOLD)


def print_with_color(s, color):
    print(string_with_color(s, color))


def print_blue(s):
    print_with_color(s, BLUE)


def print_blue_bold(s):
    print_with_color(s, BLUE_BOLD)


def print_green(s):
    print_with_color(s, GREEN)


def print_green_bold(s):
    print_with_color(s, GREEN_BOLD)


def print_red(s):
    print_with_color(s, RED)


def print_red_bold(s):
    print_with_color(s, RED_BOLD)


def print_grey(s):
    print_with_color(s, GREY)


def print_grey_bold(s):
    print_with_color(s, GREY_BOLD)


def print_yellow(s):
    print_with_color(s, YELLOW)


def print_yellow_bold(s):
    print_with_color(s, YELLOW_BOLD)


def print_info(msg):
    print(string_green_bold("[INFO]: ") + msg)


def print_warning(msg):
    print(string_yellow_bold("[WARNING]: ") + msg)


def print_error(msg):
    print(string_red_bold("[ERROR]: ") + msg)


def print_divider(divider="=", print_fn=print):
    term_size = os.get_terminal_size()
    print_fn(divider * term_size.columns)


def inquire_choices(
    choices,
    default_choice: int = 0,
    msg: str = "Choose an option by number:",
):
    """Prompt the user to pick from a list of choices.

    Args:
        choices (list[dict]): Each dict must have the keys ``key`` and ``msg``.
        default_choice (int, optional): 0‑based index of default. Defaults to 0.
        msg (str, optional): Prompt message. Defaults to "Choose an option by number:".

    Returns:
        Any: ``key`` of selected choice.
    """

    # --- Validation helpers -------------------------------------------------
    def assert_list(value):
        if not isinstance(value, list):
            raise TypeError("choices must be a list")

    def assert_dict(value):
        if not isinstance(value, dict):
            raise TypeError("each choice must be a dict")

    def assert_str(value):
        if not isinstance(value, str):
            raise TypeError("value must be str")

    def assert_int_positive(value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("expected positive int")

    # -----------------------------------------------------------------------
    assert_list(choices)
    assert len(choices) > 0, choices

    final_msg = string_blue_bold("[Choices]: ") + msg + "\n"

    for idx, choice in enumerate(choices, start=1):
        assert_dict(choice)
        assert_str(choice["key"])
        assert_str(choice["msg"])
        choice["idx"] = idx

        if idx == default_choice + 1:
            idx_label = string_blue_bold(f"[{idx}]")
            key_label = string_blue_bold(f"{choice['key']} --")
            msg_label = string_blue_bold(f"{choice['msg']}")
        else:
            idx_label = string_grey(f" {idx} ")
            key_label = string_grey(f"{choice['key']} --")
            msg_label = string_grey(f"{choice['msg']}")

        final_msg += f"{idx_label}: {key_label} {msg_label}\n"

    # Remove trailing newline
    final_msg = final_msg.rstrip("\n")

    print(final_msg)

    # Accept input only if multiple choices
    user_input = input() if len(choices) > 1 else ""
    if user_input.strip() == "":
        user_input = str(default_choice + 1)

    try:
        selection = int(user_input)
    except ValueError:
        raise ValueError(f"Invalid input: {user_input}. Please input a number.")

    assert_int_positive(selection)
    if selection > len(choices):
        raise ValueError("Selection out of range.")

    choice = choices[selection - 1]

    print(string_blue_bold(f">>> [{choice['idx']}]: {choice['msg']}"))

    return choice["key"]


def map_list_by_dict(l, d):
    """
    Maps the values of a list using a dictionary. If a value in the list is a key
    in the dictionary, it is replaced by the corresponding value in the dictionary.

    Parameters
    ----------
    l: list
        The list to map
    d: dict
        The dictionary to use for mapping

    Returns
    -------
    list
        The mapped list
    """
    for k, v in d.items():
        if k in l:
            l[l.index(k)] = v
    return l


def assert_dfs_equal(
    df1,
    df2,
    is_ignore_row_order=False,
    is_ignore_col_order=False,
):
    assert_type(df1, pd.DataFrame)
    assert_type(df2, pd.DataFrame)

    assert sorted(list(df1.columns)) == sorted(
        list(df2.columns)
    ), f"{list(df1.columns)} != {list(df2.columns)}"

    assert_type(is_ignore_row_order, bool)
    assert_type(is_ignore_col_order, bool)

    # Convert all columns to 'object' dtype
    # because pandas often uses the 'object' dtype for columns that hold different types of data, including lists or other objects.
    # Even the whole series contains only int, pandas may not automatically change the datatype of the series.
    # It can still be an 'object' dtype series holding integer values.
    df1 = df1.astype("object")
    df2 = df2.astype("object")

    if is_ignore_row_order:
        df1 = df1.sort_values(by=sorted(list(df1.columns))).reset_index(drop=True)
        df2 = df2.sort_values(by=sorted(list(df2.columns))).reset_index(drop=True)

    if is_ignore_col_order:
        df1 = df1.reindex(sorted(df1.columns), axis=1)
        df2 = df2.reindex(sorted(df2.columns), axis=1)

    assert df1.equals(df2), f"{df1} != {df2}"


def assert_dfs_equal_no_order(df1, df2):
    return assert_dfs_equal(
        df1, df2, is_ignore_row_order=True, is_ignore_col_order=True
    )


def float_list_to_sci_str(obj, precision):
    """Ray cannot log lists, so we convert them to strings.
    Precision is the number of decimal places to use, this can be useful to save RAM.
    """
    if isinstance(obj, list):
        return "[" + ", ".join(float_list_to_sci_str(x, precision) for x in obj) + "]"
    else:
        return "{:.{}e}".format(obj, precision)


from pathlib import Path


def split_path(path):
    return [str(part) for part in Path(path).parts]


def get_leaf_values(d):
    assert_type(d, dict)

    leaf_values = []

    for key, value in d.items():
        if isinstance(value, dict):
            leaf_values.extend(get_leaf_values(value))
        else:
            leaf_values.append(value)

    return leaf_values


def get_rand_str():
    return str(uuid.uuid4())


def get_tmp_dir():
    return os.path.join("/tmp", get_rand_str())


def str_eval(x):
    if isinstance(x, str):
        return eval(x)
    return x


def get_now_time():
    """Get now time."""
    return datetime.now()


def get_now_time_stamp():
    """Get now time stamp, that can be used as a submission_id."""
    return time_to_time_stamp(get_now_time())


def assert_time(time):
    """Assert time is a datetime."""
    assert_type(time, datetime)


def assert_time_stamp(time_stamp):
    """Assert time_stamp is a string."""
    assert_type(time_stamp, str)


def time_to_time_stamp(time):
    """Convert time to time_stamp, that can be used as a submission_id."""
    # replace colons, hyphens, and spaces with underscores, as they are not allowed in experiment_name
    assert_time(time)
    return time.strftime("%Y_%m_%d_%H_%M_%S_%f")


def time_stamp_to_time(time_stamp):
    """Convert time_stamp to time."""
    assert_time_stamp(time_stamp)
    return datetime.strptime(time_stamp, "%Y_%m_%d_%H_%M_%S_%f")


def assert_times(times):
    """Assert times is a list of datetimes."""
    assert_list(times)
    for time in times:
        assert_time(time)


def assert_time_stamps(time_stamps):
    """Assert time_stamps is a list of strings."""
    assert_list(time_stamps)
    for time_stamp in time_stamps:
        assert_time_stamp(time_stamp)


def get_recent_time(times):
    """Get the most recent time from a list of times."""
    assert_times(times)
    return max(times)


def get_recent_time_stamp(time_stamps):
    """Get the most recent time stamp from a list of time stamps."""
    assert_time_stamps(time_stamps)
    times = [time_stamp_to_time(time_stamp) for time_stamp in time_stamps]
    return time_to_time_stamp(get_recent_time(times))


def reorder_list_with_scores(x, scores):
    r"""
    Reorder the list x according to the scores provided. (Not in-place)

    Args:
        x: List of elements to be reordered.
        scores: List of scores based on which the reordering should be done.

    Returns:
        List: The reordered list.

    Examples::
        >>> reorder_list_with_scores(['a', 'b', 'c'], [3, 1, 2])
        ['b', 'c', 'a']
    """

    assert_list(x)
    x = x.copy()
    assert_list(scores)
    for score in scores:
        assert_number(score)
    assert len(x) == len(scores)

    # pair each element with its score, sort by score, then extract the elements in the new order
    paired = list(zip(x, scores))
    sorted_pairs = sorted(paired, key=lambda pair: pair[1])
    reordered = [element for element, score in sorted_pairs]

    return reordered


from ray import tune


def i_to_idx_key(i):
    return f"idx_{i}"


def get_scores_space(num_scores):
    r"""Get a scores_space.

    Examples::
        >>> get_scores_space(3)
        {'score_0': tune.uniform(0, 1), 'score_1': tune.uniform(0, 1), 'score_2': tune.uniform(0, 1)}
    """
    assert_int_positive(num_scores)
    scores_space = {}
    for i in range(num_scores):
        scores_space[i_to_idx_key(i)] = tune.uniform(0, 1)
    return scores_space


def get_idxes_space(num_idxes, idx_low, idx_high):
    r"""Get a idxes_space.

    Args:
        num_idxes: Number of idxes.
        idx_low: Lower bound of each idx (inclusive).
        idx_high: Upper bound of each idx (exclusive).

    Examples::
        >>> get_idxes_space(3)
        {'idx_0': tune.uniform(0, 1), 'idx_1': tune.uniform(0, 1), 'idx_2': tune.uniform(0, 1)}
    """
    assert_int_positive(num_idxes)
    idxes_space = {}
    for i in range(num_idxes):
        idxes_space[i_to_idx_key(i)] = tune.randint(idx_low, idx_high)
    return idxes_space


def idxes_space_to_idxes(idxes_space):
    r"""Convert idxes_space to idxes.

    Examples::
        >>> idxes_space_to_idxes({'idx_0': 4, 'idx_1': 1, 'idx_2': 2})
        [4, 1, 2]
    """
    assert_dict(idxes_space)
    idxes = []
    for i, kv in enumerate(idxes_space.items()):
        k, v = kv
        assert k == i_to_idx_key(i)
        assert_number(v)
        idxes.append(v)
    return idxes


def reduce_by_config_cols(df, metric_col, reduce_fn, config_cols):
    r"""Reduce by config_cols.

    Args:
        df: DataFrame.
        metric_col: Metric column.
        reduce_fn: Reduce function.
        config_cols: Config columns.

    Examples::
        >>> df = pd.DataFrame({
        ...     'config/a': [1, 1, 2, 2],
        ...     'config/b': [1, 2, 1, 2],
        ...     'metric': [1, 2, 3, 4],
        ... })
        >>> reduce_by_config_cols(df, 'metric', lambda series: series.sum(), ['config/a', 'config/b'])
           config/a  config/b  metric
        0         1         1       1
        1         1         2       2
        2         2         1       3
        3         2         2       4
        >>> reduce_by_config_cols(df, 'metric', lambda series: series.sum(), ['config/a'])
           config/a  metric
        0         1       3
        1         2       7
    """
    assert_type(df, pd.DataFrame)
    df = df.copy()
    assert_str(metric_col)
    assert not metric_col.startswith("config/")
    assert callable(reduce_fn)
    assert_list(config_cols)
    for config_col in config_cols:
        assert_str(config_col)
        assert config_col.startswith("config/")

    # sort by config_cols
    df = df.sort_values(config_cols)
    # group by config_cols
    df = df.groupby(config_cols)
    # apply reduce_fn
    df = df.apply(lambda df: pd.Series({metric_col: reduce_fn(df[metric_col])}))
    # reset MultiIndex to (normal) Index
    df = df.reset_index()

    return df


def reduce_by_reduced_col(df, metric_col, reduce_fn, reduced_col):
    r"""Reduce by reduced_col.

    Args:
        df: DataFrame.
        metric_col: Metric column.
        reduce_fn: Reduce function.
        reduced_col: Reduced column.

    Examples::
        >>> df = pd.DataFrame({
        ...     'config/a': [1, 1, 2, 2],
        ...     'config/b': [1, 2, 1, 2],
        ...     'metric': [1, 2, 3, 4],
        ... })
        >>> reduce_by_reduced_col(df, 'metric', lambda series: series.sum(), 'config/b')
           config/a  metric
        0         1       3
        1         2       7
    """

    assert_str(reduced_col)
    assert reduced_col.startswith("config/")

    # identify config_cols
    config_cols = [col for col in df.columns if col.startswith("config/")]
    config_cols.remove(reduced_col)

    return reduce_by_config_cols(
        df=df,
        metric_col=metric_col,
        reduce_fn=reduce_fn,
        config_cols=config_cols,
    )


def apply_per_group(df, group_by_cols, apply_fn):
    assert_type(df, pd.DataFrame)
    assert_list(group_by_cols)
    for group_by_col in group_by_cols:
        assert_str(group_by_col)
    assert_callable(apply_fn)

    grouped_df = df.groupby(group_by_cols)

    res_dfs = []

    for _, each_df in grouped_df:
        each_df = apply_fn(each_df)
        res_dfs.append(each_df)

    res_df = pd.concat(res_dfs).reset_index(drop=True)

    return res_df


def set_seed(seed):
    r"""Set seed for reproducibility

    Examples::

        >>> set_seed(322)
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def config_seed(config):
    assert_dict(config)
    seed = config.get("seed", None)
    if seed is not None:
        set_seed(seed)


def assert_positive_number(x):
    assert_number(x)
    assert x > 0, f"{x} is not a positive number"


def assert_non_negative(x):
    assert_number(x)
    assert x >= 0, f"{x} is not a non-negative number"


def assert_positive_int(x):
    assert_int(x)
    assert_positive_number(x)


def assert_power_of_two(n):
    # A number n is a power of two if n is greater than 0 and
    # there's only one bit set in its binary representation.
    # This is checked by the expression (n & (n - 1)) == 0.
    assert n > 0 and (n & (n - 1)) == 0, "Number is not a power of two"


def assert_dividable(dividend, divisor):
    assert_int(dividend)
    assert_positive_int(divisor)
    assert dividend % divisor == 0, f"{dividend} is not divisible by {divisor}"


def get_item_from_nested_dict(d, keys):
    r"""Get item from nested dict by keys

    Examples::

        >>> d = {'a': {'b': {'c': 1}}}
        >>> get_item_from_nested_dict(d, keys=['a', 'b', 'c'])
        1
    """
    item = d.copy()
    for key in keys:
        item = item[key]
    return item


def merge_dicts(dict1, dict2):
    """
    Recursively merge dict2 into a new dictionary based on dict1. Deep copies are used to ensure
    that the original dictionaries are not modified.
    """
    merged = deepcopy(dict1)  # Create a deep copy of dict1
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def generate_params_combinations(params, num_samples=None, is_reduced=False):
    r"""
    Generates all combinations of parameter values for pytest.

    Args:
        params: a list of tuples, each tuple is a pair of (param_name, param_values)
        num_samples: number of samples to randomly select from all combinations.
            If None, all combinations are returned.
            It is only used when is_reduced is False.
            This is useful when the number of all combinations is too large, but using is_reduced gives too few combinations to cover sufficient number of cases.
        is_reduced: whether to produce a reduced set of params,
            which take the first element of all params as a base param item,
            and then replace one parameter at a time with the rest of the values.

    Returns:
        A list of dictionaries, each dictionary is a combination of parameter values.

    Examples::

        >>> params = [('size', [10, 20]), ('color', ['red', 'blue'])]
        >>> @pytest.mark.parametrize("params", params)
            def test_someting(params):
                assert params['size'] + params['color'] == 'something'
    """

    # assert params
    assert isinstance(params, list), f"params must be a list, but got {type(params)}"
    for param in params:
        assert isinstance(param, tuple), f"param must be a tuple, but got {type(param)}"
        assert (
            len(param) == 2
        ), f"param must be a tuple of length 2, but got {len(param)}"
        assert isinstance(
            param[0], str
        ), f"param[0] must be a string, but got {type(param[0])}"
        assert isinstance(
            param[1], list
        ), f"param[1] must be a list, but got {type(param[1])}"

    # assert is_reduced
    assert_type(is_reduced, bool)

    if not is_reduced:
        keys = [p[0] for p in params]
        vals = [p[1] for p in params]

        if num_samples is None:
            # generate all combinations
            params_combinations = [dict(zip(keys, p)) for p in itertools.product(*vals)]

        else:
            # get num_all_combinations
            num_all_combinations = 1
            for val in vals:
                num_all_combinations *= len(val)

            # assert num_samples
            assert_int(num_samples)
            assert_positive_number(num_samples)
            assert num_samples <= num_all_combinations, (
                f"num_samples must be less than or equal to the number of all combinations, "
                f"but got num_samples = {num_samples} and number of all combinations = {num_all_combinations}."
            )

            # randomly sample num_samples combinations by randomly sampling values for each param
            params_combinations = []
            for _ in range(num_samples):
                params = dict()
                for key, val in zip(keys, vals):
                    params[key] = random.choice(val)
                params_combinations.append(params)

    else:
        assert num_samples is None, (
            "num_samples must be None when is_reduced is True, "
            f"but got num_samples = {num_samples}. "
            "I.e., num_samples is only used when is_reduced is False. "
        )

        params_combinations = []

        # base_param_item takes the first element of all params
        base_param_item = dict()
        for param in params:
            param_key, parame_values = param
            base_param_item[param_key] = parame_values[0]

        params_combinations.append(base_param_item)

        # and then replace one parameter at a time with the rest of the values.
        for param in params:
            param_key, parame_values = param
            for parame_value in parame_values[1:]:
                param_item = deepcopy(base_param_item)
                param_item[param_key] = parame_value
                params_combinations.append(param_item)

    return params_combinations


import re


def sanitize_folder_name(input_string):
    # Replace all characters except letters, numbers, and underscores with underscores
    return re.sub(r"[^a-zA-Z0-9_]", "_", input_string)


