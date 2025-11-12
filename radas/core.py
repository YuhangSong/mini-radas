import os
import json
import numbers
import tqdm
import torch
import itertools
import numpy as np
import pandas as pd

from .utils import *

logger = getLogger(__name__)


def load_config_from_params(logdir):
    """Load config from params.json in a logdir.

    Args:
        logdir (str): The logdir, in which we load the config from params.json.

    Returns:
        (Dict) The config.

    """
    path = os.path.join(logdir, "params.json")
    with open(path, "rt") as f:
        config = json.load(f)
    return config


def resolve_nested_dict(nested_dict):
    """Flattens a nested dict by joining keys into tuple of paths."""
    assert_type(nested_dict, dict), f"Expected dict, got {type(nested_dict)}"
    res = {}
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            for k_, v_ in resolve_nested_dict(v).items():
                res[(k,) + k_] = v_
        else:
            res[(k,)] = v
    return res


def unresolve_nested_dict(flat_dict):
    """Unflattens a flat dict by splitting tuple of paths into nested dict.
    Reverses the process of `resolve_nested_dict`.
    """
    assert_type(flat_dict, dict), f"Expected dict, got {type(flat_dict)}"
    res = {}
    for keys, value in flat_dict.items():
        d = res
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    return res




def tuple_config_k_to_str(tuple_config_k):
    assert_type(tuple_config_k, tuple)
    list_config_k = list(tuple_config_k)
    for config_k_item in list_config_k:
        assert_type(config_k_item, str)
    return ": ".join(list_config_k)




def can_eval_as_list(s):
    try:
        s = eval(s)
    except Exception:
        return False

    return isinstance(s, list)


def explode(df, columns, new_idx_col=None):
    assert_type(df, pd.DataFrame)
    assert_type(columns, list)

    for col in columns:

        def apply_fn(cell):
            if isinstance(cell, list):
                return cell

            elif isinstance(cell, str):
                if can_eval_as_list(cell):
                    return eval(cell)
                else:
                    raise ValueError(f"cell {cell} cannot be eval as list")

            elif hasattr(cell, "tolist"):
                return cell.tolist()

            else:
                raise ValueError(
                    "cell can be a list, "
                    "or a str but need to be able to be eval as a list, "
                    "or has a tolist() method. "
                    "but get {} of type {}".format(cell, type(cell))
                )

        df.loc[:, col] = df[col].apply(apply_fn)

    if new_idx_col is not None:
        df[new_idx_col] = df[columns[0]].apply(lambda x: list(range(len(x))))
        columns_to_explode = columns + [new_idx_col]
    else:
        columns_to_explode = columns

    df = df.explode(columns_to_explode)

    return df


def explode_tensor(tensor, cols, value_col="value"):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    tensor = tensor.detach().cpu()

    assert_type(tensor, torch.Tensor), f"unsupported {type(tensor)}"
    assert_type(cols, list), f"unsupported {type(cols)}"
    assert tensor.dim() == len(cols), f"{tensor.dim()} != {len(cols)}"

    flat_tensor = tensor.flatten()
    index_combinations = list(itertools.product(*[range(dim) for dim in tensor.shape]))
    df = pd.DataFrame(index_combinations, columns=cols)
    df[value_col] = flat_tensor

    return df


def explode_dict_of_tensors(dict_of_tensors, cols, **kwargs):
    assert_type(dict_of_tensors, dict), f"unsupported {type(dict_of_tensors)}"
    assert_type(cols, list), f"unsupported {type(cols)}"

    key_col = cols[0]
    assert len(cols) > 1, f"unsupported {len(cols)}"
    tensor_cols = cols[1:]

    dfs_ = []
    for key, tensor in dict_of_tensors.items():
        df_ = explode_tensor(
            tensor=tensor,
            cols=tensor_cols,
            **kwargs,
        )
        df_[key_col] = key
        dfs_.append(df_)

    df = pd.concat(dfs_, ignore_index=True)

    return df




def dataframe_to_brief_dataframe(df):
    assert_type(df, pd.DataFrame)
    return df.drop(
        columns=[
            "timestamp",
            "done",
            "trial_id",
            "date",
            "time_this_iter_s",
            "time_total_s",
            "pid",
            "hostname",
            "node_ip",
            "time_since_restore",
            "iterations_since_restore",
            "checkpoint_dir_name",
            "logdir",
        ],
        errors="ignore",
    )


context = {}


def _uninitialize_context():
    context["user_name"] = None


_uninitialize_context()


def _is_context_initialized():
    return context["user_name"] is not None


def _assert_context_uninitialized():
    assert (
        not _is_context_initialized()
    ), "Context already initialized. Please call `shutdown` first. "


def _assert_context_initialized():
    assert (
        _is_context_initialized()
    ), "Context not initialized. Please call `init` first. "


def _initialize_context(user_name):
    # add user_name to context
    assert_user_name(user_name)
    context["user_name"] = user_name


import ray


def init(user_name, ignore_reinit_error=True):
    assert_user_name(user_name)

    assert_bool(ignore_reinit_error)
    if not ignore_reinit_error:
        _assert_context_uninitialized()
    _initialize_context(user_name=user_name)


def shutdown():
    _assert_context_initialized()
    _uninitialize_context()


from ray import tune
from ray.tune import RunConfig


def get_storage_path(
    local_storage_path,
):
    _assert_context_initialized()

    user_name = context["user_name"]

    storage_path = None
    if local_storage_path is not None:
        assert_str(local_storage_path)

        storage_path = f"{local_storage_path}/{user_name}/"

    return storage_path


class TrialStartConfigCheckCallback(tune.Callback):
    r"""Check config is legal at the start of each trial."""

    def on_trial_start(self, iteration, trials, trial, **info):
        for config_item in get_leaf_values(trial.config):
            if config_item is not None:
                assert_type(
                    config_item, (numbers.Number, str, list, tuple)
                ), f"unsupported config_item: `{config_item}` of type `{type(config_item)}`"


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# this method is copied from ray's code, because we need the Tuner in the following # # #
# code to be our Tuner, not ray's Tuner # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from ray.tune.tuner import (
    Union,
    Callable,
    Type,
    Trainable,
    Optional,
    Dict,
    Any,
    pyarrow,
    ResumeConfig,
    TunerInternal,
    _force_on_current_node,
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



class Tuner(tune.Tuner):
    def __init__(
        self,
        local_storage_path=None,
        experiment_name=None,
        **kwargs,
    ):
        _assert_context_initialized()

        # not allow to specify run_config, but use run_config_kwargs instead
        assert "run_config" not in kwargs, "please use run_config_kwargs instead"

        # infer storage_path
        storage_path = get_storage_path(
            local_storage_path=local_storage_path,
        )

        # if storage_path is specified, experiment_name should be specified
        if storage_path is not None:
            assert_str(storage_path)
            assert (
                experiment_name is not None
            ), f"storage_path is specified to be {storage_path}, so experiment_name should be specified"

        # get callbacks
        callbacks = kwargs.pop("run_config_kwargs", {}).pop("callbacks", [])
        # add callbacks
        callbacks += [
            TrialStartConfigCheckCallback(),
        ]

        # create run_config
        run_config = RunConfig(
            storage_path=storage_path,
            name=experiment_name,
            callbacks=callbacks,
            failure_config=tune.FailureConfig(
                max_failures=-1,
                fail_fast=False,
            ),
            **kwargs.pop("run_config_kwargs", {}),
        )

        # record experiment_name, this is for identifying historical dataframe
        self.experiment_name = experiment_name

        super().__init__(
            run_config=run_config,
            **kwargs,
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # this method is copied from ray's code, because we need the Tuner in the following # # #
    # code to be our Tuner, not ray's Tuner # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @classmethod
    def ray_restore(
        cls,
        path: str,
        trainable: Union[str, Callable, Type[Trainable], "BaseTrainer"],
        resume_unfinished: bool = True,
        resume_errored: bool = False,
        restart_errored: bool = False,
        param_space: Optional[Dict[str, Any]] = None,
        storage_filesystem: Optional[pyarrow.fs.FileSystem] = None,
        _resume_config: Optional[ResumeConfig] = None,
    ) -> "Tuner":
        """Restores Tuner after a previously failed run.

        All trials from the existing run will be added to the result table. The
        argument flags control how existing but unfinished or errored trials are
        resumed.

        Finished trials are always added to the overview table. They will not be
        resumed.

        Unfinished trials can be controlled with the ``resume_unfinished`` flag.
        If ``True`` (default), they will be continued. If ``False``, they will
        be added as terminated trials (even if they were only created and never
        trained).

        Errored trials can be controlled with the ``resume_errored`` and
        ``restart_errored`` flags. The former will resume errored trials from
        their latest checkpoints. The latter will restart errored trials from
        scratch and prevent loading their last checkpoints.

        .. note::

            Restoring an experiment from a path that's pointing to a *different*
            location than the original experiment path is supported.
            However, Ray Tune assumes that the full experiment directory is available
            (including checkpoints) so that it's possible to resume trials from their
            latest state.

            For example, if the original experiment path was run locally,
            then the results are uploaded to cloud storage, Ray Tune expects the full
            contents to be available in cloud storage if attempting to resume
            via ``Tuner.restore("s3://...")``. The restored run will continue
            writing results to the same cloud storage location.

        Args:
            path: The local or remote path of the experiment directory
                for an interrupted or failed run.
                Note that an experiment where all trials finished will not be resumed.
                This information could be easily located near the end of the
                console output of previous run.
            trainable: The trainable to use upon resuming the experiment.
                This should be the same trainable that was used to initialize
                the original Tuner.
            param_space: The same `param_space` that was passed to
                the original Tuner. This can be optionally re-specified due
                to the `param_space` potentially containing Ray object
                references (tuning over Datasets or tuning over
                several `ray.put` object references). **Tune expects the
                `param_space` to be unmodified**, and the only part that
                will be used during restore are the updated object references.
                Changing the hyperparameter search space then resuming is NOT
                supported by this API.
            resume_unfinished: If True, will continue to run unfinished trials.
            resume_errored: If True, will re-schedule errored trials and try to
                restore from their latest checkpoints.
            restart_errored: If True, will re-schedule errored trials but force
                restarting them from scratch (no checkpoint will be loaded).
            storage_filesystem: Custom ``pyarrow.fs.FileSystem``
                corresponding to the ``path``. This may be necessary if the original
                experiment passed in a custom filesystem.
            _resume_config: [Experimental] Config object that controls how to resume
                trials of different statuses. Can be used as a substitute to
                `resume_*` and `restart_*` flags above.
        """
        unfinished = (
            ResumeConfig.ResumeType.RESUME
            if resume_unfinished
            else ResumeConfig.ResumeType.SKIP
        )
        errored = ResumeConfig.ResumeType.SKIP
        if resume_errored:
            errored = ResumeConfig.ResumeType.RESUME
        elif restart_errored:
            errored = ResumeConfig.ResumeType.RESTART

        resume_config = _resume_config or ResumeConfig(
            unfinished=unfinished, errored=errored
        )

        if not ray.util.client.ray.is_connected():
            tuner_internal = TunerInternal(
                restore_path=path,
                resume_config=resume_config,
                trainable=trainable,
                param_space=param_space,
                storage_filesystem=storage_filesystem,
            )
            return Tuner(_tuner_internal=tuner_internal)
        else:
            tuner_internal = _force_on_current_node(
                ray.remote(num_cpus=0)(TunerInternal)
            ).remote(
                restore_path=path,
                resume_config=resume_config,
                trainable=trainable,
                param_space=param_space,
                storage_filesystem=storage_filesystem,
            )
            return Tuner(_tuner_internal=tuner_internal)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @classmethod
    def restore(
        cls,
        experiment_name,
        # storage_path
        local_storage_path=None,
        # override default restore behavior
        resume_unfinished=True,
        resume_errored=False,
        restart_errored=True,
        **kwargs,
    ) -> "Tuner":
        assert (
            "path" not in kwargs
        ), "please use `local_storage_path`, and `experiment_name` instead"

        storage_path = get_storage_path(
            local_storage_path=local_storage_path,
        )

        assert storage_path is not None
        assert (
            experiment_name is not None
        ), "restore tuner requires experiment_name to be specified"

        restored_tuner = cls.ray_restore(
            path=os.path.join(storage_path, experiment_name),
            resume_unfinished=resume_unfinished,
            resume_errored=resume_errored,
            restart_errored=restart_errored,
            **kwargs,
        )

        # record experiment_name, this is for identifying historical dataframe
        restored_tuner.experiment_name = experiment_name

        return restored_tuner

    def get_dataframe(self):
        """Shortcut to dataframe"""
        return self.get_results().get_dataframe()

    def get_brief_dataframe(self):
        """Shortcut to brief_dataframe"""
        return dataframe_to_brief_dataframe(self.get_dataframe())

    def get_progress_dataframe(self):
        results = self.get_results()
        df = []
        for result in results:
            df.append(result.metrics_dataframe)
        return pd.concat(df, ignore_index=True)

    def get_brief_progress_dataframe(self):
        return dataframe_to_brief_dataframe(self.get_progress_dataframe())


from radas.utils import *



def ls_local_folder(local_storage_path, folder_path):
    if folder_path is None:
        folder_path = ""
    full_path = os.path.join(local_storage_path, folder_path)

    if not os.path.exists(full_path):
        return []

    folder_items = []
    for item in os.listdir(full_path):
        item_path = os.path.join(full_path, item)
        if os.path.isfile(item_path) or os.path.isdir(item_path):
            folder_items.append(item)

    return folder_items




def rm_local_folder(local_storage_path, folder_path):
    if folder_path is None:
        folder_path = ""
    full_path = os.path.join(local_storage_path, folder_path)

    # If path doesn't exist or isn't a directory, mimic cloud behavior: do nothing
    if not os.path.exists(full_path) or not os.path.isdir(full_path):
        return

    # Collect all files and directories to delete
    items_to_delete = []
    for root, dirs, files in os.walk(full_path, topdown=False):
        for name in files:
            items_to_delete.append(os.path.join(root, name))
        for name in dirs:
            items_to_delete.append(os.path.join(root, name))

    iterator = tqdm.tqdm(
        items_to_delete, total=len(items_to_delete), unit="file", leave=False
    )

    for path in iterator:
        iterator.set_description(
            f"rm_local_folder: {os.path.relpath(path, local_storage_path)}"
        )
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                os.rmdir(path)
        except Exception as e:
            print(f"Failed to delete {path}: {e}")

    # Finally, remove the folder itself
    try:
        os.rmdir(full_path)
    except Exception as e:
        print(f"Failed to delete root folder {full_path}: {e}")



def ls_local_users(local_storage_path):
    return ls_local_folder(
        local_storage_path=local_storage_path,
        folder_path=None,
    )




def ls_local_experiments(local_storage_path):
    _assert_context_initialized()
    user_name = context["user_name"]
    return ls_local_folder(
        local_storage_path=local_storage_path,
        folder_path=user_name,
    )




def is_local_experiment_existing(local_storage_path, experiment_name):
    existing_local_experiments = ls_local_experiments(
        local_storage_path=local_storage_path,
    )
    return {
        "is_existing": experiment_name in existing_local_experiments,
        "existing_local_experiments": existing_local_experiments,
    }


def rm_local_experiment(
    local_storage_path,
    experiment_name,
    is_ignore_not_exist=False,
):
    _assert_context_initialized()
    user_name = context["user_name"]
    res = is_local_experiment_existing(
        local_storage_path=local_storage_path,
        experiment_name=experiment_name,
    )
    if not is_ignore_not_exist:
        assert res[
            "is_existing"
        ], f"{experiment_name} not found among {res['existing_local_experiments']}"
    rm_local_folder(
        local_storage_path=local_storage_path,
        folder_path=os.path.join(user_name, experiment_name),
    )


def rm_experiment(local_storage_path, experiment_name):
    if local_storage_path is not None:
        rm_local_experiment(
            local_storage_path=local_storage_path,
            experiment_name=experiment_name,
            is_ignore_not_exist=True,
        )



def rm_local_user(
    local_storage_path,
    is_ignore_not_exist=False,
):
    _assert_context_initialized()
    user_name = context["user_name"]
    existing_local_users = ls_local_users(local_storage_path=local_storage_path)
    if not is_ignore_not_exist:
        assert (
            user_name in existing_local_users
        ), f"{user_name} not found among {existing_local_users}"
    rm_local_folder(
        local_storage_path=local_storage_path,
        folder_path=user_name,
    )

