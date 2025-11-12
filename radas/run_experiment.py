import random

random.seed(1342)
seeds = [random.randint(1, 65535) for _ in range(1000)]

import ray
import radas as rd
import ray.tune as tune

from .utils import *


import seaborn as sns

import matplotlib

PLOT_AXES = ["x", "y", "hue", "style", "size", "col", "row"]

def run_trainable_with_param_space_and_meta_config(param_space, meta_config):
    # rd.init
    rd.init(
        user_name=meta_config["user_name"],
        # allow reinitialization as this can be run in a local mode, where init is called outside this function
        ignore_reinit_error=True,
    )

    # Tuner_kwargs
    Tuner_kwargs = dict(
        # wrap trainable with resources
        trainable=ray.tune.with_resources(
            meta_config["trainable"],
            resources=meta_config["resources"],
        ),
        local_storage_path=meta_config["local_storage_path"],
        experiment_name=meta_config["experiment_name"],
    )

    if not meta_config["is_restore"]:

        if meta_config["is_rm_experiment"]:
            rd.rm_experiment(
                local_storage_path=meta_config["local_storage_path"],
                experiment_name=meta_config["experiment_name"],
            )

        # if not restore, create a new tuner
        tuner = rd.Tuner(
            param_space=param_space,
            **Tuner_kwargs,
            **meta_config["tuner_init_kwargs"],
        )

    else:
        # if restore, restore the tuner
        tuner = rd.Tuner.restore(
            **Tuner_kwargs,
            **meta_config["tuner_restore_kwargs"],
        )

    # run the tuner
    tuner.fit()

    return tuner


def default_process_sns_fn(sns):
    sns.set_theme(context="talk")
    return sns



async def run_experiment(
    #
    user_name,
    trainable,
    experiment_name,
    ssh_key_path=None,
    gram_per_trial="0MiB",
    runtime_env=None,
    resources={
        "cpu": 1,
        "gpu": 0,
    },
    #
    run_with="local",
    local_storage_path="~/ray_results/",
    #
    tuner_init_kwargs=dict(),
    tuner_restore_kwargs=dict(),
    #
    num_seeds=None,
    param_space=dict(),
    #
    tuner_to_df_fn=lambda tuner: tuner.get_brief_dataframe(),
    process_df_fn=lambda df: df,
    process_df_fn_kwargs=dict(),
    process_sns_fn=default_process_sns_fn,
    plot_fn=None,
    plot_kwargs=dict(),
    rename_mapping=dict(),
    matplotlib_rcParams_update=dict(),
    process_g_fn=lambda g: g,
    #
    dos=["run", "analyze"],
):

    assert_list(dos)
    for do in dos:
        assert do in ["run", "analyze"]

    rd.init(
        user_name=user_name,
    )

    _return = {}

    if "run" in dos:

        # get is_existing
        print_info("Checking if the experiment exists...")
        if local_storage_path is not None:
            is_existing = rd.is_local_experiment_existing(
                local_storage_path=local_storage_path,
                experiment_name=experiment_name,
            )["is_existing"]

        if run_with.startswith("cluster:"):
            cluster = run_with.split(":")[1]
        else:
            cluster = None

        # get choice
        choices = [
            {
                "key": "run",
                "msg": "Run a new experiment (if there is a running experiment, it will be stopped; if there is a existing experiment in the storage, it will be removed)",
            },
        ]
        if is_existing:
            choices.append(
                {
                    "key": "restore",
                    "msg": f"Restore an existing experiment from the storage (if there is a running experiment, it will be stopped)",
                }
            )
        if run_with.startswith("cluster:"):
            choices.append(
                {
                    "key": "attach",
                    "msg": f"Attach to a running experiment",
                }
            )
            choices.append(
                {
                    "key": "stop_attach",
                    "msg": f"Stop and attach to a running experiment",
                }
            )
        choice = inquire_choices(choices=choices)

        # choice -> arguments
        if choice == "run":
            is_rm_experiment = True
            is_restore = False
            is_stop = False
            is_attach = False

            answer = inquire_choices(
                msg=(
                    "Ready to run new experiment. Kind reminder to think about if you have other cleanups you want to do that are not managed by radas (e.g., wandb or swanlab)."
                ),
                choices=[
                    {
                        "key": "proceed",
                        "msg": f"Proceed",
                    },
                    {
                        "key": "exit",
                        "msg": f"Exit",
                    },
                ],
            )
            if answer == "exit":
                raise Exception("User exited.")

        elif choice == "restore":
            is_rm_experiment = False
            is_restore = True
            is_stop = False
            is_attach = False
        elif choice == "attach":
            is_rm_experiment = False
            is_restore = False
            is_stop = False
            is_attach = True
        elif choice == "stop_attach":
            is_rm_experiment = False
            is_restore = False
            is_stop = True
            is_attach = True

        # _param_space is to set some default values
        _param_space = dict(
            experiment_name=experiment_name,
        )

        # num_seeds to apply on _param_space
        if num_seeds is not None:
            assert_int_positive(num_seeds)
            _param_space.update(
                dict(
                    seed=tune.grid_search(seeds[:num_seeds]),
                )
            )

        # param_space is to override the default values in _param_space
        assert_dict(param_space)
        _param_space.update(param_space)

        submit_job_kwargs_and_meta_config = dict(
            trainable=trainable,
            user_name=user_name,
            ssh_key_path=ssh_key_path,
            resources=resources,
            local_storage_path=local_storage_path,
            is_rm_experiment=is_rm_experiment,
            experiment_name=experiment_name,
            tuner_init_kwargs=tuner_init_kwargs,
            is_restore=is_restore,
            tuner_restore_kwargs=tuner_restore_kwargs,
        )

        if run_with == "local":

            tuner = run_trainable_with_param_space_and_meta_config(
                param_space=_param_space,
                meta_config=submit_job_kwargs_and_meta_config,
            )

        else:
            raise ValueError(f"Invalid run_with: {run_with}")

    if "analyze" in dos:

        tuner = rd.Tuner.restore(
            local_storage_path=local_storage_path,
            experiment_name=experiment_name,
            trainable=trainable,
        )

        assert_callable(tuner_to_df_fn)
        df = tuner_to_df_fn(tuner)

        assert_callable(process_df_fn)
        assert_dict(process_df_fn_kwargs)
        df = process_df_fn(df, **process_df_fn_kwargs)

        if plot_fn is not None:
            assert_callable(plot_fn)

            # process_sns_fn
            global sns
            assert_callable(process_sns_fn)
            sns = process_sns_fn(sns)

            # _plot_kwargs
            _plot_kwargs = dict()

            # facet_kws
            facet_kws = dict(
                legend_out=False,
                margin_titles=True,
            )
            if plot_fn in [sns.relplot]:
                _plot_kwargs.update(
                    dict(
                        facet_kws=facet_kws,
                    )
                )
            elif plot_fn in [sns.catplot]:
                _plot_kwargs.update(facet_kws)

            # update _plot_kwargs with plot_kwargs
            assert_dict(plot_kwargs)
            _plot_kwargs.update(plot_kwargs)

            # df_for_plot
            df_for_plot = df.copy()

            # rename_mapping
            assert_dict(rename_mapping)
            for k, v in _plot_kwargs.items():
                if k in PLOT_AXES:
                    if v in rename_mapping:
                        _plot_kwargs[k] = rename_mapping[v]
                        df_for_plot = df_for_plot.rename(columns={v: rename_mapping[v]})

            # matplotlib_rcParams_update
            assert_dict(matplotlib_rcParams_update)
            matplotlib.rcParams.update(matplotlib_rcParams_update)

            # plot
            g = plot_fn(
                data=df_for_plot,
                **_plot_kwargs,
            )

            assert_callable(process_g_fn)
            g = process_g_fn(g)

        else:
            print_info(
                "Note that `plot_fn` is None, will not plot `df`, `df` is returned though in case you want to plot it yourself."
            )

        _return.update(
            {
                "df": df,
            }
        )

    _return.update(
        {
            "tuner": tuner,
        }
    )

    return _return
