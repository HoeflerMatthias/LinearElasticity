#!/usr/bin/env python3

# %%

import multiprocessing
import itertools
import copy
import os

from pinn_source.utils.file_utils import load_config


#############################################################################
# File utilities
#############################################################################

def params_to_filename(params, keylist = [], separator = ""):

    def format_entry(obj, key, path = ""):
        code = ""
        if isinstance(obj, list):
            if key == 'layers':
                key_str = 'L'
            elif key == 'phases':
                key_str = 'P'
            else:
                key_str = key
            if isinstance(obj[0], str):
                code += path + key_str + ''.join([l + '_' for l in obj])
            else:
                code += path + key_str + ''.join(['%d_' % l for l in obj])
        elif isinstance(obj, float):
            if key not in ['lr1', 'lr2']:
                code += path + key + "%1.1e" % (obj)
        elif isinstance(obj, bool):
            if obj == True:
                code += path + key
        elif isinstance(obj, int):
            code += path + key + "%d" % (obj)
        elif isinstance(obj, str):
            code += path + obj
        return code

    filename = ""
    keylist_comp = [key.split('/') for key in keylist if '/' in key]
    keylist = [key for key in keylist if '/' not in key]
    for key_comp in keylist_comp:
        obj = params[key_comp[0]]
        for key in key_comp[1:]:
            obj = obj[key]
        filename += format_entry(obj, key_comp[-1], ''.join([s[:3] for s in key_comp[:-1]]))

    for key in params:
        if key in keylist or isinstance(params[key], dict) or len(keylist) == 0:

            obj = params[key]
            if isinstance(obj, dict):
                if key == 'fourier_params' and params['fourier'] == 1:
                    filename += separator + params_to_filename(params['fourier_params'], keylist)
                elif key == 'time_params':
                    filename += separator + params_to_filename(params['time_params'], keylist)
                elif key == 'slicing_params' and params['slicing'] == 1:
                    filename += separator + params_to_filename(params['slicing_params'], keylist)
                elif key == 'bc':
                    filename += separator + params_to_filename(params['bc'], keylist)
                elif key == 'model':
                    filename += separator + params_to_filename(params['model'], keylist)
                elif key == 'inverse_params':
                    filename += separator + params_to_filename(params['inverse_params'], keylist)
            else:
                filename += separator + format_entry(obj, key)
    return filename

def replace_item(obj, keylist, replace_value):
    if isinstance(keylist, list):
        if len(keylist) > 1:
            obj[keylist[0]] = replace_item(obj[keylist[0]], keylist[1:], replace_value)
        else:
            obj[keylist[0]] = replace_value
    else:
        obj[keylist] = replace_value

    return obj

#############################################################################
# Multi processing
#############################################################################

def _build_paramlist(setup_file, config, seeds, keylist, extract_params_from_setup=False):
    setup = load_config(setup_file)

    if extract_params_from_setup:
        setup = setup['parameter']

    paramlist = []

    keys, values = zip(*config.items())
    for bundle in itertools.product(*values, seeds):

        params = copy.deepcopy(setup)
        c = dict(zip(keys, bundle[:-1]))
        params['seed'] = bundle[-1]

        for key in keys:
            key_comp = key.split('/') if '/' in key else key
            params = replace_item(params, key_comp, c[key])

        filename = params_to_filename(params, keylist)
        paramlist += [(params, filename)]

    # Make directories
    for params, _ in paramlist:
        base_dir = params['program']['base_dir']
        for key in params['program']:
            if key != 'base_dir':
                os.makedirs(os.path.join(base_dir, params['program'][key]), exist_ok=True)

    return paramlist


def _detect_gpus():
    """Return list of GPU IDs visible to TensorFlow."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return list(range(len(gpus)))
    except Exception:
        return []


def _gpu_worker(args):
    """Worker that sets CUDA_VISIBLE_DEVICES before importing TF."""
    gpu_id, run_module, run_func_name, params, keylist, experiment_name = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import importlib
    mod = importlib.import_module(run_module)
    runfunc = getattr(mod, run_func_name)
    _run_single(runfunc, params, keylist, experiment_name)


def _make_algorithm_fn(solver_run, keylist):
    """Create an algorithm_fn(params, seed) for ExperimentRunner."""
    def algorithm_fn(params, seed):
        filename = params_to_filename(params, keylist)
        return solver_run(params, filename)
    return algorithm_fn


def _make_post_run_fn():
    """Create a post_run_fn(result) that logs PINNs artifacts."""
    def post_run_fn(result):
        from pinn_source.mlflow_logging import log_pinns_artifacts
        log_pinns_artifacts(
            filename=result['filename'],
            loss_handler=result['loss_handler'],
            train_handler=result['train_handler'],
            artifact_dirs=result.get('artifact_dirs'),
            timings=result.get('timings'),
        )
    return post_run_fn


def _run_single(solver_run, params, keylist, experiment_name):
    """Run a single parameter set, with ExperimentRunner if MLflow is available."""
    seed = params['seed']

    if os.environ.get("MLFLOW_TRACKING_URI"):
        from pinn_source.experiment_runner import ExperimentRunner

        runner = ExperimentRunner(
            params=params,
            algorithm_fn=_make_algorithm_fn(solver_run, keylist),
            experiment_name=experiment_name,
            post_run_fn=_make_post_run_fn(),
        )
        runner.run(seed)
    else:
        filename = params_to_filename(params, keylist)
        solver_run(params, filename)


def setup_trial(runfunc, setup_file, config, seeds, keylist,
                exception_handling=False, extract_params_from_setup=False,
                experiment_name="lin_elast:pinns"):
    """Build parameter combinations and run them.

    Auto-detects available GPUs. With multiple GPUs and multiple runs,
    distributes work round-robin across GPUs using spawned processes.
    With a single GPU (or single run), runs sequentially.
    """
    paramlist = _build_paramlist(setup_file, config, seeds, keylist,
                                 extract_params_from_setup)

    gpus = _detect_gpus()

    if len(gpus) > 1 and len(paramlist) > 1:
        _run_multi_gpu(runfunc, paramlist, gpus, exception_handling,
                       keylist, experiment_name)
    else:
        for params, filename in paramlist:
            if exception_handling:
                try:
                    _run_single(runfunc, params, keylist, experiment_name)
                except Exception as inst:
                    print(type(inst))
                    print(inst.args)
                    print(inst)
                    print("error for ", filename)
            else:
                _run_single(runfunc, params, keylist, experiment_name)

    return paramlist


def _run_multi_gpu(runfunc, paramlist, gpus, exception_handling,
                   keylist, experiment_name):
    """Distribute runs across GPUs using a process pool with spawn context."""
    import concurrent.futures

    # Resolve module path for the run function so workers can import it
    run_module = runfunc.__module__
    run_func_name = runfunc.__qualname__

    ctx = multiprocessing.get_context('spawn')
    num_workers = len(gpus)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, mp_context=ctx
    ) as executor:
        futures = {}
        gpu_cycle = itertools.cycle(gpus)

        for params, filename in paramlist:
            gpu_id = next(gpu_cycle)
            worker_args = (gpu_id, run_module, run_func_name,
                           params, keylist, experiment_name)
            future = executor.submit(_gpu_worker, worker_args)
            futures[future] = (gpu_id, filename)

        for future in concurrent.futures.as_completed(futures):
            gpu_id, filename = futures[future]
            try:
                future.result()
                print(f"[GPU {gpu_id}] Done: {filename}")
            except Exception as exc:
                if exception_handling:
                    print(f"[GPU {gpu_id}] Failed: {filename}: {exc}")
                else:
                    raise
