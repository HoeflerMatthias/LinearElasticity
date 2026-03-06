#!/usr/bin/env python3

# %%

import multiprocessing
import json
import itertools
import copy
import os

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
    with open(setup_file, 'r') as f:
        setup = json.load(f)

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


def _gpu_worker(args):
    """Worker that sets CUDA_VISIBLE_DEVICES before importing TF."""
    gpu_id, run_module, run_func_name, arg = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import importlib
    mod = importlib.import_module(run_module)
    runfunc = getattr(mod, run_func_name)
    runfunc(arg)


def setup_trial(runfunc, setup_file, config, seeds, keylist,
                parallel=False, num_processes=1, gpus=None,
                exception_handling=False, extract_params_from_setup=False):
    """Build parameter combinations and run them.

    Parameters
    ----------
    gpus : list[int] or None
        GPU IDs to distribute runs across (e.g. [0, 1, 2]).
        Each run is assigned to the next free GPU.  Requires the
        run function to be importable by module path (uses spawn).
        When set, *parallel* and *num_processes* are ignored.
    """
    paramlist = _build_paramlist(setup_file, config, seeds, keylist,
                                 extract_params_from_setup)

    if gpus and len(gpus) > 1:
        _run_multi_gpu(runfunc, paramlist, gpus, exception_handling)
    elif not parallel:
        for arg in paramlist:
            if exception_handling:
                try:
                    runfunc(arg)
                except Exception as inst:
                    print(type(inst))
                    print(inst.args)
                    print(inst)
                    print("error for ", arg)
            else:
                runfunc(arg)
    else:
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(runfunc, paramlist)

    return paramlist


def _run_multi_gpu(runfunc, paramlist, gpus, exception_handling):
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

        for arg in paramlist:
            gpu_id = next(gpu_cycle)
            worker_args = (gpu_id, run_module, run_func_name, arg)
            future = executor.submit(_gpu_worker, worker_args)
            futures[future] = (gpu_id, arg[1])

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