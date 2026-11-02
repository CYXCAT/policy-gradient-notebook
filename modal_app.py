"""Run every (environment, algorithm, seed) experiment from policy_gradient.py in parallel on Modal.

    modal run modal_app.py::main                                # all 5 environments
    modal run modal_app.py::main --envs CartPole-v1,Acrobot-v1  # a subset

Results (.npy) and charts (.png) are written locally to results_modal/.
"""
import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("gymnasium[box2d,mujoco]>=1.1", "numpy", "matplotlib")
    .add_local_python_source("policy_gradient")
)
app = modal.App("policy-gradient", image=image)


@app.function(cpu=1.0, timeout=4 * 3600)
def remote_run(env_id: str, algo: str, seed: int):
    import policy_gradient as pg
    return pg.run_one(env_id, algo, seed)


@app.function(cpu=1.0, timeout=4 * 3600)
def remote_run_cfg(env_id: str, algo: str, seed: int, overrides: dict):
    import policy_gradient as pg
    return pg.run_one(env_id, algo, seed, overrides)


@app.local_entrypoint()
def sweep(spec: str = "sweep.json"):
    """spec: JSON list of {"env", "algo", "seeds", "cfg"} entries. Prints mean last-50 return per entry."""
    import json
    import numpy as np
    entries = json.load(open(spec))
    jobs = [(e["env"], e["algo"], s, e["cfg"]) for e in entries for s in range(e.get("seeds", 3))]
    print(f"{len(jobs)} sweep runs -> Modal")
    out = {}
    for (env_id, algo, seed, r, dt), job in zip(remote_run_cfg.starmap(jobs), jobs):
        key = (env_id, algo, json.dumps(job[3], sort_keys=True))
        out.setdefault(key, []).append((r[-50:].mean(), r[len(r) // 2:].mean()))
    for (e, a, c), v in sorted(out.items()):
        v = np.array(v)
        print(f"{e:22s} {a:24s} last50={v[:,0].mean():8.1f} 2nd-half={v[:,1].mean():8.1f} per-seed={np.round(v[:,0],0).tolist()} {c}")


@app.local_entrypoint()
def main(envs: str = ""):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import policy_gradient as pg

    pg.RESULTS_DIR = "results_modal"
    os.makedirs(pg.RESULTS_DIR, exist_ok=True)
    env_ids = envs.split(",") if envs else list(pg.ENV_CONFIGS)

    jobs = [(e, a, s) for e in env_ids for a in pg.ALGOS for s in range(pg.N_SEEDS)
            if not os.path.exists(pg.result_path(e, a))]
    print(f"{len(jobs)} runs -> Modal")
    buf = {}
    for e, a, s, r, dt in remote_run.starmap(jobs, order_outputs=False):
        print(f"  {e:20s} {a:24s} seed={s}  last-50 mean={r[-50:].mean():8.1f}  ({dt:5.0f}s)")
        buf.setdefault((e, a), {})[s] = r
        if len(buf[(e, a)]) == pg.N_SEEDS:
            np.save(pg.result_path(e, a), np.stack([buf[(e, a)][k] for k in range(pg.N_SEEDS)]))

    for e in env_ids:
        pg.plot_env(e)
    print(f"charts saved to {pg.RESULTS_DIR}/")
