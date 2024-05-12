import gurobipy as gp

options = {
    "WLSACCESSID": "f083dae9-b2fa-4266-8b61-2d4c9afe65eb",
    "WLSSECRET": "b87b0a66-c3c9-4087-bbf6-26357117e655",
    "LICENSEID": 2514088,
}
with gp.Env(params=options) as env, gp.Model(env=env) as model:
    # Formulate problem
    model.optimize()