import numpy as np
import pickle

import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import seaborn as sns

def reformat_on_policy_pretrain_cache(on_policy_pretrain_cache, n_exp, n_episodes):
    for eps in [10, 20, 30]:
        for steps in [1000, 3000]:
            for i in range(n_exp):
                if f"pretrain_{eps}_eps_{steps}_steps_{i}" in on_policy_pretrain_cache:
                    for j, value in enumerate(on_policy_pretrain_cache[f"pretrain_{eps}_eps_{steps}_steps_{i}"]):
                        if isinstance(value, np.ndarray) and value.shape == (1,):
                            on_policy_pretrain_cache[f"pretrain_{eps}_eps_{steps}_steps_{i}"][j] = value[0]
                    l = len(on_policy_pretrain_cache[f"pretrain_{eps}_eps_{steps}_steps_{i}"])
                    if l < n_episodes:
                        on_policy_pretrain_cache[f"pretrain_{eps}_eps_{steps}_steps_{i}"].extend([on_policy_pretrain_cache[f"pretrain_{eps}_eps_{steps}_steps_{i}"][-1]] * (n_episodes - l))
                # print(f"pretrain_{eps}_eps_{steps}_steps_{i}: {on_policy_pretrain_cache[f'pretrain_{eps}_eps_{steps}_steps_{i}'][5:15]}")
    return on_policy_pretrain_cache

def load_cache(env_name, n_eps, suffix):
    with open(
        f"data/cache_{env_name}_Neps_{n_eps}{suffix}.pkl", "rb"
    ) as file:
        cache = pickle.load(file)
    return cache

def load_cache_on_policy_or_rand(env_name, is_rand=False):
    tmp_str = "_rand" if is_rand else ""
    with open(
        f"data/cache_{env_name}_on_policy_pretrain_exp{tmp_str}.pkl", "rb"
    ) as file:
        cache = pickle.load(file)
    return cache

def process_on_policy_pretrain(cache, n_pretrain_eps, n_episodes, n_pretrain_steps, n_exp, is_rand=False):
    n_online_eps = n_episodes - n_pretrain_eps
    returns = np.zeros((n_exp, n_episodes))
    for i in range(n_exp):
        returns[i] = np.squeeze(np.array(cache[
            f"pretrain_{n_pretrain_eps}_eps_{n_pretrain_steps}_steps_{i}{'_rand' if is_rand else ''}"
        ][: n_pretrain_eps]+cache[
            f"pretrain_{n_pretrain_eps}_eps_{n_pretrain_steps}_steps_{i}{'_rand' if is_rand else ''}"
        ][-n_online_eps:]))
    return np.mean(returns, axis=0), np.std(returns, axis=0)

def processing_offline_online_data(avg_offline_returns, online_cache, n_pretrain_eps, online_cache_key, n_exp, n_episodes):
    """
    Merge offline and the average of online data (offline_returns[:n_pretrain_eps] + average(online_cache[online_cache_key][-n_online_eps:])).
    avg_offline_returns: the returns of the offline data
    online_cache: the cache of the online data
    n_pretrain_eps: the number of pretrain episodes
    online_cache_key: the key of the online data
    n_exp: the number of experiments
    n_episodes: the number of episodes

    Returns: the merged data
    """
    returns = np.zeros((n_exp, n_episodes))
    for i in range(n_exp):
        for j in range(n_episodes - n_pretrain_eps):
            returns[i][n_pretrain_eps + j] = online_cache[f"{online_cache_key}_{i}"][j]
    average_returns = np.mean(returns, axis=0)
    std_returns = np.std(returns, axis=0)
    average_returns[:n_pretrain_eps] = avg_offline_returns[:n_pretrain_eps]
    std_returns[:n_pretrain_eps] = np.zeros(n_pretrain_eps)
    return average_returns, std_returns

def extract_data(hyperparams, Qwen_7B, Qwen_32B, DS_7B, DS_14B, Qwen_7B_SFT, mean_random):
    suffix = ""
    cache10 = load_cache(hyperparams["env"].split("-")[0], 10, suffix)
    cache20 = load_cache(hyperparams["env"].split("-")[0], 20, suffix)
    cache30 = load_cache(hyperparams["env"].split("-")[0], 30, suffix)
    on_policy_pretrain_cache = load_cache_on_policy_or_rand(hyperparams["env"].split("-")[0])
    rand_pretrain_cache = load_cache_on_policy_or_rand(hyperparams["env"].split("-")[0], is_rand=True)
    
    try:
        suffix = "SFT"
        cache10SFT = load_cache(hyperparams["env"].split("-")[0], 10, suffix)
        cache20SFT = load_cache(hyperparams["env"].split("-")[0], 20, suffix)
        cache30SFT = load_cache(hyperparams["env"].split("-")[0], 30, suffix)

        mean_pretrain_7b_1000_pre_10SFT, std_pretrain_7b_1000_pre_10SFT = processing_offline_online_data(Qwen_7B, cache10SFT, 10, "pretrain_7b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_7b_3000_pre_10SFT, std_pretrain_7b_3000_pre_10SFT = processing_offline_online_data(Qwen_7B, cache10SFT, 10, "pretrain_7b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_7b_1000_pre_20SFT, std_pretrain_7b_1000_pre_20SFT = processing_offline_online_data(Qwen_7B, cache20SFT, 20, "pretrain_7b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_7b_3000_pre_20SFT, std_pretrain_7b_3000_pre_20SFT = processing_offline_online_data(Qwen_7B, cache20SFT, 20, "pretrain_7b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_7b_1000_pre_30SFT, std_pretrain_7b_1000_pre_30SFT = processing_offline_online_data(Qwen_7B, cache30SFT, 30, "pretrain_7b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_7b_3000_pre_30SFT, std_pretrain_7b_3000_pre_30SFT = processing_offline_online_data(Qwen_7B, cache30SFT, 30, "pretrain_7b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])

        out_dict_SFT = {
            "mean_pretrain_7b_1000_pre_10SFT": mean_pretrain_7b_1000_pre_10SFT,
            "std_pretrain_7b_1000_pre_10SFT": std_pretrain_7b_1000_pre_10SFT,
            "mean_pretrain_7b_3000_pre_10SFT": mean_pretrain_7b_3000_pre_10SFT,
            "std_pretrain_7b_3000_pre_10SFT": std_pretrain_7b_3000_pre_10SFT,
            "mean_pretrain_7b_1000_pre_20SFT": mean_pretrain_7b_1000_pre_20SFT,
            "std_pretrain_7b_1000_pre_20SFT": std_pretrain_7b_1000_pre_20SFT,
            "mean_pretrain_7b_3000_pre_20SFT": mean_pretrain_7b_3000_pre_20SFT,
            "std_pretrain_7b_3000_pre_20SFT": std_pretrain_7b_3000_pre_20SFT,
            "mean_pretrain_7b_1000_pre_30SFT": mean_pretrain_7b_1000_pre_30SFT,
            "std_pretrain_7b_1000_pre_30SFT": std_pretrain_7b_1000_pre_30SFT,
            "mean_pretrain_7b_3000_pre_30SFT": mean_pretrain_7b_3000_pre_30SFT,
            "std_pretrain_7b_3000_pre_30SFT": std_pretrain_7b_3000_pre_30SFT,
        }
    except:
        pass
    try:
        suffix = "DS"
        cache10DS = load_cache(hyperparams["env"].split("-")[0], 10, suffix)
        cache20DS = load_cache(hyperparams["env"].split("-")[0], 20, suffix)
        cache30DS = load_cache(hyperparams["env"].split("-")[0], 30, suffix)

        mean_pretrain_7b_1000_pre_10DS, std_pretrain_7b_1000_pre_10DS = processing_offline_online_data(DS_7B, cache10DS, 10, "pretrain_7b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_7b_3000_pre_10DS, std_pretrain_7b_3000_pre_10DS = processing_offline_online_data(DS_7B, cache10DS, 10, "pretrain_7b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_14b_1000_pre_10DS, std_pretrain_14b_1000_pre_10DS = processing_offline_online_data(DS_14B, cache10DS, 10, "pretrain_32b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_14b_3000_pre_10DS, std_pretrain_14b_3000_pre_10DS = processing_offline_online_data(DS_14B, cache10DS, 10, "pretrain_32b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_7b_1000_pre_20DS, std_pretrain_7b_1000_pre_20DS = processing_offline_online_data(DS_7B, cache20DS, 20, "pretrain_7b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_7b_3000_pre_20DS, std_pretrain_7b_3000_pre_20DS = processing_offline_online_data(DS_7B, cache20DS, 20, "pretrain_7b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_14b_1000_pre_20DS, std_pretrain_14b_1000_pre_20DS = processing_offline_online_data(DS_14B, cache20DS, 20, "pretrain_32b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_14b_3000_pre_20DS, std_pretrain_14b_3000_pre_20DS = processing_offline_online_data(DS_14B, cache20DS, 20, "pretrain_32b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_7b_1000_pre_30DS, std_pretrain_7b_1000_pre_30DS = processing_offline_online_data(DS_7B, cache30DS, 30, "pretrain_7b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_7b_3000_pre_30DS, std_pretrain_7b_3000_pre_30DS = processing_offline_online_data(DS_7B, cache30DS, 30, "pretrain_7b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_14b_1000_pre_30DS, std_pretrain_14b_1000_pre_30DS = processing_offline_online_data(DS_14B, cache30DS, 30, "pretrain_32b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
        mean_pretrain_14b_3000_pre_30DS, std_pretrain_14b_3000_pre_30DS = processing_offline_online_data(DS_14B, cache30DS, 30, "pretrain_32b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])

        out_dict_DS = {
            "mean_pretrain_7b_1000_pre_10DS": mean_pretrain_7b_1000_pre_10DS,
            "std_pretrain_7b_1000_pre_10DS": std_pretrain_7b_1000_pre_10DS,
            "mean_pretrain_7b_3000_pre_10DS": mean_pretrain_7b_3000_pre_10DS,
            "std_pretrain_7b_3000_pre_10DS": std_pretrain_7b_3000_pre_10DS,
            "mean_pretrain_14b_1000_pre_10DS": mean_pretrain_14b_1000_pre_10DS,
            "std_pretrain_14b_1000_pre_10DS": std_pretrain_14b_1000_pre_10DS,
            "mean_pretrain_14b_3000_pre_10DS": mean_pretrain_14b_3000_pre_10DS,
            "std_pretrain_14b_3000_pre_10DS": std_pretrain_14b_3000_pre_10DS,
            "mean_pretrain_7b_1000_pre_20DS": mean_pretrain_7b_1000_pre_20DS,
            "std_pretrain_7b_1000_pre_20DS": std_pretrain_7b_1000_pre_20DS,
            "mean_pretrain_7b_3000_pre_20DS": mean_pretrain_7b_3000_pre_20DS,
            "std_pretrain_7b_3000_pre_20DS": std_pretrain_7b_3000_pre_20DS,
            "mean_pretrain_14b_1000_pre_20DS": mean_pretrain_14b_1000_pre_20DS,
            "std_pretrain_14b_1000_pre_20DS": std_pretrain_14b_1000_pre_20DS,
            "mean_pretrain_14b_3000_pre_20DS": mean_pretrain_14b_3000_pre_20DS,
            "std_pretrain_14b_3000_pre_20DS": std_pretrain_14b_3000_pre_20DS,
            "mean_pretrain_7b_1000_pre_30DS": mean_pretrain_7b_1000_pre_30DS,
            "std_pretrain_7b_1000_pre_30DS": std_pretrain_7b_1000_pre_30DS,
            "mean_pretrain_7b_3000_pre_30DS": mean_pretrain_7b_3000_pre_30DS,
            "std_pretrain_7b_3000_pre_30DS": std_pretrain_7b_3000_pre_30DS,
            "mean_pretrain_14b_1000_pre_30DS": mean_pretrain_14b_1000_pre_30DS,
            "std_pretrain_14b_1000_pre_30DS": std_pretrain_14b_1000_pre_30DS,
            "mean_pretrain_14b_3000_pre_30DS": mean_pretrain_14b_3000_pre_30DS,
            "std_pretrain_14b_3000_pre_30DS": std_pretrain_14b_3000_pre_30DS,
        }
    except:
        pass
    online_returns = np.zeros((hyperparams["n_exp"], hyperparams["n_episodes"]))

    on_policy_pretrain_cache = reformat_on_policy_pretrain_cache(on_policy_pretrain_cache, hyperparams["n_exp"], hyperparams["n_episodes"])

    mean_on_pol_pretrain_1000_pre_10, std_on_pol_pretrain_1000_pre_10 = process_on_policy_pretrain(on_policy_pretrain_cache, 10, hyperparams["n_episodes"], 1000, hyperparams["n_exp"])
    mean_on_pol_pretrain_3000_pre_10, std_on_pol_pretrain_3000_pre_10 = process_on_policy_pretrain(on_policy_pretrain_cache, 10, hyperparams["n_episodes"], 3000, hyperparams["n_exp"])
    mean_on_pol_pretrain_1000_pre_20, std_on_pol_pretrain_1000_pre_20 = process_on_policy_pretrain(on_policy_pretrain_cache, 20, hyperparams["n_episodes"], 1000, hyperparams["n_exp"])
    mean_on_pol_pretrain_3000_pre_20, std_on_pol_pretrain_3000_pre_20 = process_on_policy_pretrain(on_policy_pretrain_cache, 20, hyperparams["n_episodes"], 3000, hyperparams["n_exp"])
    mean_on_pol_pretrain_1000_pre_30, std_on_pol_pretrain_1000_pre_30 = process_on_policy_pretrain(on_policy_pretrain_cache, 30, hyperparams["n_episodes"], 1000, hyperparams["n_exp"])
    mean_on_pol_pretrain_3000_pre_30, std_on_pol_pretrain_3000_pre_30 = process_on_policy_pretrain(on_policy_pretrain_cache, 30, hyperparams["n_episodes"], 3000, hyperparams["n_exp"])

    mean_rand_pretrain_1000_pre_10, std_rand_pretrain_1000_pre_10 = process_on_policy_pretrain(rand_pretrain_cache, 10, hyperparams["n_episodes"], 1000, hyperparams["n_exp"], True)
    mean_rand_pretrain_3000_pre_10, std_rand_pretrain_3000_pre_10 = process_on_policy_pretrain(rand_pretrain_cache, 10, hyperparams["n_episodes"], 3000, hyperparams["n_exp"], True)
    mean_rand_pretrain_1000_pre_20, std_rand_pretrain_1000_pre_20 = process_on_policy_pretrain(rand_pretrain_cache, 20, hyperparams["n_episodes"], 1000, hyperparams["n_exp"], True)
    mean_rand_pretrain_3000_pre_20, std_rand_pretrain_3000_pre_20 = process_on_policy_pretrain(rand_pretrain_cache, 20, hyperparams["n_episodes"], 3000, hyperparams["n_exp"], True)
    mean_rand_pretrain_1000_pre_30, std_rand_pretrain_1000_pre_30 = process_on_policy_pretrain(rand_pretrain_cache, 30, hyperparams["n_episodes"], 1000, hyperparams["n_exp"], True)
    mean_rand_pretrain_3000_pre_30, std_rand_pretrain_3000_pre_30 = process_on_policy_pretrain(rand_pretrain_cache, 30, hyperparams["n_episodes"], 3000, hyperparams["n_exp"], True)

    mean_mix_7b_pre_10, std_mix_7b_pre_10 = processing_offline_online_data(Qwen_7B, cache10, 10, "finetune_7b", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_mix_32b_pre_10, std_mix_32b_pre_10 = processing_offline_online_data(Qwen_32B,  cache10, 10, "finetune_32b", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_pretrain_32b_1000_pre_10, std_pretrain_32b_1000_pre_10 = processing_offline_online_data(Qwen_32B, cache10, 10, "pretrain_32b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_pretrain_7b_1000_pre_10, std_pretrain_7b_1000_pre_10 = processing_offline_online_data(Qwen_7B, cache10, 10, "pretrain_7b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_pretrain_32b_3000_pre_10, std_pretrain_32b_3000_pre_10 = processing_offline_online_data(Qwen_32B, cache10, 10, "pretrain_32b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_pretrain_7b_3000_pre_10, std_pretrain_7b_3000_pre_10 = processing_offline_online_data(Qwen_7B, cache10, 10, "pretrain_7b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])
    
    mean_mix_7b_pre_20, std_mix_7b_pre_20 = processing_offline_online_data(Qwen_7B, cache20, 20, "finetune_7b", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_mix_32b_pre_20, std_mix_32b_pre_20 = processing_offline_online_data(Qwen_32B,  cache20, 20, "finetune_32b", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_pretrain_32b_1000_pre_20, std_pretrain_32b_1000_pre_20 = processing_offline_online_data(Qwen_32B, cache20, 20, "pretrain_32b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_pretrain_7b_1000_pre_20, std_pretrain_7b_1000_pre_20 = processing_offline_online_data(Qwen_7B, cache20, 20, "pretrain_7b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_pretrain_32b_3000_pre_20, std_pretrain_32b_3000_pre_20 = processing_offline_online_data(Qwen_32B, cache20, 20, "pretrain_32b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_pretrain_7b_3000_pre_20, std_pretrain_7b_3000_pre_20 = processing_offline_online_data(Qwen_7B, cache20, 20, "pretrain_7b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])

    mean_mix_7b_pre_30, std_mix_7b_pre_30 = processing_offline_online_data(Qwen_7B, cache30, 30, "finetune_7b", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_mix_32b_pre_30, std_mix_32b_pre_30 = processing_offline_online_data(Qwen_32B,  cache30, 30, "finetune_32b", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_pretrain_32b_1000_pre_30, std_pretrain_32b_1000_pre_30 = processing_offline_online_data(Qwen_32B, cache30, 30, "pretrain_32b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_pretrain_7b_1000_pre_30, std_pretrain_7b_1000_pre_30 = processing_offline_online_data(Qwen_7B, cache30, 30, "pretrain_7b_1000", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_pretrain_32b_3000_pre_30, std_pretrain_32b_3000_pre_30 = processing_offline_online_data(Qwen_32B, cache30, 30, "pretrain_32b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])
    mean_pretrain_7b_3000_pre_30, std_pretrain_7b_3000_pre_30 = processing_offline_online_data(Qwen_7B, cache30, 30, "pretrain_7b_3000", hyperparams["n_exp"], hyperparams["n_episodes"])

    for i in range(hyperparams["n_exp"]):
        online_returns[i] = cache10[f"online_{i}"][: hyperparams["n_episodes"]]

    x = range(hyperparams["n_episodes"])
    mean_onl = np.mean(online_returns, axis=0)
    std_onl = np.std(online_returns, axis=0)

    out_dict = {
        "x": x,
        "mean_onl": mean_onl,
        "std_onl": std_onl,
        "Qwen_7B": Qwen_7B,
        "Qwen_32B": Qwen_32B,
        "mean_random": mean_random,
        "mean_mix_7b_pre_10": mean_mix_7b_pre_10,
        "mean_mix_32b_pre_10": mean_mix_32b_pre_10,
        "mean_pretrain_32b_1000_pre_10": mean_pretrain_32b_1000_pre_10,
        "mean_pretrain_7b_1000_pre_10": mean_pretrain_7b_1000_pre_10,
        "mean_pretrain_32b_3000_pre_10": mean_pretrain_32b_3000_pre_10,
        "mean_pretrain_7b_3000_pre_10": mean_pretrain_7b_3000_pre_10,
        "std_mix_32b_pre_10": std_mix_32b_pre_10,
        "std_mix_7b_pre_10": std_mix_7b_pre_10,
        "std_pretrain_32b_1000_pre_10": std_pretrain_32b_1000_pre_10,
        "std_pretrain_7b_1000_pre_10": std_pretrain_7b_1000_pre_10,
        "std_pretrain_32b_3000_pre_10": std_pretrain_32b_3000_pre_10,
        "std_pretrain_7b_3000_pre_10": std_pretrain_7b_3000_pre_10,
        "mean_on_pol_pretrain_1000_pre_10": mean_on_pol_pretrain_1000_pre_10,
        "std_on_pol_pretrain_1000_pre_10": std_on_pol_pretrain_1000_pre_10,
        "mean_on_pol_pretrain_3000_pre_10": mean_on_pol_pretrain_3000_pre_10,
        "std_on_pol_pretrain_3000_pre_10": std_on_pol_pretrain_3000_pre_10,
        "mean_rand_pretrain_1000_pre_10": mean_rand_pretrain_1000_pre_10,
        "std_rand_pretrain_1000_pre_10": std_rand_pretrain_1000_pre_10,
        "mean_rand_pretrain_3000_pre_10": mean_rand_pretrain_3000_pre_10,
        "std_rand_pretrain_3000_pre_10": std_rand_pretrain_3000_pre_10,
        "mean_mix_7b_pre_20": mean_mix_7b_pre_20,  # 20 here
        "mean_mix_32b_pre_20": mean_mix_32b_pre_20,
        "mean_pretrain_32b_1000_pre_20": mean_pretrain_32b_1000_pre_20,
        "mean_pretrain_7b_1000_pre_20": mean_pretrain_7b_1000_pre_20,
        "mean_pretrain_32b_3000_pre_20": mean_pretrain_32b_3000_pre_20,
        "mean_pretrain_7b_3000_pre_20": mean_pretrain_7b_3000_pre_20,
        "std_mix_32b_pre_20": std_mix_32b_pre_20,
        "std_mix_7b_pre_20": std_mix_7b_pre_20,
        "std_pretrain_32b_1000_pre_20": std_pretrain_32b_1000_pre_20,
        "std_pretrain_7b_1000_pre_20": std_pretrain_7b_1000_pre_20,
        "std_pretrain_32b_3000_pre_20": std_pretrain_32b_3000_pre_20,
        "std_pretrain_7b_3000_pre_20": std_pretrain_7b_3000_pre_20,
        "mean_on_pol_pretrain_1000_pre_20": mean_on_pol_pretrain_1000_pre_20,
        "std_on_pol_pretrain_1000_pre_20": std_on_pol_pretrain_1000_pre_20,
        "mean_on_pol_pretrain_3000_pre_20": mean_on_pol_pretrain_3000_pre_20,
        "std_on_pol_pretrain_3000_pre_20": std_on_pol_pretrain_3000_pre_20,
        "mean_rand_pretrain_1000_pre_20": mean_rand_pretrain_1000_pre_20,
        "std_rand_pretrain_1000_pre_20": std_rand_pretrain_1000_pre_20,
        "mean_rand_pretrain_3000_pre_20": mean_rand_pretrain_3000_pre_20,
        "std_rand_pretrain_3000_pre_20": std_rand_pretrain_3000_pre_20,
        "mean_mix_7b_pre_30": mean_mix_7b_pre_30,  # 30 here
        "mean_mix_32b_pre_30": mean_mix_32b_pre_30,
        "mean_pretrain_32b_1000_pre_30": mean_pretrain_32b_1000_pre_30,
        "mean_pretrain_7b_1000_pre_30": mean_pretrain_7b_1000_pre_30,
        "mean_pretrain_32b_3000_pre_30": mean_pretrain_32b_3000_pre_30,
        "mean_pretrain_7b_3000_pre_30": mean_pretrain_7b_3000_pre_30,
        "std_mix_32b_pre_30": std_mix_32b_pre_30,
        "std_mix_7b_pre_30": std_mix_7b_pre_30,
        "std_pretrain_32b_1000_pre_30": std_pretrain_32b_1000_pre_30,
        "std_pretrain_7b_1000_pre_30": std_pretrain_7b_1000_pre_30,
        "std_pretrain_32b_3000_pre_30": std_pretrain_32b_3000_pre_30,
        "std_pretrain_7b_3000_pre_30": std_pretrain_7b_3000_pre_30,
        "mean_on_pol_pretrain_1000_pre_30": mean_on_pol_pretrain_1000_pre_30,
        "std_on_pol_pretrain_1000_pre_30": std_on_pol_pretrain_1000_pre_30,
        "mean_on_pol_pretrain_3000_pre_30": mean_on_pol_pretrain_3000_pre_30,
        "std_on_pol_pretrain_3000_pre_30": std_on_pol_pretrain_3000_pre_30,
        "mean_rand_pretrain_1000_pre_30": mean_rand_pretrain_1000_pre_30,
        "std_rand_pretrain_1000_pre_30": std_rand_pretrain_1000_pre_30,
        "mean_rand_pretrain_3000_pre_30": mean_rand_pretrain_3000_pre_30,
        "std_rand_pretrain_3000_pre_30": std_rand_pretrain_3000_pre_30,
    }
    try:
        for k, v in out_dict_SFT.items():
            out_dict[k] = v
    except:
        pass
    try:
        for k, v in out_dict_DS.items():
            out_dict[k] = v
    except:
        pass

    if hyperparams["env"] == "CliffWalking-v0" or hyperparams["env"] == "FrozenLake-v1":
        print("Environment is CliffWalking or FrozenLake. Extracting dataset for Transfer Coefficient calculation ...")
        for i in range(hyperparams["n_exp"]):
            for n_eps, cache in [(10, cache10), (20, cache20), (30, cache30)]:
                out_dict[f"pretrain_7b_1000_{n_eps}_{i}_dataset"] = cache[f"pretrain_7b_1000_{i}_dataset"]
                out_dict[f"pretrain_32b_1000_{n_eps}_{i}_dataset"] = cache[f"pretrain_32b_1000_{i}_dataset"]
                out_dict[f"on_pol_1000_{n_eps}_{i}_dataset"] = on_policy_pretrain_cache[f"pretrain_{n_eps}_eps_1000_steps_{i}_dataset"]
                out_dict[f"on_pol_1000_{n_eps}_{i}_offline_dataset"] = on_policy_pretrain_cache[f"pretrain_{n_eps}_eps_1000_steps_{i}_offline_dataset"]
                out_dict[f"rand_1000_{n_eps}_{i}_dataset"] = rand_pretrain_cache[f"pretrain_{n_eps}_eps_1000_steps_{i}_rand_dataset"]
                out_dict[f"rand_1000_{n_eps}_{i}_offline_dataset"] = rand_pretrain_cache[f"pretrain_{n_eps}_eps_1000_steps_{i}_rand_offline_dataset"]
    return out_dict


def plot_main(hyperparams, cache, Qwen_7B, Qwen_32B, mean_random):
    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(12, 6))
    plt.plot(
        cache["mean_onl"], label=f'On-policy. Cum. reward={int(sum(cache["mean_onl"]))}'
    )
    plt.fill_between(
        cache["x"],
        cache["mean_onl"] - cache["std_onl"] / np.sqrt(hyperparams["n_exp"]),
        cache["mean_onl"] + cache["std_onl"] / np.sqrt(hyperparams["n_exp"]),
        alpha=0.3,
    )
    plt.plot(
        cache["mean_pretrain_7b_3000_pre_10"],
        label=f'LORO. Cum. reward={int(sum(cache["mean_pretrain_7b_3000_pre_10"]))}',
    )
    plt.fill_between(
        cache["x"],
        cache["mean_pretrain_7b_3000_pre_10"]
        - cache["std_pretrain_7b_3000_pre_10"] / np.sqrt(hyperparams["n_exp"]),
        cache["mean_pretrain_7b_3000_pre_10"]
        + cache["std_pretrain_7b_3000_pre_10"] / np.sqrt(hyperparams["n_exp"]),
        alpha=0.3,
    )

    plt.plot(Qwen_7B, label=f"Qwen-7B. Cum. reward={int(sum(Qwen_7B))}", linestyle=":")
    plt.plot(
        Qwen_32B, label=f"Qwen-32B. Cum. reward={int(sum(Qwen_32B))}", linestyle=":"
    )
    plt.plot(
        mean_random, label=f"Random. Cum. reward={int(sum(mean_random))}", linestyle=":"
    )

    plt.title(f'{hyperparams["env"].split("-")[0]}')
    plt.xlabel("# of episodes")
    plt.ylabel("Episode reward")
    if "MountainCar" in hyperparams["env"]:
        plt.legend(loc="upper right")
    else:
        plt.legend(loc="lower right")
    # plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    if "Cliff" in hyperparams["env"]:
        plt.ylim(-500, 0)
    if "MountainCar" in hyperparams["env"]:
        plt.ylim(-210, -50)
    # plt.xticks(np.arange(0, 101, 10))
    plt.savefig(
        f'figs/{hyperparams["env"].split("-")[0]}_main.pdf',
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.show()


def plot_pretrain(hyperparams, cache):
    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(12, 6))
    plt.plot(
        cache["mean_onl"], label=f'On-policy. Cum. reward={int(sum(cache["mean_onl"]))}'
    )
    plt.fill_between(
        cache["x"],
        cache["mean_onl"] - cache["std_onl"] / np.sqrt(hyperparams["n_exp"]),
        cache["mean_onl"] + cache["std_onl"] / np.sqrt(hyperparams["n_exp"]),
        alpha=0.3,
    )
    plt.plot(
        cache["mean_pretrain_7b_3000_pre_10"],
        label=f'LORO. Cum. reward={int(sum(cache["mean_pretrain_7b_3000_pre_10"]))}',
    )
    plt.fill_between(
        cache["x"],
        cache["mean_pretrain_7b_3000_pre_10"]
        - cache["std_pretrain_7b_3000_pre_10"] / np.sqrt(hyperparams["n_exp"]),
        cache["mean_pretrain_7b_3000_pre_10"]
        + cache["std_pretrain_7b_3000_pre_10"] / np.sqrt(hyperparams["n_exp"]),
        alpha=0.3,
    )
    plt.plot(
        cache["mean_on_pol_pretrain_3000_pre_10"],
        label=f'Pretrain w/ On-policy data. Cum. reward={int(sum(cache["mean_on_pol_pretrain_3000_pre_10"]))}',
    )
    plt.fill_between(
        cache["x"],
        cache["mean_on_pol_pretrain_3000_pre_10"]
        - cache["std_on_pol_pretrain_3000_pre_10"] / np.sqrt(hyperparams["n_exp"]),
        cache["mean_on_pol_pretrain_3000_pre_10"]
        + cache["std_on_pol_pretrain_3000_pre_10"] / np.sqrt(hyperparams["n_exp"]),
        alpha=0.3,
    )
    plt.plot(
        cache["mean_rand_pretrain_3000_pre_10"],
        label=f'Pretrain w/ random policy data. Cum. reward={int(sum(cache["mean_rand_pretrain_3000_pre_10"]))}',
    )
    plt.fill_between(
        cache["x"],
        cache["mean_rand_pretrain_3000_pre_10"]
        - cache["std_rand_pretrain_3000_pre_10"] / np.sqrt(hyperparams["n_exp"]),
        cache["mean_rand_pretrain_3000_pre_10"]
        + cache["std_rand_pretrain_3000_pre_10"] / np.sqrt(hyperparams["n_exp"]),
        alpha=0.3,
    )

    plt.title(f'{hyperparams["env"].split("-")[0]}')
    plt.xlabel("# of episodes")
    plt.ylabel("Episode reward")
    if "Cliff" in hyperparams["env"] or "MountainCar" in hyperparams["env"]:
        plt.ylim(-500, 0)
    if "MountainCar" in hyperparams["env"]:
        plt.ylim(-210, -50)
    if "MountainCar" in hyperparams["env"]:
        plt.legend(loc="upper right")
    else:
        plt.legend(loc="lower right")
    plt.savefig(
        f'figs/{hyperparams["env"].split("-")[0]}_pretrain.pdf',
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.show()


def plot_mix(hyperparams, cache):
    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(12, 6))
    plt.plot(
        cache["mean_onl"], label=f'On-policy. Cum. reward={int(sum(cache["mean_onl"]))}'
    )
    plt.fill_between(
        cache["x"],
        cache["mean_onl"] - cache["std_onl"] / np.sqrt(hyperparams["n_exp"]),
        cache["mean_onl"] + cache["std_onl"] / np.sqrt(hyperparams["n_exp"]),
        alpha=0.3,
    )
    plt.plot(
        cache["mean_pretrain_7b_3000_pre_10"],
        label=f'LORO. Cum. reward={int(sum(cache["mean_pretrain_7b_3000_pre_10"]))}',
    )
    plt.fill_between(
        cache["x"],
        cache["mean_pretrain_7b_3000_pre_10"]
        - cache["std_pretrain_7b_3000_pre_10"] / np.sqrt(hyperparams["n_exp"]),
        cache["mean_pretrain_7b_3000_pre_10"]
        + cache["std_pretrain_7b_3000_pre_10"] / np.sqrt(hyperparams["n_exp"]),
        alpha=0.3,
    )
    plt.plot(
        cache["mean_mix_7b_pre_10"],
        label=f'Mix data w/o pretrain. Cum. reward={int(sum(cache["mean_mix_7b_pre_10"]))}',
    )
    plt.fill_between(
        cache["x"],
        cache["mean_mix_7b_pre_10"]
        - cache["std_mix_7b_pre_10"] / np.sqrt(hyperparams["n_exp"]),
        cache["mean_mix_7b_pre_10"]
        + cache["std_mix_7b_pre_10"] / np.sqrt(hyperparams["n_exp"]),
        alpha=0.3,
    )

    plt.title(f'{hyperparams["env"].split("-")[0]}')
    plt.xlabel("# of episodes")
    plt.ylabel("Episode reward")
    if "Cliff" in hyperparams["env"]:
        plt.ylim(-500, 0)
    if "MountainCar" in hyperparams["env"]:
        plt.ylim(-210, -50)
    if "MountainCar" in hyperparams["env"]:
        plt.legend(loc="upper right")
    else:
        plt.legend(loc="lower right")
    plt.savefig(
        f'figs/{hyperparams["env"].split("-")[0]}_mix.pdf',
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.show()


def plot_sft_lcot(hyperparams, cache):
    plt.rcParams["font.size"] = 12
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(
            cache["mean_onl"],
            label=f'On-policy. Cum. reward={int(sum(cache["mean_onl"]))}',
        )
        plt.fill_between(
            cache["x"],
            cache["mean_onl"] - cache["std_onl"] / np.sqrt(hyperparams["n_exp"]),
            cache["mean_onl"] + cache["std_onl"] / np.sqrt(hyperparams["n_exp"]),
            alpha=0.3,
        )
        plt.plot(
            cache["mean_pretrain_7b_3000_pre_10"],
            label=f'LORO. Cum. reward={int(sum(cache["mean_pretrain_7b_3000_pre_10"]))}',
        )
        plt.fill_between(
            cache["x"],
            cache["mean_pretrain_7b_3000_pre_10"]
            - cache["std_pretrain_7b_3000_pre_10"] / np.sqrt(hyperparams["n_exp"]),
            cache["mean_pretrain_7b_3000_pre_10"]
            + cache["std_pretrain_7b_3000_pre_10"] / np.sqrt(hyperparams["n_exp"]),
            alpha=0.3,
        )
        plt.plot(
            cache["mean_pretrain_7b_3000_pre_10SFT"],
            label=f'SFT. Cum. reward={int(sum(cache["mean_pretrain_7b_3000_pre_10SFT"]))}',
        )
        plt.fill_between(
            cache["x"],
            cache["mean_pretrain_7b_3000_pre_10SFT"]
            - cache["std_pretrain_7b_3000_pre_10SFT"] / np.sqrt(hyperparams["n_exp"]),
            cache["mean_pretrain_7b_3000_pre_10SFT"]
            + cache["std_pretrain_7b_3000_pre_10SFT"] / np.sqrt(hyperparams["n_exp"]),
            alpha=0.3,
        )

        plt.title(f'{hyperparams["env"].split("-")[0]}')
        plt.xlabel("# of episodes")
        plt.ylabel("Episode reward")
        if "Cliff" in hyperparams["env"]:
            plt.ylim(-500, 0)
        if "MountainCar" in hyperparams["env"]:
            plt.ylim(-210, -50)
        if "MountainCar" in hyperparams["env"]:
            plt.legend(loc="upper right")
        else:
            plt.legend(loc="lower right")
    except:
        pass
    try:
        plt.plot(
            cache["mean_pretrain_7b_3000_pre_10DS"],
            label=f'Long-CoT 7B. Cum. reward={int(sum(cache["mean_pretrain_7b_3000_pre_10DS"]))}',
        )
        plt.fill_between(
            cache["x"],
            cache["mean_pretrain_7b_3000_pre_10DS"]
            - cache["std_pretrain_7b_3000_pre_10DS"] / np.sqrt(hyperparams["n_exp"]),
            cache["mean_pretrain_7b_3000_pre_10DS"]
            + cache["std_pretrain_7b_3000_pre_10DS"] / np.sqrt(hyperparams["n_exp"]),
            alpha=0.3,
        )
        plt.plot(
            cache["mean_pretrain_14b_3000_pre_10DS"],
            label=f'Long-CoT 14B. Cum. reward={int(sum(cache["mean_pretrain_14b_3000_pre_10DS"]))}',
        )
        plt.fill_between(
            cache["x"],
            cache["mean_pretrain_14b_3000_pre_10DS"]
            - cache["std_pretrain_14b_3000_pre_10DS"] / np.sqrt(hyperparams["n_exp"]),
            cache["mean_pretrain_14b_3000_pre_10DS"]
            + cache["std_pretrain_14b_3000_pre_10DS"] / np.sqrt(hyperparams["n_exp"]),
            alpha=0.3,
        )
    except:
        pass
    plt.savefig(
        f'figs/{hyperparams["env"].split("-")[0]}_sft_lcot.pdf',
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.show()


def plot_model_size(hyperparams, cache):
    plt.rcParams["font.size"] = 12
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    for j in range(2):
        if j == 0:
            pretrain_step = 1000
        else:
            pretrain_step = 3000
        for i in range(3):
            std_onl = cache[f"std_onl"] / np.sqrt(hyperparams["n_exp"])
            mean_loro7 = cache[f"mean_pretrain_7b_{pretrain_step}_pre_{i+1}0"]
            std_loro7 = cache[f"std_pretrain_7b_{pretrain_step}_pre_{i+1}0"] / np.sqrt(
                hyperparams["n_exp"]
            )
            mean_loro32 = cache[f"mean_pretrain_32b_{pretrain_step}_pre_{i+1}0"]
            std_loro32 = cache[
                f"std_pretrain_32b_{pretrain_step}_pre_{i+1}0"
            ] / np.sqrt(hyperparams["n_exp"])

            rbf = Rbf(
                cache["x"],
                cache["mean_onl"],
                function="multiquadric",
                smooth=hyperparams["smooth"],
            )
            smooth_mean_onl = rbf(cache["x"])
            rbf = Rbf(
                cache["x"],
                mean_loro7,
                function="multiquadric",
                smooth=hyperparams["smooth"],
            )
            smooth_mean_loro7 = rbf(cache["x"])
            rbf = Rbf(
                cache["x"],
                mean_loro32,
                function="multiquadric",
                smooth=hyperparams["smooth"],
            )
            smooth_mean_loro32 = rbf(cache["x"])

            axs[j, i].plot(
                smooth_mean_onl,
                label=f'On-policy. Cum. reward={int(sum(cache["mean_onl"]))}',
            )
            axs[j, i].fill_between(
                cache["x"],
                smooth_mean_onl - std_onl,
                smooth_mean_onl + std_onl,
                alpha=0.3,
            )
            axs[j, i].plot(
                smooth_mean_loro7, label=f"LORO 7B. Cum. reward={int(sum(mean_loro7))}"
            )
            axs[j, i].fill_between(
                cache["x"],
                smooth_mean_loro7 - std_loro7,
                smooth_mean_loro7 + std_loro7,
                alpha=0.3,
            )
            axs[j, i].plot(
                smooth_mean_loro32,
                label=f"LORO 32B. Cum. reward={int(sum(mean_loro32))}",
            )
            axs[j, i].fill_between(
                cache["x"],
                smooth_mean_loro32 - std_loro32,
                smooth_mean_loro32 + std_loro32,
                alpha=0.3,
            )

            axs[j, i].set_title(f"Smoothed pretrain {i+1}0 eps, {pretrain_step} steps")
            axs[j, i].set_xlabel("# of episodes")
            axs[j, i].set_ylabel("Episode reward")
            if "Cliff" in hyperparams["env"]:
                axs[j, i].set_ylim(-500, 0)
            if "MountainCar" in hyperparams["env"]:
                axs[j, i].set_ylim(-210, -50)

    if "MountainCar" in hyperparams["env"]:
        axs[0, 0].legend(loc="upper right")
    else:
        axs[0, 0].legend(loc="lower right")
    plt.savefig(
        f'figs/{hyperparams["env"].split("-")[0]}_model_size.pdf',
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.show()


def plot_pretrain_step(hyperparams, cache):
    plt.rcParams["font.size"] = 12
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    for j in range(2):
        if j == 0:
            model_size = 7
        else:
            model_size = 32
        for i in range(3):
            mean_onl = cache[f"mean_onl"]
            std_onl = cache[f"std_onl"] / np.sqrt(hyperparams["n_exp"])
            mean_loro_1k = cache[f"mean_pretrain_{model_size}b_1000_pre_{i+1}0"]
            std_loro_1k = cache[
                f"std_pretrain_{model_size}b_1000_pre_{i+1}0"
            ] / np.sqrt(hyperparams["n_exp"])
            mean_loro_3k = cache[f"mean_pretrain_{model_size}b_3000_pre_{i+1}0"]
            std_loro_3k = cache[
                f"std_pretrain_{model_size}b_3000_pre_{i+1}0"
            ] / np.sqrt(hyperparams["n_exp"])

            rbf = Rbf(
                cache["x"],
                cache["mean_onl"],
                function="multiquadric",
                smooth=hyperparams["smooth"],
            )
            smooth_mean_onl = rbf(cache["x"])
            rbf = Rbf(
                cache["x"],
                mean_loro_1k,
                function="multiquadric",
                smooth=hyperparams["smooth"],
            )
            smooth_mean_loro_1k = rbf(cache["x"])
            rbf = Rbf(
                cache["x"],
                mean_loro_3k,
                function="multiquadric",
                smooth=hyperparams["smooth"],
            )
            smooth_mean_loro_3k = rbf(cache["x"])

            axs[j, i].plot(
                smooth_mean_onl,
                label=f'On-policy. Cum. reward={int(sum(cache["mean_onl"]))}',
            )
            axs[j, i].fill_between(
                cache["x"],
                smooth_mean_onl - std_onl,
                smooth_mean_onl + std_onl,
                alpha=0.3,
            )
            axs[j, i].plot(
                smooth_mean_loro_1k,
                label=f"LORO 1k pretrain steps. Cum. reward={int(sum(mean_loro_1k))}",
            )
            axs[j, i].fill_between(
                cache["x"],
                smooth_mean_loro_1k - std_loro_1k,
                smooth_mean_loro_1k + std_loro_1k,
                alpha=0.3,
            )
            axs[j, i].plot(
                smooth_mean_loro_3k,
                label=f"LORO 3k pretrain steps. Cum. reward={int(sum(mean_loro_3k))}",
            )
            axs[j, i].fill_between(
                cache["x"],
                smooth_mean_loro_3k - std_loro_3k,
                smooth_mean_loro_3k + std_loro_3k,
                alpha=0.3,
            )

            axs[j, i].set_title(f"Smoothed pretrain {i+1}0 eps, {model_size}B")
            axs[j, i].set_xlabel("# of episodes")
            axs[j, i].set_ylabel("Episode reward")
            if "Cliff" in hyperparams["env"]:
                axs[j, i].set_ylim(-500, 0)
            if "MountainCar" in hyperparams["env"]:
                axs[j, i].set_ylim(-210, -50)

    if "MountainCar" in hyperparams["env"]:
        axs[0, 0].legend(loc="upper right")
    else:
        axs[0, 0].legend(loc="lower right")
    plt.savefig(
        f'figs/{hyperparams["env"].split("-")[0]}_pretrain_step.pdf',
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.show()


def plot_pretrain_eps(hyperparams, cache):
    plt.rcParams["font.size"] = 12
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    for j in range(2):
        if j == 0:
            model_size = 7
        else:
            model_size = 32
        for i in range(2):
            if i == 0:
                pretrain_step = 1000
            else:
                pretrain_step = 3000
            mean_onl = cache[f"mean_onl"]
            std_onl = cache[f"std_onl"] / np.sqrt(hyperparams["n_exp"])
            mean_loro_10 = cache[f"mean_pretrain_{model_size}b_{pretrain_step}_pre_10"]
            std_loro_10 = cache[
                f"std_pretrain_{model_size}b_{pretrain_step}_pre_10"
            ] / np.sqrt(hyperparams["n_exp"])
            mean_loro_20 = cache[f"mean_pretrain_{model_size}b_{pretrain_step}_pre_20"]
            std_loro_20 = cache[
                f"std_pretrain_{model_size}b_{pretrain_step}_pre_20"
            ] / np.sqrt(hyperparams["n_exp"])
            mean_loro_30 = cache[f"mean_pretrain_{model_size}b_{pretrain_step}_pre_30"]
            std_loro_30 = cache[
                f"std_pretrain_{model_size}b_{pretrain_step}_pre_30"
            ] / np.sqrt(hyperparams["n_exp"])

            rbf = Rbf(
                cache["x"],
                cache["mean_onl"],
                function="multiquadric",
                smooth=hyperparams["smooth"],
            )
            smooth_mean_onl = rbf(cache["x"])
            rbf = Rbf(
                cache["x"],
                mean_loro_10,
                function="multiquadric",
                smooth=hyperparams["smooth"],
            )
            smooth_mean_loro_10 = rbf(cache["x"])
            rbf = Rbf(
                cache["x"],
                mean_loro_20,
                function="multiquadric",
                smooth=hyperparams["smooth"],
            )
            smooth_mean_loro_20 = rbf(cache["x"])
            rbf = Rbf(
                cache["x"],
                mean_loro_30,
                function="multiquadric",
                smooth=hyperparams["smooth"],
            )
            smooth_mean_loro_30 = rbf(cache["x"])

            axs[j, i].plot(
                smooth_mean_onl, label=f"On-policy. Cum. reward={int(sum(mean_onl))}"
            )
            axs[j, i].fill_between(
                cache["x"],
                smooth_mean_onl - std_onl,
                smooth_mean_onl + std_onl,
                alpha=0.3,
            )
            axs[j, i].plot(
                smooth_mean_loro_10,
                label=f"LORO 10 pretrain eps. Cum. reward={int(sum(mean_loro_10))}",
            )
            axs[j, i].fill_between(
                cache["x"],
                smooth_mean_loro_10 - std_loro_10,
                smooth_mean_loro_10 + std_loro_10,
                alpha=0.3,
            )
            axs[j, i].plot(
                smooth_mean_loro_20,
                label=f"LORO 20 pretrain eps. Cum. reward={int(sum(mean_loro_20))}",
            )
            axs[j, i].fill_between(
                cache["x"],
                smooth_mean_loro_20 - std_loro_20,
                smooth_mean_loro_20 + std_loro_20,
                alpha=0.3,
            )
            axs[j, i].plot(
                smooth_mean_loro_30,
                label=f"LORO 30 pretrain eps. Cum. reward={int(sum(mean_loro_30))}",
            )
            axs[j, i].fill_between(
                cache["x"],
                smooth_mean_loro_30 - std_loro_30,
                smooth_mean_loro_30 + std_loro_30,
                alpha=0.3,
            )

            axs[j, i].set_title(
                f"Smoothed pretrain {pretrain_step} steps, {model_size}B"
            )
            axs[j, i].set_xlabel("# of episodes")
            axs[j, i].set_ylabel("Episode reward")
            if "Cliff" in hyperparams["env"]:
                axs[j, i].set_ylim(-500, 0)
            if "MountainCar" in hyperparams["env"]:
                axs[j, i].set_ylim(-210, -50)

    if "MountainCar" in hyperparams["env"]:
        axs[0, 0].legend(loc="upper right")
    else:
        axs[0, 0].legend(loc="lower right")
    plt.savefig(
        f'figs/{hyperparams["env"].split("-")[0]}_pretrain_eps.pdf',
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.show()


def plot_pretrain_big(hyperparams, cache):
    plt.rcParams["font.size"] = 12
    fig, axs = plt.subplots(4, 3, figsize=(16, 20))
    for j in range(4):
        if j < 2:
            model_size = 7
        else:
            model_size = 32
        if j % 2 == 0:
            pretrain_step = 1000
        else:
            pretrain_step = 3000
        for i in range(3):
            mean_loro = cache[f"mean_pretrain_{model_size}b_{pretrain_step}_pre_{i+1}0"]
            std_loro = cache[
                f"std_pretrain_{model_size}b_{pretrain_step}_pre_{i+1}0"
            ] / np.sqrt(hyperparams["n_exp"])
            mean_on_pol = cache[f"mean_on_pol_pretrain_{pretrain_step}_pre_{i+1}0"]
            std_on_pol = cache[
                f"std_on_pol_pretrain_{pretrain_step}_pre_{i+1}0"
            ] / np.sqrt(hyperparams["n_exp"])
            mean_rand = cache[f"mean_rand_pretrain_{pretrain_step}_pre_{i+1}0"]
            std_rand = cache[f"std_rand_pretrain_{pretrain_step}_pre_{i+1}0"] / np.sqrt(
                hyperparams["n_exp"]
            )
            axs[j, i].plot(
                cache["mean_onl"],
                label=f'On-policy. Cum. reward={int(sum(cache["mean_onl"]))}',
            )
            axs[j, i].fill_between(
                cache["x"],
                cache["mean_onl"] - cache["std_onl"] / np.sqrt(hyperparams["n_exp"]),
                cache["mean_onl"] + cache["std_onl"] / np.sqrt(hyperparams["n_exp"]),
                alpha=0.3,
            )
            axs[j, i].plot(mean_loro, label=f"LORO. Cum. reward={int(sum(mean_loro))}")
            axs[j, i].fill_between(
                cache["x"], mean_loro - std_loro, mean_loro + std_loro, alpha=0.3
            )
            axs[j, i].plot(
                mean_on_pol,
                label=f"Pretrain w/ On-policy data. Cum. reward={int(sum(mean_on_pol))}",
            )
            axs[j, i].fill_between(
                cache["x"],
                mean_on_pol - std_on_pol,
                mean_on_pol + std_on_pol,
                alpha=0.3,
            )
            axs[j, i].plot(
                mean_rand,
                label=f"Pretrain w/ random policy data. Cum. reward={int(sum(mean_rand))}",
            )
            axs[j, i].fill_between(
                cache["x"], mean_rand - std_rand, mean_rand + std_rand, alpha=0.3
            )

            axs[j, i].set_title(
                f"Smoothed pretrain {i+1}0 eps, {model_size}B, {pretrain_step} steps"
            )
            axs[j, i].set_xlabel("# of episodes")
            axs[j, i].set_ylabel("Episode reward")
            if "Cliff" in hyperparams["env"]:
                axs[j, i].set_ylim(-500, 0)
            if "MountainCar" in hyperparams["env"]:
                axs[j, i].set_ylim(-210, -50)

    if "MountainCar" in hyperparams["env"]:
        axs[0, 0].legend(loc="upper right")
    else:
        axs[0, 0].legend(loc="lower right")
    plt.savefig(
        f'figs/{hyperparams["env"].split("-")[0]}_pretrain_big.pdf',
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.show()


def plot_sft_lcot_big(hyperparams, cache):
    plt.rcParams["font.size"] = 12
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    for j in range(2):
        if j == 0:
            pretrain_step = 1000
        else:
            pretrain_step = 3000
        for i in range(3):
            try:
                mean_loro = cache[f"mean_pretrain_7b_{pretrain_step}_pre_{i+1}0"]
                std_loro = cache[
                    f"std_pretrain_7b_{pretrain_step}_pre_{i+1}0"
                ] / np.sqrt(hyperparams["n_exp"])
                mean_sft = cache[f"mean_pretrain_7b_{pretrain_step}_pre_{i+1}0SFT"]
                std_sft = cache[
                    f"mean_pretrain_7b_{pretrain_step}_pre_{i+1}0SFT"
                ] / np.sqrt(hyperparams["n_exp"])

                rbf = Rbf(
                    cache["x"],
                    cache["mean_onl"],
                    function="multiquadric",
                    smooth=hyperparams["smooth"],
                )
                smooth_mean_onl = rbf(cache["x"])
                rbf = Rbf(
                    cache["x"],
                    mean_loro,
                    function="multiquadric",
                    smooth=hyperparams["smooth"],
                )
                smooth_mean_loro = rbf(cache["x"])
                rbf = Rbf(
                    cache["x"],
                    mean_sft,
                    function="multiquadric",
                    smooth=hyperparams["smooth"],
                )
                smooth_mean_sft = rbf(cache["x"])

                axs[j, i].plot(
                    smooth_mean_onl,
                    label=f'On-policy. Cum. reward={int(sum(cache["mean_onl"]))}',
                )
                axs[j, i].fill_between(
                    cache["x"],
                    smooth_mean_onl - cache["std_onl"] / np.sqrt(hyperparams["n_exp"]),
                    smooth_mean_onl + cache["std_onl"] / np.sqrt(hyperparams["n_exp"]),
                    alpha=0.3,
                )
                axs[j, i].plot(
                    smooth_mean_loro, label=f"LORO. Cum. reward={int(sum(mean_loro))}"
                )
                axs[j, i].fill_between(
                    cache["x"],
                    smooth_mean_loro - std_loro,
                    smooth_mean_loro + std_loro,
                    alpha=0.3,
                )
                axs[j, i].plot(
                    smooth_mean_sft, label=f"SFT. Cum. reward={int(sum(mean_sft))}"
                )
                axs[j, i].fill_between(
                    cache["x"],
                    smooth_mean_sft - std_sft,
                    smooth_mean_sft + std_sft,
                    alpha=0.3,
                )

                axs[j, i].set_title(
                    f"Smoothed pretrain {i+1}0 eps, 7B, {pretrain_step} steps"
                )
                axs[j, i].set_xlabel("# of episodes")
                axs[j, i].set_ylabel("Episode reward")
                if "Cliff" in hyperparams["env"]:
                    axs[j, i].set_ylim(-500, 0)
                if "MountainCar" in hyperparams["env"]:
                    axs[j, i].set_ylim(-210, -50)
            except Exception as e:
                print(e)
            try:
                mean_ds7 = cache[f"mean_pretrain_7b_{pretrain_step}_pre_{i+1}0DS"]
                std_ds7 = cache[
                    f"mean_pretrain_7b_{pretrain_step}_pre_{i+1}0DS"
                ] / np.sqrt(hyperparams["n_exp"])
                mean_ds14 = cache[f"mean_pretrain_14b_{pretrain_step}_pre_{i+1}0DS"]
                std_ds14 = cache[
                    f"mean_pretrain_14b_{pretrain_step}_pre_{i+1}0DS"
                ] / np.sqrt(hyperparams["n_exp"])
                rbf = Rbf(
                    cache["x"],
                    mean_ds7,
                    function="multiquadric",
                    smooth=hyperparams["smooth"],
                )
                smooth_mean_ds7 = rbf(cache["x"])
                rbf = Rbf(
                    cache["x"],
                    mean_ds14,
                    function="multiquadric",
                    smooth=hyperparams["smooth"],
                )
                smooth_mean_ds14 = rbf(cache["x"])
                axs[j, i].plot(
                    smooth_mean_ds7,
                    label=f"Long-CoT 7B. Cum. reward={int(sum(mean_ds7))}",
                )
                axs[j, i].fill_between(
                    cache["x"],
                    smooth_mean_ds7 - std_ds7,
                    smooth_mean_ds7 + std_ds7,
                    alpha=0.3,
                )
                axs[j, i].plot(
                    smooth_mean_ds14,
                    label=f"Long-CoT 14B. Cum. reward={int(sum(mean_ds14))}",
                )
                axs[j, i].fill_between(
                    cache["x"],
                    smooth_mean_ds14 - std_ds14,
                    smooth_mean_ds14 + std_ds14,
                    alpha=0.3,
                )
            except:
                pass

    if "MountainCar" in hyperparams["env"]:
        axs[0, 0].legend(loc="upper right")
    else:
        axs[0, 0].legend(loc="lower right")
    plt.savefig(
        f'figs/{hyperparams["env"].split("-")[0]}_sft_lcot_bigs.pdf',
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.show()

    # hyperparams["SFT"] = False
    # hyperparams["n_episodes"] = 10 #Online training 10 eps
    # hyperparams["max_episode_len"] = 200

class PolicyWrapper:
    """
    Wrapper for the d3rlpy policy to be used in the rollout function
    """
    def __init__(self, policy):
        self.policy = policy

    def act(self, obs):
        return self.policy.predict(obs)

    def add_env_hist(self, obs, reward, action):
        pass

    def assign_reward(self, reward):
        pass


def dataset_to_coverage_dict(dataset, n_pretrain_eps, merge_time_step=False):
    coverage_dict = {}
    observations_list = []
    actions_list = []
    for i in range(n_pretrain_eps):
        observations = dataset.episodes[i].observations
        actions = np.squeeze(dataset.episodes[i].actions)
        observations_list+=observations.tolist()
        actions_list+=actions.tolist()
        for h in range(len(observations)):
            if merge_time_step:
                time_step = 0
            else:
                time_step = h # h is the time step
            if time_step not in coverage_dict.keys():
                coverage_dict[time_step] = {}
            # observations[time_step] is the one-hot encoding of the state (single number represents the location). Only for FrozenLake-v1 and CliffWalking-v0
            if isinstance(observations[h], np.int64): #LLM collected datasets store observation as integer. Should standardize
                state_action_key = (observations[h], actions[h])
            elif isinstance(observations[h], np.ndarray) and len(observations[h]) == 16: #FrozenLake-v1 from final eval dataset. Obs is one-hot encoding.
                state_action_key = (np.argmax(observations[h].flatten()), actions[h])
            elif isinstance(observations[h], np.ndarray) and len(observations[h]) == 48: #CliffWalking-v0 using one-hot encoding for both pretrain and final eval datasets
                state_action_key = (np.argmax(observations[h].flatten()), actions[h])
            else:
                print(f"observations[h]: {observations[h]}, type: {type(observations[h])}")
                assert False, "Not FrozenLake-v1 or CliffWalking-v0"
            if state_action_key not in coverage_dict[time_step].keys():
                coverage_dict[time_step][state_action_key] = 0
            coverage_dict[time_step][state_action_key] += 1
    return coverage_dict

def get_prob_dict(coverage_dict):
    """
    Calculate the upper bound of the transfer coefficient for each state-action pair.
    
    Args:
        coverage_dict: Dictionary with time step 'h' as keys and coverage_dict[i] as values
    """
    h_list = list(coverage_dict.keys())
    h_list.sort()
    out_dict = {}
    for h in h_list:
        total_visits = sum(coverage_dict[h].values())
        for key, value in coverage_dict[h].items():
            # key is now (obs_tuple, action_tuple)
            out_dict[h, key[0], key[1]] = value/ total_visits
    return out_dict

def get_transfer_coef_upper_bound(prob_dict_online, prob_dict_offline):
    max_transfer_coef = 0
    max_key = None
    # Check if the key is in the online prob_dict exists in the offline prob_dict
    set_online_keys = set(prob_dict_online.keys())
    set_offline_keys = set(prob_dict_offline.keys())
    difference = set_online_keys - set_offline_keys
    difference2 = set_offline_keys - set_online_keys
    # print(f"prob_dict_offline: {prob_dict_offline}")
    # print(f"set_online_keys: {set_online_keys}")
    # print(f"set_offline_keys: {set_offline_keys}")
    # print(f"difference: {difference}")
    # print(f"Number of keys in online but not in offline: {len(difference)}. No of online keys: {len(set_online_keys)}. No of offline keys: {len(set_offline_keys)}")
    if len(difference) > 0:
        # print(f"The following keys are in the online prob_dict but not in the offline prob_dict: {difference}")
        return np.inf, None, (len(difference), len(set_online_keys), len(set_offline_keys), len(difference2))

    for key, value in prob_dict_offline.items():
        if key not in prob_dict_online.keys():
            continue # transfer_coef is 0
        if prob_dict_online[key]/value > max_transfer_coef:
            max_transfer_coef = prob_dict_online[key]/value
            max_key = key
    return max_transfer_coef, max_key, (len(difference), len(set_online_keys), len(set_offline_keys), len(difference2))

def datasets_to_transfer_coef_upper_bound(dataset_online, dataset_offline, n_pretrain_eps, n_online_eval_eps=10, merge_time_step=False):
    """
    n_online_eval_eps: number of episodes to evaluate the online policy (after finished training)
    """
    online_coverage_dict = dataset_to_coverage_dict(dataset_online, n_online_eval_eps, merge_time_step)
    offline_coverage_dict = dataset_to_coverage_dict(dataset_offline, n_pretrain_eps, merge_time_step)
    prob_dict_online = get_prob_dict(online_coverage_dict)
    prob_dict_offline = get_prob_dict(offline_coverage_dict)
    # print(f"offline_coverage_dict: {offline_coverage_dict}")
    # print(f"online_coverage_dict: {online_coverage_dict}")
    return get_transfer_coef_upper_bound(prob_dict_online, prob_dict_offline)

def plot_coverage_heatmap(dataset, h=None, model_name="Qwen 32B", figsize=(10, 8), cmap='Blues', annot=True, fmt='.4f'):
    """
    Plot a heatmap of state-action coverage distribution.
    
    Args:
        h: time step. If None, uniform mixture of all time steps will be plotted
        dataset: dataset to plot the coverage heatmap
        model_name: Name of the model for the plot title
        figsize: Figure size as tuple (width, height)
        cmap: Colormap for the heatmap
        annot: Whether to show annotations on the heatmap
        fmt: Format string for annotations
    """
    tmp_str = "" if h == None else f" - step {h}"
    #coverage_dict: Dictionary with time step 'h' as keys and coverage_dict[i] as values
    #coverage_dict[i]: Dictionary with (state, action) tuples as keys and visit counts as values 
    coverage_dict = dataset_to_coverage_dict(dataset, len(dataset.episodes), merge_time_step=h is None)
    if h is None:
        h=0 #Default index for the mixture of all time steps distribution in coverage_dict
    d = coverage_dict[h]
    # Normalize the coverage dictionary values to get probabilities
    total_visits = sum(d.values())
    normalized_coverage = {key: value / total_visits for key, value in d.items()}
    
    print(f"Total visits: {total_visits}")
    print(f"Normalized coverage (first 5 entries): {dict(list(normalized_coverage.items())[:5])}")
    
    # Extract unique observations and actions
    observations = sorted(list(set([key[0] for key in d.keys()])))
    actions = sorted(list(set([key[1] for key in d.keys()])))
    # print(observations)
    # print(actions)
    
    # Create a matrix for the heatmap
    heatmap_matrix = np.zeros((len(observations), len(actions)))
    
    # Fill the matrix with normalized probabilities
    for (obs, action), prob in normalized_coverage.items():
        try:
            obs_idx = observations.index(obs)
            action_idx = actions.index(action)
            heatmap_matrix[obs_idx, action_idx] = prob
        except ValueError as e:
            print(f"Warning: Could not find index for state '{obs}' or action '{action}'. Error: {e}")
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(heatmap_matrix,
                xticklabels=actions, 
                yticklabels=observations,
                annot=annot, 
                fmt=fmt,
                cmap=cmap,
                cbar_kws={'label': 'Probability'})
    
    plt.title(f'State-Action Coverage Distribution ({model_name}{tmp_str})')
    plt.xlabel('Action')
    plt.ylabel('State/Observation')
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Number of unique state-action pairs: {len(d)}")
    print(f"Number of unique states: {len(observations)}")
    print(f"Number of unique actions: {len(actions)}")
    
    if normalized_coverage:
        print(f"Maximum probability: {max(normalized_coverage.values()):.4f}")
        print(f"Minimum probability: {min(normalized_coverage.values()):.4f}")
    
    print("--------------------------------")
    
    return normalized_coverage, heatmap_matrix, observations, actions

def plot_observation_traces(dataset, width, height, episode_indices=None, figsize=(12, 10), 
                           line_colors=None, line_styles=None, alpha=0.7, 
                           show_grid=True, show_episode_numbers=True, title="Observation Traces",
                           linewidth=2, noise_std=0.1, fig=None, ax=None, name="", save_path=None):
    """
    Visualize the trace of observations from episodes as a grid with connected lines.
    
    Args:
        dataset: Dataset containing episodes with observations
        width: Width of the grid
        height: Height of the grid
        episode_indices: List of episode indices to plot. If None, plots all episodes
        figsize: Figure size as tuple (width, height)
        line_colors: List of colors for each episode line. If None, uses default colors
        line_styles: List of line styles for each episode. If None, uses solid lines
        alpha: Transparency of the lines
        show_grid: Whether to show the grid lines
        show_episode_numbers: Whether to show episode numbers in legend
        title: Title for the plot
        linewidth: Thickness of the trace lines
        noise_std: Standard deviation of noise to add to coordinates (0 for no noise)
        fig: Existing matplotlib figure object. If None, creates a new figure
        ax: Existing matplotlib axes object. If None, creates new axes
    
    Returns:
        fig: matplotlib figure object
        ax: matplotlib axes object
    """
    
    # Determine which episodes to plot
    if episode_indices is None:
        episode_indices = list(range(len(dataset.episodes)))
    
    # Set up colors and line styles
    if line_colors is None:
        # Use a color palette that works well for multiple episodes
        colors = plt.cm.Set3(np.linspace(0, 1, len(episode_indices)))
    else:
        colors = line_colors
    
    if line_styles is None:
        line_styles = ['-'] * len(episode_indices)
    
    # Create figure and axes if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        # Set up the grid only for new figures
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
    else:
        # Use existing figure and axes
        # Check if the axes limits need to be updated
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        new_xlim = (min(current_xlim[0], 0), max(current_xlim[1], width))
        new_ylim = (min(current_ylim[0], 0), max(current_ylim[1], height))
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
    
    # Draw grid if requested
    if show_grid:
        for i in range(width + 1):
            ax.axvline(x=i, color='gray', alpha=0.3, linewidth=0.5)
        for i in range(height + 1):
            ax.axhline(y=i, color='gray', alpha=0.3, linewidth=0.5)
    
    # Plot each episode trace
    for idx, episode_idx in enumerate(episode_indices):
        if episode_idx >= len(dataset.episodes):
            print(f"Warning: Episode index {episode_idx} is out of range. Skipping.")
            continue
            
        observations = dataset.episodes[episode_idx].observations
        
        # Convert observations to grid coordinates
        x_coords = []
        y_coords = []
        
        # Generate noise for this episode (same noise for all points in the episode)
        if noise_std > 0:
            episode_noise_x = np.random.normal(0, noise_std)
            episode_noise_y = np.random.normal(0, noise_std)
        else:
            episode_noise_x = 0
            episode_noise_y = 0
        
        for obs in observations:
            # Handle different observation formats
            if isinstance(obs, np.int64):
                # Integer observation - convert to grid coordinates
                grid_pos = obs
                x = grid_pos % width
                y = height - 1 - (grid_pos // width)  # Flip y-axis to match typical grid layout
            elif isinstance(obs, np.ndarray):
                # One-hot encoded observation - find the position
                if len(obs) == width * height:  # Direct one-hot for grid
                    grid_pos = np.argmax(obs)
                    x = grid_pos % width
                    y = height - 1 - (grid_pos // width)
                else:
                    # For other one-hot encodings, assume it represents a discrete state
                    grid_pos = np.argmax(obs)
                    x = grid_pos % width
                    y = height - 1 - (grid_pos // width)
            else:
                print(f"Warning: Unknown observation format: {type(obs)}. Skipping.")
                continue
            
            # Convert to cell center coordinates and add noise
            x_center = x + 0.5 + episode_noise_x
            y_center = y + 0.5 + episode_noise_y
            
            x_coords.append(x_center)
            y_coords.append(y_center)
        
        if len(x_coords) > 0:
            # Plot the trace line
            label = f"Episode {episode_idx}" if show_episode_numbers else f"Trace {idx}"
            ax.plot(x_coords, y_coords, 
                   color=colors[idx % len(colors)], 
                   linestyle=line_styles[idx % len(line_styles)],
                   alpha=alpha, 
                   linewidth=linewidth, 
                #    label=label,
                #    marker='o',
                   markersize=4,
                   markeredgecolor='black',
                   markeredgewidth=0.5)
            
            # Mark start and end points
            if len(x_coords) > 0:
                # Start point (green)
                ax.plot(x_coords[0], y_coords[0], 'o', 
                       color=colors[0], markersize=8, label=f'{name} Start' if idx == 0 else "")
                # End point (red)
                ax.plot(x_coords[-1], y_coords[-1], '+', 
                       color=colors[0], markersize=18, label=f'{name} End' if idx == 0 else "")
    
    # Customize the plot
    if fig is None or ax is None:
        ax.set_xlabel('Grid X Position')
        ax.set_ylabel('Grid Y Position')
    
    # Always set the title
    ax.set_title(title)
    
    # Set tick labels to show grid positions (always update for proper grid display)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.set_xticklabels(range(width))
    ax.set_yticklabels(range(height))
    
    # Disable the default grid to avoid center values
    ax.grid(False)
    
    # Add legend (only for new figures)
    # if len(episode_indices) > 1 and (fig is None or ax is None):
    #     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Apply tight_layout only for new figures
    # if fig is None or ax is None:
    #     plt.tight_layout()
    if save_path is not None:
        plt.savefig(
            save_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )
    
    return fig, ax