import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

"""Adapted from https://github.com/mims-harvard/Raindrop"""


def build_dataloaders(
    data_root="data/",
    dataset_name="P12",
    split=1,
    batch_size=32,
   
    dev=False
):

    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(
        data_root, dataset_name, split
    )
    if dev:
        Ptrain, Pval, Ptest, ytrain, yval, ytest = (
            Ptrain[:100],
            Pval[:100],
            Ptest[:100],
            ytrain[:100],
            yval[:100],
            ytest[:100],
        )
    logger.info(
        f"Train data: {len(Ptrain)} Val data: {len(Pval)} Test data: {len(Ptest)}"
    )

    (
        train_times,
        train_deltas,
        train_values,
        train_indicators,
        train_static,
        train_labels,
        val_times,
        val_deltas,
        val_values,
        val_indicators,
        val_static,
        val_labels,
        test_times,
        test_deltas,
        test_values,
        test_indicators,
        test_static,
        test_labels,
    ) = process_data(Ptrain, Pval, Ptest, ytrain, yval, ytest, dataset_name)

    train_set = TensorDataset(
        train_times, train_deltas, train_values, train_indicators, train_labels
    )
    val_set = TensorDataset(
        val_times, val_deltas, val_values, val_indicators, val_labels
    )
    test_set = TensorDataset(
        test_times, test_deltas, test_values, test_indicators, test_labels
    )

    collate_fn = Collator()

    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=2,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader, test_dataloader


class Collator:

    def __call__(self, batch):
        data_dict = self.collate_fn(batch)
        return data_dict

    def collate_fn(self, batch):
        n_channels = batch[0][1].shape[1]
        # truncation
        max_len = max([torch.argmax(sample[0], 0) for sample in batch]) + 1

        times = torch.zeros([len(batch), max_len])
        values = torch.zeros([len(batch), max_len, n_channels])
        indicators = torch.zeros([len(batch), max_len, n_channels])
        deltas = torch.zeros([len(batch), max_len, n_channels])
        labels = torch.zeros([len(batch)])
        for b_idx, (time, delta, value, indicator, label) in enumerate(batch):
            times[b_idx] = time[:max_len]
            deltas[b_idx] = delta[:max_len]
            values[b_idx] = value[:max_len]
            indicators[b_idx] = indicator[:max_len]
            labels[b_idx] = label

        data_dict = {
            "time": times,
            "delta": deltas,
            "value": values,
            "indicator": indicators,
            "label": labels.long(),
        }
        return data_dict


def get_data_split(data_root, dataset_name, split):
    Pdict_list = np.load(
        data_root + dataset_name + "/processed_data/PTdict_list.npy",
        allow_pickle=True,
    )
    arr_outcomes = np.load(
        data_root + dataset_name + "/processed_data/arr_outcomes.npy",
        allow_pickle=True,
    )

    idx_train, idx_val, idx_test = np.load(
        data_root + dataset_name + "/splits/split" + str(split) + ".npy",
        allow_pickle=True,
    )

    Ptrain, Pval, Ptest = (
        Pdict_list[idx_train],
        Pdict_list[idx_val],
        Pdict_list[idx_test],
    )

    y = arr_outcomes[:, -1].reshape((-1, 1))
    ytrain, yval, ytest = y[idx_train], y[idx_val], y[idx_test]

    return Ptrain, Pval, Ptest, ytrain, yval, ytest


def process_data(
    train_data, val_data, test_data, ytrain, yval, ytest, dataset_name="P12"
):
    if dataset_name in ["P12", "P19"]:
        length, n_channels = train_data[0]["arr"].shape
        static_dim = len(train_data[0]["extended_static"])

        times = np.zeros((len(train_data), length, 1))
        values = np.zeros((len(train_data), length, n_channels))
        statics = np.zeros((len(train_data), static_dim))

        for i in range(len(train_data)):
            times[i] = train_data[i]["time"] / 60
            values[i] = train_data[i]["arr"]
            statics[i] = train_data[i]["extended_static"]
        time_max = np.max(times)
        data_mean, data_std = getStats(values)
        static_mean, static_std = getStats_static(statics, dataset_name=dataset_name)
        # np.save('data_mean.npy', data_mean)
        # np.save('data_std.npy', data_std)
        # print(time_max)
        (
            train_times,
            train_delta,
            train_values,
            train_indicator,
            train_static,
            train_label,
        ) = tensorize_normalize(
            train_data, ytrain, data_mean, data_std, static_mean, static_std, time_max
        )
        (val_times, val_delta, val_values, val_indicator, val_static, val_label) = (
            tensorize_normalize(
                val_data, yval, data_mean, data_std, static_mean, static_std, time_max
            )
        )
        (
            test_times,
            test_delta,
            test_values,
            test_indicator,
            test_static,
            test_label,
        ) = tensorize_normalize(
            test_data, ytest, data_mean, data_std, static_mean, static_std, time_max
        )
    elif dataset_name == "PAM":
        length, n_channels = train_data[0].shape

        values = train_data
        data_mean, data_std = getStats(values)

        (
            train_times,
            train_delta,
            train_values,
            train_indicator,
            train_static,
            train_label,
        ) = tensorize_normalize_other(train_data, ytrain, data_mean, data_std)
        (val_times, val_delta, val_values, val_indicator, val_static, val_label) = (
            tensorize_normalize_other(val_data, yval, data_mean, data_std)
        )
        (
            test_times,
            test_delta,
            test_values,
            test_indicator,
            test_static,
            test_label,
        ) = tensorize_normalize_other(test_data, ytest, data_mean, data_std)

    return (
        train_times,
        train_delta,
        train_values,
        train_indicator,
        train_static,
        train_label,
        val_times,
        val_delta,
        val_values,
        val_indicator,
        val_static,
        val_label,
        test_times,
        test_delta,
        test_values,
        test_indicator,
        test_static,
        test_label,
    )


def getStats(P_tensor):
    N, length, n_channels = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(n_channels, -1)
    mf = np.zeros((n_channels, 1))
    stdf = np.ones((n_channels, 1))
    eps = 1e-7
    for f in range(n_channels):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        mf[f] = np.mean(vals_f)
        stdf[f] = np.std(vals_f)
        stdf[f] = np.max([stdf[f].item(), eps])
    return mf, stdf


def getStats_static(P_tensor, dataset_name="P12"):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))
    ms = np.zeros((S, 1))
    ss = np.ones((S, 1))

    if dataset_name == "P12":
        # ['Age' 'Gender=0' 'Gender=1' 'Height' 'ICUType=1' 'ICUType=2' 'ICUType=3' 'ICUType=4' 'Weight']
        bool_categorical = [0, 1, 1, 0, 1, 1, 1, 1, 0]
    elif dataset_name == "P19":
        # ['Age' 'Gender' 'Unit1' 'Unit2' 'HospAdmTime' 'ICULOS']
        bool_categorical = [0, 1, 0, 0, 0, 0]
    elif dataset_name == "eICU":
        # ['apacheadmissiondx' 'ethnicity' 'gender' 'admissionheight' 'admissionweight'] -> 399 dimensions
        bool_categorical = [1] * 397 + [0] * 2

    for s in range(S):
        if bool_categorical == 0:  # if not categorical
            vals_s = Ps[s, :]
            vals_s = vals_s[vals_s > 0]
            ms[s] = np.mean(vals_s)
            ss[s] = np.std(vals_s)
    return ms, ss


def tensorize_normalize(
    data, label, data_mean, data_std, static_mean, static_std, time_max
):
    T, F = data[0]["arr"].shape
    D = len(data[0]["extended_static"])

    values = np.zeros((len(data), T, F))
    times = np.zeros((len(data), T, 1))
    static = np.zeros((len(data), D))
    for i in range(len(data)):
        times[i] = data[i]["time"] / 60
        values[i] = data[i]["arr"]
        static[i] = data[i]["extended_static"]

    values, indicator = mask_normalize(values, data_mean, data_std)

    times = torch.tensor(times / time_max).squeeze(-1)

    delta = calculate_delta(times.cpu().numpy(), indicator.cpu().numpy())

    static = mask_normalize_static(static, static_mean, static_std)
    static = torch.tensor(static)

    label = torch.tensor(label[:, 0], dtype=torch.long)
    return times, delta, values, indicator, static, label


def tensorize_normalize_other(data, label, data_mean, data_std):
    T, F = data[0].shape
    times = np.zeros((len(data), T, 1))
    for i in range(len(data)):
        time = torch.linspace(0, T, T).reshape(-1, 1)
        times[i] = time
    values, indicator = mask_normalize(data, data_mean, data_std)

    times = (torch.tensor(times) / 60.0).squeeze(-1)
    times = times / torch.max(times)

    delta = calculate_delta(times.cpu().numpy(), indicator.cpu().numpy())

    static = None

    label = torch.tensor(label[:, 0]).type(torch.long)
    return times, delta, values, indicator, static, label


def mask_normalize(values, mf, stdf):
    """Normalize time series variables. Missing ones are set to zero after normalization."""
    N, T, F = values.shape
    Pf = values.transpose((2, 0, 1)).reshape(F, -1)
    M = 1 * (values > 0) + 0 * (values <= 0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    return torch.tensor(Pnorm_tensor), torch.tensor(M)


def mask_normalize_static(P_tensor, ms, ss):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))

    # input normalization
    for s in range(S):
        Ps[s] = (Ps[s] - ms[s]) / (ss[s] + 1e-18)

    # set missing values to zero after normalization
    for s in range(S):
        idx_missing = np.where(Ps[s, :] <= 0)
        Ps[s, idx_missing] = 0

    # reshape back
    Pnorm_tensor = Ps.reshape((S, N)).transpose((1, 0))
    return Pnorm_tensor


def calculate_delta(observed_tp, observed_mask):
    # observed_tp:[B,L]  observed_mask:[B,L,K],
    # return [B,L,K]
    if observed_tp.ndim == 2:
        tmp_time = observed_mask * np.expand_dims(observed_tp, axis=-1)  # [B,L,K]
    else:
        tmp_time = observed_tp.copy()

    b, l, k = tmp_time.shape

    new_mask = observed_mask.copy()
    new_mask[:, 0, :] = 1
    tmp_time[new_mask == 0] = np.nan
    tmp_time = tmp_time.transpose((1, 0, 2))  # [L,B,K]
    tmp_time = np.reshape(tmp_time, (l, b * k))  # [L, B*K]

    # padding the missing value with the next value
    df1 = pd.DataFrame(tmp_time)
    df1 = df1.ffill()
    tmp_time = np.array(df1)

    tmp_time = np.reshape(tmp_time, (l, b, k))
    tmp_time = tmp_time.transpose((1, 0, 2))  # [B,L,K]

    tmp_time[:, 1:] -= tmp_time[:, :-1]
    del new_mask
    delta = torch.tensor(tmp_time * observed_mask)
    return delta


if __name__ == "__main__":
    train_dataloader, val_dataloader, test_dataloader = build_dataloaders(
        batch_size=2, dev=True
    )
    sample = next(iter(train_dataloader))
