import os
import torch
import numpy as np
import pickle
from sklearn.decomposition import PCA
from tabular_model import *
from tabular_util import *

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

_, config = load_config_yaml("config.yaml")
config["device"] = torch.device("cpu")  # ('cuda:'+ config['gpu'])


def infer(model, infer_csv_path, output_name, pca):
    model.eval()

    infer_df = pd.read_csv(infer_csv_path)
    infer_dataset = LongitudinalSingleDataset(infer_df)
    inferDataLoader = DataLoader(
        infer_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    result_dir = f'./results/{config["dataset_name"]}/{config["model_name"]}/{ckpt_label}/'
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, output_name if output_name.endswith(".npy") else output_name + ".npy")

    rid_list, tab_list, lb_list, recon_list, z_list, age_list = [], [], [], [], [], []

    with torch.no_grad():
        for _, sample in enumerate(inferDataLoader, 0):
            tab = sample["tab"].to(config["device"], dtype=torch.float).unsqueeze(1)
            if not torch.isfinite(tab).all():
                nan_indices = torch.nonzero(torch.isnan(tab))
                for i, _, j in nan_indices:
                    tab[i, _, j] = torch.nanmean(tab[:, :, j])
            zero_mx = torch.zeros_like(tab)
            zs, recons = model(tab, zero_mx)

            rid_list.extend(sample["rid"])
            tab_list.append(tab.detach().cpu().numpy())
            recon_list.append(recons[0].detach().cpu().numpy())
            z_list.append(zs[0].detach().cpu().numpy())
            age_list.append(sample["age"].numpy())
            lb_list.append(sample["lb"].detach().cpu().numpy())

    # Concatenate arrays
    tab_list = np.concatenate(tab_list, axis=0)
    recon_list = np.concatenate(recon_list, axis=0)
    z_list = np.concatenate(z_list, axis=0)
    age_list = np.concatenate(age_list, axis=0)
    lb_list = np.concatenate(lb_list, axis=0)

    # PCA processing
    if pca is None:
        pca_model = PCA(n_components=2)
        pcs = pca_model.fit_transform(z_list)
        pickle_path = os.path.join(result_dir, "pca_transformer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(pca_model, f)
        print(f"Fitted new PCA and saved transformer to {pickle_path}")
    else:
        # Load pre-trained transformer
        if isinstance(pca, str):
            with open(pca, "rb") as f:
                pca_model = pickle.load(f)
        else:
            pca_model = pca
        pcs = pca_model.transform(z_list)

    # Save results
    results = {
        "RID": np.array(rid_list),
        "age": age_list,
        "lb": lb_list,
        "tab": tab_list,
        "recon": recon_list,
        "z": z_list,
        "pc": pcs,  # shape (n_samples, 2)
    }
    np.save(path, results, allow_pickle=True)
    print(f"Saved inference results to {path}")


def load_ckpt(ckpt_path):
    flag, config_load = load_config_yaml(os.path.join(ckpt_path, "config.yaml"))
    # load config file
    if flag:
        print("load yaml config file")
        for key in config_load.keys():  # if yaml has, use yaml's param, else use config
            if key == "phase" or key == "gpu" or key == "continue_train" or key == "ckpt_name":
                continue
            if key in config.keys():
                config[key] = config_load[key]
            else:
                print("current config do not have yaml param")
    else:
        save_config_yaml(ckpt_path, config)

    # Load model
    model = LSP(
        num_neighbours=config["num_neighbours"],
        dims=config["dims"],
        agg_method=config["agg_method"],
        gpu=config["device"],
        activation="leakyrelu",
        dropout=0.0,
        slope=0.2,
        batch_norm=True,
    ).to(config["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, min_lr=1e-5)

    [optimizer, scheduler, model], start_epoch = load_checkpoint_by_key(
        [optimizer, scheduler, model],
        ckpt_path,
        ["optimizer", "scheduler", "model"],
        config["device"],
        config["ckpt_name"],
    )
    print(model.parameters())

    return config, model, optimizer, scheduler, start_epoch


######################
# Inference settings #
######################

trial = 2
infer_name = "NACC_single"
ckpt_dir = "./ckpt/ADNI1GO234/LSP/"
ckpt_label = next(folder for folder in os.listdir(ckpt_dir) if folder.startswith(f"trial{trial}_"))
config["ckpt_path"] = os.path.join("./ckpt/", config["dataset_name"], config["model_name"], ckpt_label)

config, model, optimizer, scheduler, start_epoch = load_ckpt(config["ckpt_path"])

infer_csv_path = f"data/ADNI1GO234/splits/trial{trial}/preadj_{infer_name}.csv"

output_name = f"{infer_name}_results.npy"
pca = os.path.join("./results/", config["dataset_name"], config["model_name"], ckpt_label, "pca_transformer.pkl")
print(f"\nInference for {infer_name} data, trial {trial}...")
infer(model, infer_csv_path, output_name, pca)


##################################
# Batch inference for all trials #
##################################
# for trial in range(50):
#     ckpt_dir = './ckpt/ADNI1GO234/LSP/'
#     ckpt_label = next(folder for folder in os.listdir(ckpt_dir) if folder.startswith(f"trial{trial}_"))
#     config["ckpt_path"] = os.path.join("./ckpt/", config["dataset_name"], config["model_name"], ckpt_label)

#     config, model, optimizer, scheduler, start_epoch = load_ckpt(config["ckpt_path"])

#     for phase in ["train", "val", "test", "single"]:
#         infer_csv_path = f"./data/{config['dataset_name']}/splits/trial{trial}/preadj_{phase}.csv"
#         output_name = f"{phase}_results.npy"
#         pca = None
#         if phase in ["val", "test", "single"]:
#             pca = os.path.join('./results/', config['dataset_name'], config['model_name'], ckpt_label, "pca_transformer.pkl")
#         print(f"\nInference for {phase} data, trial {trial}...")
#         infer(model, infer_csv_path, output_name, pca)
