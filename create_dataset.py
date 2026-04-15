#script per creare il dataset per naeem
import os
import pandas as pd
import argparse
from sampling import sample_from_mask, load_trained_model
import torch
import yaml

if __name__ == "__main__":

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    dirs = config["directories"]
    samp = config["sampling"]
    params = config["parameters"] 
    dataset = config["dataset"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_list', default=dataset["dataset_list"], help='csv file to write generated dataset information')
    parser.add_argument('--info_masks', default=dataset["info_masks"], help='csv file with mask subject information')
    parser.add_argument('--masks', default=dataset["masks"], help='masks directory')
    parser.add_argument('--data_len', type=int, default=dataset["data_len"])
    parser.add_argument('--seed', type=int, default=dataset["seed"], help='initial seed value')
    parser.add_argument('--sample_dir', default=dataset["sample_dir"], help='directory to save generated dataset')
    parser.add_argument("--checkpoint", type=int, default=samp["checkpoint"], help="Checkpoint step to load the model from")
    parser.add_argument("--checkpoints_dir", type=str, default=dirs["checkpoints_dir"], help="Directory of checkpoints")
    parser.add_argument("--input_size", type=int, default=params["input_size"], help="Input size for the model")
    parser.add_argument("--num_channels", type=int, default=params["num_channels"], help="Number of channels in the model")
    parser.add_argument("--num_res_blocks", type=int, default=params["num_res_blocks"], help="Number of residual blocks in the model")
    parser.add_argument("--in_channels", type=int, default=params["in_channels"], help="Number of input channels (image + mask)")
    parser.add_argument("--out_channels", type=int, default=params["out_channels"], help="Number of output channels (velocity field)")
    parser.add_argument("--num_classes", type=int, default=params["num_classes"], help="Number of classes (CN, MCI, AD)")
    parser.add_argument("--ema", type=bool, default=samp["ema"], help="Whether to use EMA weights for sampling")
    args = parser.parse_args()

    label_to_idx = {
        "CN": 0,
        "MCI": 1,
        "AD": 2
    }

    seed = args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset directory
    os.makedirs(args.sample_dir, exist_ok=True)

    # from masks subject info
    df = pd.read_csv(args.info_masks)
    len_subj = len(df['Subject'])

    # load the trained model for sampling
    model = load_trained_model(args.checkpoints_dir, args.checkpoint, args.input_size, args.num_channels, args.num_res_blocks, args.in_channels, args.out_channels, args.num_classes, args.ema, device)

    with open(args.dataset_list, 'w') as f:
        # header
        f.write("Subject,Diagnosis,Seed\n")
        # loop on diagnosis and subjects to write the csv file
        for diagnosis in ['CN', 'MCI', 'AD']:
            count = 0 # reset counter to loop on subjects for diagnosis
            
            #select subjects by diagnosis
            df_diag = df[df['Group'] == diagnosis] # or df[df['Diagnosis'] == index_to_label[diagnosis]]
            len_diag = len(df_diag)
             
            for i in range(args.data_len):
                round = count // len_diag # to loop on subjects
                idx = count - len_diag * round # index to select the subject for the current diagnosis

                # write the subject, diagnosis and seed to the csv file
                subject = df_diag['Subject'].iloc[idx]
                f.write(f"{subject},{diagnosis},{seed}\n")

                # sampling
                target_label = label_to_idx[diagnosis]
                mask_path = os.path.join(args.masks, f"{subject}_mask.nii.gz")
                sample_from_mask(model, mask_path, num_samples=1, sample_dir=args.sample_dir, target_label=target_label, seed=seed, device=device)

                # increment of 1 to change the seed for every sample!!
                seed += 1
                count += 1