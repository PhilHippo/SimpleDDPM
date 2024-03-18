from torchsummary import summary
import torch
from Diffusion.Model import UNet

modelConfig = {
    "state": "eval",  # train or eval
    "epoch": 200,
    "batch_size": 40,
    "T": 1000,
    "channel": 128,
    "channel_mult": [1, 2, 3, 4],
    "attn": [2],
    "num_res_blocks": 2,
    "dropout": 0.15,
    "lr": 1e-4,
    "multiplier": 2.0,
    "beta_1": 1e-4,
    "beta_T": 0.02,
    "img_size": 32,
    "grad_clip": 1.0,
    "device": "cuda:0",  ### MAKE SURE YOU HAVE A GPU !!!
    "training_load_weight": None,
    "save_weight_dir": "./Checkpoints/",
    "test_load_weight": "ckpt_199_.pt",
    "sampled_dir": "./SampledImgs/noguidance/",
    "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
    "sampledImgName": "Sampled.png",
    "nrow": 8,
}

device = torch.device(modelConfig["device"])

model = net_model = UNet(
    T=modelConfig["T"],
    ch=modelConfig["channel"],
    ch_mult=modelConfig["channel_mult"],
    attn=modelConfig["attn"],
    num_res_blocks=modelConfig["num_res_blocks"],
    dropout=modelConfig["dropout"],
).to(device)

summary(model, [(3, 32, 32), (1000,)])
