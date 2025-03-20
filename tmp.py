path = "local_output2/ar-ckpt-giter015K-ep99-iter52-last.pth"
import torch
model = torch.load(path, map_location="cpu")
sd = model["trainer"]["gpt_fsdp"]

#drop all keys not contains adapter
sd = {k: v for k, v in sd.items() if "adapter" in k}

#save to new path
torch.save(sd, "local_output2/ar-ckpt-giter015K-ep99-iter52-last-adapter.pth")
