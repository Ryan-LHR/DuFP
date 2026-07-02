
import models
import torch
path = '/root/Projects/arachne-master/lhr/lhr_data/models/imagenet100_test/fastvit_t8.pth'
# torch.save(model, path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(path, map_location=torch.device(device))
