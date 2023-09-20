import pickle
from fastapi import FastAPI
from time import sleep
import torchvision.models as models
import timm
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorchvideo.data.encoded_video import EncodedVideo
import torch
import numpy as np
from pydantic import BaseModel
class OurModel(LightningModule):
        def __init__(self):
            super(OurModel,self).__init__()
            

        def forward(self,video):
            x=self.video_model(video)
            x=self.relu(x)
            x=self.linear(x)
            return x
import __main__
setattr(__main__, "OurModel", OurModel)

##params
num_video_samples=20
video_duration=2
model_name='efficient_x3d_xs'
batch_size=8
scheduler='cosine'
clipmode='random'
img_size=224
model_path = 'models/efficient_x3d.sav'
app = FastAPI()

from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler, labeled_video_dataset

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
    Permute
)

from torchvision.transforms import (
    Compose,
    # Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize
)

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
video_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(num_video_samples),
                    # Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                      #Determines the shorter spatial dim of the video (i.e. width or height) and scales it to the given size
                    RandomShortSideScale(min_size=img_size+16, max_size=img_size+32),
                    CenterCropVideo(img_size),
                    RandomHorizontalFlip(p=0.5),
                  ]
                ),
              ),
            ]
        )

class video_path(BaseModel):
    video_path:str


        
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

@app.post('/')
async def predict(item:video_path):
    video=EncodedVideo.from_path(item.video_path)
    video_data=video.get_clip(0,2)
    video_data=video_transform(video_data)
    model=loaded_model.cuda()
    inputs=video_data['video'].cuda()
    inputs=torch.unsqueeze(inputs,0)
    preds=model(inputs)
    preds=preds.detach().cpu().numpy()
    preds=np.where(preds>0.5, 1,0)
    if preds[0][0] == 0:
        final_pred = 'normal'
    else:
        final_pred = 'shoplifting'
    return final_pred