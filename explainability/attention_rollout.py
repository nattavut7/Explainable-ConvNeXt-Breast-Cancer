
import torch
import numpy as np
import cv2

def attention_rollout(feature_maps):

    attention = torch.mean(feature_maps,dim=1)

    attention = attention.squeeze().cpu().numpy()

    attention = cv2.resize(attention,(224,224))

    attention = (attention - attention.min()) / (attention.max() - attention.min())

    heatmap = cv2.applyColorMap(np.uint8(255*attention),cv2.COLORMAP_JET)

    return heatmap
