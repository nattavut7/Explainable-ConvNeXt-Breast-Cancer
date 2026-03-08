
import torch

def tta_predict(model,image):

    model.eval()

    transforms = [
        lambda x: x,
        lambda x: torch.flip(x,[3]),
        lambda x: torch.flip(x,[2]),
        lambda x: torch.rot90(x,1,[2,3]),
        lambda x: torch.rot90(x,2,[2,3])
    ]

    predictions = []

    for t in transforms:

        aug = t(image)

        with torch.no_grad():

            pred = torch.softmax(model(aug),dim=1)

        predictions.append(pred)

    final_pred = torch.mean(torch.stack(predictions),0)

    return final_pred
