import os
import json
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sagemaker_inference import content_types, default_inference_handler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model structure
class VisionT(nn.Module):
    def __init__(self):
        super(VisionT, self).__init__()
        # Replace this with your model architecture
        self.model = resnet50()
        self.model.to(device)

    def forward(self, x):
        return self.model(x)

visiont = VisionT()

# Load the pre-trained model
def model_fn(model_dir):
    visiont.model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location=device))
    visiont.model.eval()
    return visiont

# Deserialize the Invoke request body into an object
def input_fn(request_body, request_content_type):
    if request_content_type == content_types.CONTENT_TYPE_JSON:
        data = json.loads(request_body)
        data = torch.tensor(data, device=device)
        return data
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

# Serialize the prediction result into the desired response content type
def output_fn(prediction, response_content_type):
    if response_content_type == content_types.CONTENT_TYPE_JSON:
        return json.dumps(prediction.tolist())
    else:
        raise ValueError("Unsupported content type: {}".format(response_content_type))

# Perform the prediction and return the result
def predict_fn(input_data, model):
    with torch.no_grad():
        output = model(input_data)
        return output.cpu()

# Define the handler
handler = default_inference_handler(input_fn, predict_fn, output_fn, model_fn=model_fn)

if __name__ == "__main__":
    handler.handle("test")

