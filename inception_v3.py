import torch
import torch.nn as nn
import torchvision.models as models

class InceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3,self).__init__()
        # accomodating for the image dataset
        self.model = models.inception_v3(init_weights=False, num_classes=num_classes, aux_logits=False)
        self.model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
        self.model.fc = nn.Linear(2048, num_classes)
        
    
    def forward(self, x):
        x = x.unsqueeze(1)

        # converting input and weights to float32 type
        x = x.to(torch.float32)
        self.model.Conv2d_1a_3x3.conv.weight = self.model.Conv2d_1a_3x3.conv.weight.to(torch.float32)
        x = self.model(x)
        return x
    
