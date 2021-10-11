from torchvision import models
from torchsummary import summary

vgg = models.vgg16(pretrained=False)
print(vgg)
print(summary(vgg, (3, 224, 224), device="cpu"))
