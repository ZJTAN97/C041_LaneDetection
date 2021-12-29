from config import config
from torchvision import transforms


transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(
            (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)
        ),
        transforms.ToTensor(),
    ]
)
