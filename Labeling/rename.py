import os

all_images = os.listdir('./training_images')

for i, image in enumerate(all_images):
    os.rename(f'./training_images/{image}', f'./training_images/image_{i}.jpg')