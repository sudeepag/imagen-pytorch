import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer

from COCODataset import COCODataset

# unet for imagen

unet1 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
)

unet2 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = (2, 4, 8, 8),
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True)
)

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    unets = unet1,
    text_encoder_name = 't5-base',
    image_sizes = 64,
    timesteps = 1000,
    cond_drop_prob = 0.1
).cuda()

# wrap imagen with the trainer class

trainer = ImagenTrainer(imagen, split_valid_from_train = True).cuda()

# mock images (get a lot of this) and text encodings from large T5


# feed images into imagen, training each unet in the cascade


dataset = COCODataset('../train2014', '../annotations_trainval2014/annotations', image_size = 64)

trainer.add_train_dataset(dataset, batch_size = 16)

# working training loop

for i in range(200000):
    loss = trainer.train_step(unet_number = 1, max_batch_size = 4)

    print(f'loss: {loss}')


    if not (i % 50):
        valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = 4)
        print(f'valid loss: {valid_loss}')

    if not (i % 100) and trainer.is_main: # is_main makes sure this can run in distributed
        images = trainer.sample(texts = [
            'a puppy looking anxiously at a giant donut on the table',
            'the milky way galaxy in the style of monet'
        ], cond_scale = 3., return_pil_images = True)
        images[0].save(f'./sample-{i // 100}.png')
