# image-superres
Implementation of image super-resolution using generative adversarial network (GAN)
The paper from which the model has been taken for implementation is available here : https://arxiv.org/pdf/1609.04802.pdf
The model consists of two networks: generator and discriminator. The generator is used to obtain the high resolution image from the low resolution image and discriminator is used to differentiate between original higher resolution image and generator super resolved image. Discriminator is used in training to train the generator to generate super-resolved images closer to the original high resolution image. Simultaneously, discriminator is also trained to better differentiate between true high resolution and generator super-resolved images.
## Generator network
![gen_net](https://user-images.githubusercontent.com/24764839/127691842-b587444e-3cae-4ceb-9d56-7730a47dd502.JPG)
## Discriminator network
![disc_net](https://user-images.githubusercontent.com/24764839/127691899-6ada2363-a44f-4aba-99d6-9d9b6ecf0ff6.JPG)
