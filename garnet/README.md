- Github: https://github.com/KimRass/garnet

# Paper Summary
- Paper: [The Surprisingly Straightforward Scene Text Removal Method With Gated Attention and
Region of Interest Generation: A Comprehensive Prominent Model Analysis](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760436.pdf)
## Related Works
- To the best of our knowledge, there is no other previous study that considered both TSR and TSSR while performing STR.
- *EnsNet: Simple and fast because it doesn’t need any auxiliary inputs. However, its results are blurry and of low quality. Furthermore, its practical applications are limited because it is impossible to only erase text in a certain region.*
- *MTRNet: Requires the generation of text box region masks through manual or automatic means.*
- MTRNet++ and EraseNet: Proposed a coarse-refinement two-stage network. While the results are of higher quality, the model is much bigger, slower, and too complicated.
- *Zdenek et al.: Used an auxiliary text detector to retrieve the text box mask, then attempted to erase text through a general inpainting method.*
- *Tang et al.: Can train their model in a rather efficient manner because they erase text by cropping only the text regions. However, this method ignores all global contexts other than the cropped region and has difficulty precisely cropping the region of curved texts.*
- T. Nakamura et al. [12] proposed an evaluation method using an auxiliary text detector. An auxiliary text detector obtains detection results on the images with text removed. Then, it evaluates the model performance by calculating Precision, Recall, and F-score values. A lower value means that the texts are better erased.
## Experiment
- We use the same standardized training/testing dataset to evaluate the performance of several previous methods after standardized re-implementation.
## Methodology
- We also introduce a simple yet extremely effective Gated Attention (GA) and Region-of-Interest
Generation (RoIG) methodology in this paper.
- *First, our proposed method takes a text box mask as the input to our model, leading to the option for users to selectively erase only the text that they wish to.* Second, our proposed method can localize the TSR and TSSR to erase text in a surgical manner.
### Gated Attention
- *GA uses attention to focus on the text stroke as well as the textures and colors of the surrounding regions to remove text from the input image much more precisely.*
- GA is a simple yet extremely effective method that uses attention to focus on the text stroke and its surroundings.
- GA not only distinguishes between the background and the text stroke, but also utilizes attention to identify both Text Stroke Region (TSR) and the Text Stroke Surrounding Region (TSSR).
- Previous STR models [18, 17] used a text box region as well as a text stroke region in an attempt to perform precise text removal. However, TSSR was mostly overlooked. After finding inspiration from observing how humans must alternate between paying attention to the text stroke regions and the surrounding regions of the text while manually performing STR, we devised the GA.
### Region-of-Interest Genration
- *RoIG is applied to focus on only the region with text instead of the entire image to train the model more efficiently.*
- *Because all previous studies performed STR on the entire image, artifacts frequently occurred in non-text regions. Thus, we devised the RoIG, which allows our STR model to only generate a result image from within the text box region instead of wasting resources attempting to perform STR on the full image.*
### Architecture
- The generator G takes the image and corresponding text box mask as its input and produces a non-text image that is visually plausible. The discriminator D takes both the input of the generator and the target images as its input and differentiates between real images and images produced by the generator.
#### Generator
- The generator has an FCN-ResNet18 backbone and skip connections between the encoder and decoder. The model is composed of five convolution layers paired with five deconvolution layers with a kernel size of 4, stride of 2, and padding of 1. The convolution pathway is composed of two residual blocks, which contains the proposed Gated Attention (GA) module.
#### Discriminator
- For training, we use a local-aware discriminator proposed in EnsNet, which only penalizes the erased text patches. The discriminator is trained with locality-dependent labels, which indicate text stroke regions in the output tensor. It guides the discriminator to focus on text regions.
## Train
- *We generated a pseudo text stroke mask, automatically calculated by taking the pixel value difference between the input image and ground truth image.* The TSR masks help train the TSR attention module to distinguish the TSR.
- *The TSSR mask is the intersection region between the exterior of the pseudo text stroke mask and the interior of the text box mask.* It makes the TSSR attention module train to focus on the colors and textures of the TSSR.
### Loss
- The GA learns the gate parameter on its own, and can thus adjust the respective attention ratios allocated to TSR and TSSR during training. The loss function is designed to be applied only within the text box regions, not to the entire score map.
## Metric
- **In this paper, we use Detection Eval [19] as an evaluation metric and CRAFT [1] as an auxiliary detector. However, that method only indicates how much text has been erased, not output quality.**
- **S. Zhang et al. [22] proposed using the evaluation method that is used in image inpainting. They used PSNR, SSIM, MSE, AGE, pEPs, pCEPs to evaluate image quality. The higher the value of PSNR and SSIM, and the lower the value of other metrics, the better the quality of the output image. We use PSNR, SSIM, and AGE for evaluation.**


- Most STR methods attempt to perform in-painting of TSR as well as reconstruc- tion of the entire image. However, our approach can skip the reconstruction of non-masked regions altogether because if the text box region is given as an input to the STR model, there is no need to render the entire image for the output. Therefore, we modified the loss function so that our model’s generator only has to focus on the text box region during training. Note that the generator’s loss is only calculated with respect to the text box region. Every other region is con- sidered don’t care and therefore is irrelevant during training Synthetic data. The Oxford Synthetic text dataset [5] is adopted for train- ing and evaluation. The dataset contains around 800,000 images composed of 8,000 text-free images. We randomly selected 95% images for training, selected 10,000 images for testing, and used the rest for validation. Note that the back- ground images in the train set and test set are mutually exclusive. Real data. SCUT-EnsText [9] is a real dataset for scene text removal. The dataset, which was manually generated from Chinese and English text images, contains 2,749 train and 813 test images. In this paper, we adopted these images for training and evaluation. Preprocessing. We need stroke-level segmentation masks to train our model. However, the existing datasets do not provide stroke-level segmentation masks. Therefore, we created it automatically by calculating the pixel value difference between the input image and the ground truth image. To suppress noise, we set a threshold of 25. We combined synthetic and real datasets. In total, we used 738,113 images for training. The test set was used separately to distinguish between performance on real and synthetic datasets.
- Spatial attention does a poor job finding the text strokes and the surrounding regions if simply applied in STR. In comparison, we can see that the GA pays more attention to the text stroke regions in low-level features while paying more attention to the surrounding regions of the text in high-level features
- the GA module puts more emphasis on the surrounding regions of text strokes rather than the text strokes as it approaches higher-level features.
- GA uses attention on the text strokes and the surrounding region’s colors and textures to surgically erase text from images. RoIG makes the generator focus on only the region with text instead of the entire image for more efficient training.