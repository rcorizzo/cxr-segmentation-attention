# Localization-Aware Deep Medical Image Classification via Segmentation Driven Gradient-based Attention
The repository contains a method that calculates bounding boxes in real-time during segmentation and integrates Grad-CAM to enhance the loss function for deep classification models.

![Image](https://github.com/user-attachments/assets/93bfd5a1-b525-49c0-9641-a6d620d5be7a)

## Data Source
- **NIH ChestX-ray14 Dataset**  
  Provided by the National Institutes of Health Clinical Center.  
  Download link: [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)

- **Ottawa Dataset**  
  Collected in collaboration with The Ottawa Hospital in Canada.  
  Due to privacy restrictions and data sharing agreements, this dataset is not publicly available.

- **Chest X-ray Masks and Labels (Kaggle)**  
  This dataset was obtained from Kaggle: [Chest X-ray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels).  
  It contains chest X-ray images with corresponding segmentation masks and disease labels. The dataset is publicly available for research use.  The UNet model used in this work was trained using this dataset for lung region segmentation.

## Data preparation
For NIH dataset, all compressed packages of images can be downloaded in batch through the .py script contained in the "images" folder. Structure after downloading and extracting all files of images for NIH data:
```
/data/NIH/
  images_01/
    00000001_000.png
    00000001_001.png
  images_02/
    00001336_000.png
    00001337_000.png
  ...
```
Data structure required for model training and evaluation (after running data_prepare.py in data folder):
```
/data/class1_class2/
  train/
    class1/
      img1.png
    class2/
      img2.png
  val/
    class1/
      img3.png
    class/2
      img4.png
  test/
    class1/
      img5.png
    class2/
      img6.png
```
The [`model`](./model) folder contains a text file with the download link to the pre-trained U-Net model. Structure of training data:
```
/data/Ottawa_masks_512/
  img1.png
  img2.png
  img3.png
```

## Training and evaluation
To run the classification task with cross entropy loss:
```
python model_loss_ce.py 
--path: Path of data
--nclass: Number of classes, 2 or 3 
--gpu: Specify which gpu to use, not required, default is 0
```
To run the classification task with the proposed loss:
```
python model_loss_attent.py 
--dataset: Specify the dateset
--path: Path of data, ../data/class-name/  
--backbone: Backbone model, e.g. pvt, vgg or resent
--task: Specify a classification task: ne, np, nep or neps
--gpu: Specify a gpu if there are more than one
--batch: Specify the batch size
--lamda: Specify a lambda value betweeen 0 to 1
--thresh: Specify a threshold value betweeen 0 to 1
--isAdaptive: A boolean flag indicates whether the value of lambda is adaptive, e.g. 0.9
```
Upon completion of the training and evaluation process, the following files are generated:

- **Model checkpoint (`.pt` file):**  
  The trained model is saved with a filename that encodes the hyperparameters `lambda` and `threshold` used during training.

- **Performance log (`.txt` file):**  
  This file contains detailed metrics on the model’s performance evaluated on the validation and test datasets. The filename similarly reflects the corresponding `lambda` and `threshold` values.

For reference, a sample performance log can be found in the [`scripts/results/`](scripts/results) directory.

__Examples of commands__

To train a PVT model with cross entropy loss, 3 classes:
```
python pvt_loss_ce.py --path ../data/nofind_effusion_pneumothorax/ --nclass 3
```
To train a PVT model with proposed loss, dataset is NIH, binary classification (No Finding and Effusion), gpu is 0, lambda is 0.25, threshold is 0.7, this repository provides a trained PVT model (please refer to the link provided in the text file within [`model`](./model) folder) for below command:
```
python model_loss_attent.py \
--dataset nih \
--path ../data/NIH/nofind_effusion/ \
--backbone pvt \
--task ne \
--gpu 0 \
--lamda 0.25 \
--thresh 0.7

```
To train a VGG16 model with proposed loss, dataset is Ottawa, multi-class (No Finding, Effusion, Pneumothorax, and Subcutaneous emphysema), gpu is 0, lambda is adaptive, threshold is 0.9:
```
python model_loss_attent.py \
--dataset ottawa \
--path ../data/Ottawa/nf_e_p_s/ \
--backbone vgg \
--task neps \
--gpu 0 \
--isAdaptive \
--thresh 0.9

```
To train a ResNet50 model with proposed loss, dataset is NIH, binary classes (No Finding vs Pneumothorax), gpu is 1, lambda is 0.25, threshold is 0.7:
```
python model_loss_attent.py \
--dataset nih \
--path ../data/NIH/nofind_pneumothorax/ \
--backbone resnet \
--task np \
--gpu 1 \
--lamda 0.25 \
--thresh 0.7
```

## Results
##### Effusion vs No Finding (NIH dataset)
<table class="dataintable">
<tr>
   <th>Model</th>
   <th>Metric</th>
   <th>λ = 1</th>
   <th>λ = 0.75</th>
   <th>λ = 0.5</th>
   <th>λ = 0.25</th>
   <th>λ = Adapt</th>
</tr>

<tr>
   <td rowspan="4">PVT</td>
   <td>Acc / Rec</td>
   <td>0.817</td>
   <td>0.822</td>
   <td>0.828</td>
   <td>0.821</td>
   <td>0.855</td>
</tr>
<tr>
    <td>Prec</td>
    <td>0.823</td>
    <td>0.824</td>
    <td>0.828</td>
    <td>0.821</td>
    <td>0.857</td>
</tr>
<tr>
    <td>F1</td>
    <td>0.817</td>
    <td>0.821</td>
    <td>0.828</td>
    <td>0.821</td>
    <td>0.854</td>
</tr>
<tr>
    <td>AUC</td>
    <td>0.884</td>
    <td>0.896</td>
    <td>0.896</td>
    <td>0.894</td>
    <td>0.909</td>
</tr>

<tr>
   <td rowspan="4">VGG16</td>
   <td>Acc / Rec</td>
   <td>0.834</td>
   <td>0.821</td>
   <td>0.821</td>
   <td>0.820</td>
   <td>0.825</td>
</tr>
<tr>
    <td>Prec</td>
    <td>0.835</td>
    <td>0.822</td>
    <td>0.822</td>
    <td>0.824</td>
    <td>0.826</td>
</tr>
<tr>
    <td>F1</td>
    <td>0.834</td>
    <td>0.821</td>
    <td>0.821</td>
    <td>0.820</td>
    <td>0.825</td>
</tr>
<tr>
    <td>AUC</td>
    <td>0.907</td>
    <td>0.900</td>
    <td>0.896</td>
    <td>0.900</td>
    <td>0.891</td>
</tr>

<tr>
   <td rowspan="4">ResNet50</td>
   <td>Acc / Rec</td>
   <td>0.809</td>
   <td>0.808</td>
   <td>0.784</td>
   <td>0.815</td>
   <td>0.809</td>
</tr>
<tr>
    <td>Prec</td>
    <td>0.814</td>
    <td>0.809</td>
    <td>0.788</td>
    <td>0.815</td>
    <td>0.810</td>
</tr>
<tr>
    <td>F1</td>
    <td>0.808</td>
    <td>0.808</td>
    <td>0.784</td>
    <td>0.815</td>
    <td>0.809</td>
</tr>
<tr>
    <td>AUC</td>
    <td>0.870</td>
    <td>0.863</td>
    <td>0.860</td>
    <td>0.874</td>
    <td>0.872</td>
</tr>
</table>

##### Effusion vs No Finding (Ottawa dataset)
<table class="dataintable">
<tr>
   <th>Model</th>
   <th>Metric</th>
   <th>λ = 1</th>
   <th>λ = 0.75</th>
   <th>λ = 0.5</th>
   <th>λ = 0.25</th>
   <th>λ = Adapt</th>
</tr>

<tr>
   <td rowspan="4">PVT</td>
   <td>Acc / Rec</td>
   <td>0.628</td>
   <td>0.660</td>
   <td>0.692</td>
   <td>0.654</td>
   <td>0.628</td>
</tr>
<tr>
    <td>Prec</td>
    <td>0.629</td>
    <td>0.662</td>
    <td>0.693</td>
    <td>0.659</td>
    <td>0.630</td>
</tr>
<tr>
    <td>F1</td>
    <td>0.628</td>
    <td>0.659</td>
    <td>0.692</td>
    <td>0.651</td>
    <td>0.627</td>
</tr>
<tr>
    <td>AUC</td>
    <td>0.743</td>
    <td>0.723</td>
    <td>0.752</td>
    <td>0.678</td>
    <td>0.683</td>
</tr>

<tr>
   <td rowspan="4">VGG16</td>
   <td>Acc / Rec</td>
   <td>0.673</td>
   <td>0.667</td>
   <td>0.673</td>
   <td>0.654</td>
   <td>0.692</td>
</tr>
<tr>
    <td>Prec</td>
    <td>0.682</td>
    <td>0.667</td>
    <td>0.674</td>
    <td>0.662</td>
    <td>0.692</td>
</tr>
<tr>
    <td>F1</td>
    <td>0.669</td>
    <td>0.666</td>
    <td>0.672</td>
    <td>0.649</td>
    <td>0.692</td>
</tr>
<tr>
    <td>AUC</td>
    <td>0.726</td>
    <td>0.735</td>
    <td>0.745</td>
    <td>0.745</td>
    <td>0.760</td>
</tr>

<tr>
   <td rowspan="4">ResNet50</td>
   <td>Acc / Rec</td>
   <td>0.654</td>
   <td>0.635</td>
   <td>0.628</td>
   <td>0.673</td>
   <td>0.667</td>
</tr>
<tr>
    <td>Prec</td>
    <td>0.659</td>
    <td>0.638</td>
    <td>0.630</td>
    <td>0.673</td>
    <td>0.669</td>
</tr>
<tr>
    <td>F1</td>
    <td>0.651</td>
    <td>0.632</td>
    <td>0.627</td>
    <td>0.673</td>
    <td>0.665</td>
</tr>
<tr>
    <td>AUC</td>
    <td>0.736</td>
    <td>0.672</td>
    <td>0.739</td>
    <td>0.737</td>
    <td>0.711</td>
</tr>
</table>

##### Pneumothorax vs No Finding (NIH dataset)						
<table class="dataintable">
<tr>
   <th>Model</th>
   <th>Metric</th>
   <th>λ = 1</th>
   <th>λ = 0.75</th>
   <th>λ = 0.5</th>
   <th>λ = 0.25</th>
   <th>λ = Adapt</th>
</tr>

<tr>
   <td rowspan="4">PVT</td>
   <td>Acc / Rec</td>
   <td>0.771</td>
   <td>0.793</td>
   <td>0.784</td>
   <td>0.759</td>
   <td>0.790</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.772</td>
   <td>0.793</td>
   <td>0.784</td>
   <td>0.760</td>
   <td>0.791</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.771</td>
   <td>0.793</td>
   <td>0.783</td>
   <td>0.758</td>
   <td>0.790</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.842</td>
   <td>0.872</td>
   <td>0.858</td>
   <td>0.841</td>
   <td>0.868</td>
</tr>

<tr>
   <td rowspan="4">VGG16</td>
   <td>Acc / Rec</td>
   <td>0.765</td>
   <td>0.763</td>
   <td>0.768</td>
   <td>0.776</td>
   <td>0.778</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.766</td>
   <td>0.764</td>
   <td>0.768</td>
   <td>0.776</td>
   <td>0.778</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.765</td>
   <td>0.763</td>
   <td>0.768</td>
   <td>0.776</td>
   <td>0.778</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.841</td>
   <td>0.828</td>
   <td>0.848</td>
   <td>0.848</td>
   <td>0.860</td>
</tr>

<tr>
   <td rowspan="4">ResNet50</td>
   <td>Acc / Rec</td>
   <td>0.727</td>
   <td>0.749</td>
   <td>0.731</td>
   <td>0.738</td>
   <td>0.743</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.727</td>
   <td>0.751</td>
   <td>0.733</td>
   <td>0.739</td>
   <td>0.746</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.727</td>
   <td>0.749</td>
   <td>0.731</td>
   <td>0.738</td>
   <td>0.742</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.817</td>
   <td>0.821</td>
   <td>0.807</td>
   <td>0.818</td>
   <td>0.804</td>
</tr>
</table>

##### Pneumothorax vs No Finding (Ottawa dataset)
<table class="dataintable">
<tr>
   <th>Model</th>
   <th>Metric</th>
   <th>λ = 1</th>
   <th>λ = 0.75</th>
   <th>λ = 0.5</th>
   <th>λ = 0.25</th>
   <th>λ = Adapt</th>
</tr>

<tr>
   <td rowspan="4">PVT</td>
   <td>Acc / Rec</td>
   <td>0.565</td>
   <td>0.578</td>
   <td>0.595</td>
   <td>0.595</td>
   <td>0.642</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.565</td>
   <td>0.578</td>
   <td>0.595</td>
   <td>0.596</td>
   <td>0.642</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.564</td>
   <td>0.576</td>
   <td>0.594</td>
   <td>0.594</td>
   <td>0.642</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.626</td>
   <td>0.632</td>
   <td>0.654</td>
   <td>0.623</td>
   <td>0.653</td>
</tr>

<tr>
   <td rowspan="4">VGG16</td>
   <td>Acc / Rec</td>
   <td>0.634</td>
   <td>0.642</td>
   <td>0.681</td>
   <td>0.647</td>
   <td>0.629</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.647</td>
   <td>0.643</td>
   <td>0.681</td>
   <td>0.648</td>
   <td>0.629</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.625</td>
   <td>0.642</td>
   <td>0.681</td>
   <td>0.646</td>
   <td>0.629</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.707</td>
   <td>0.681</td>
   <td>0.714</td>
   <td>0.740</td>
   <td>0.680</td>
</tr>

<tr>
   <td rowspan="4">ResNet50</td>
   <td>Acc / Rec</td>
   <td>0.552</td>
   <td>0.547</td>
   <td>0.504</td>
   <td>0.608</td>
   <td>0.556</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.557</td>
   <td>0.549</td>
   <td>0.505</td>
   <td>0.609</td>
   <td>0.558</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.541</td>
   <td>0.544</td>
   <td>0.496</td>
   <td>0.607</td>
   <td>0.553</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.591</td>
   <td>0.569</td>
   <td>0.504</td>
   <td>0.627</td>
   <td>0.583</td>
</tr>
</table>

##### Effusion, Pneumothorax and No Finding (NIH dataset)						
<table class="dataintable">
<tr>
   <th>Model</th>
   <th>Metric</th>
   <th>λ = 1</th>
   <th>λ = 0.75</th>
   <th>λ = 0.5</th>
   <th>λ = 0.25</th>
   <th>λ = Adapt</th>
</tr>

<tr>
   <td rowspan="4">PVIT</td>
   <td>Acc / Rec</td>
   <td>0.651</td>
   <td>0.651</td>
   <td>0.670</td>
   <td>0.669</td>
   <td>0.677</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.651</td>
   <td>0.651</td>
   <td>0.672</td>
   <td>0.669</td>
   <td>0.678</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.652</td>
   <td>0.651</td>
   <td>0.671</td>
   <td>0.669</td>
   <td>0.677</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.812</td>
   <td>0.819</td>
   <td>0.843</td>
   <td>0.828</td>
   <td>0.839</td>
</tr>

<tr>
   <td rowspan="4">VGG16</td>
   <td>Acc / Rec</td>
   <td>0.674</td>
   <td>0.653</td>
   <td>0.640</td>
   <td>0.660</td>
   <td>0.641</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.674</td>
   <td>0.657</td>
   <td>0.641</td>
   <td>0.660</td>
   <td>0.641</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.674</td>
   <td>0.651</td>
   <td>0.640</td>
   <td>0.660</td>
   <td>0.641</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.831</td>
   <td>0.822</td>
   <td>0.820</td>
   <td>0.822</td>
   <td>0.817</td>
</tr>

<tr>
   <td rowspan="4">ResNet50</td>
   <td>Acc / Rec</td>
   <td>0.631</td>
   <td>0.610</td>
   <td>0.601</td>
   <td>0.620</td>
   <td>0.632</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.631</td>
   <td>0.609</td>
   <td>0.598</td>
   <td>0.622</td>
   <td>0.632</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.626</td>
   <td>0.609</td>
   <td>0.598</td>
   <td>0.609</td>
   <td>0.630</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.800</td>
   <td>0.781</td>
   <td>0.781</td>
   <td>0.793</td>
   <td>0.794</td>
</tr>
</table>

##### Effusion, Pneumothorax and No Finding (Ottawa dataset)	
<table class="dataintable">
<tr>
   <th>Model</th>
   <th>Metric</th>
   <th>λ = 1</th>
   <th>λ = 0.75</th>
   <th>λ = 0.5</th>
   <th>λ = 0.25</th>
   <th>λ = Adapt</th>
</tr>

<tr>
   <td rowspan="4">PVIT</td>
   <td>Acc / Rec</td>
   <td>0.457</td>
   <td>0.491</td>
   <td>0.453</td>
   <td>0.474</td>
   <td>0.496</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.465</td>
   <td>0.488</td>
   <td>0.462</td>
   <td>0.472</td>
   <td>0.491</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.443</td>
   <td>0.488</td>
   <td>0.453</td>
   <td>0.470</td>
   <td>0.488</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.643</td>
   <td>0.663</td>
   <td>0.644</td>
   <td>0.650</td>
   <td>0.677</td>
</tr>

<tr>
   <td rowspan="4">VGG16</td>
   <td>Acc / Rec</td>
   <td>0.474</td>
   <td>0.521</td>
   <td>0.521</td>
   <td>0.496</td>
   <td>0.474</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.466</td>
   <td>0.518</td>
   <td>0.516</td>
   <td>0.486</td>
   <td>0.474</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.459</td>
   <td>0.517</td>
   <td>0.507</td>
   <td>0.482</td>
   <td>0.445</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.697</td>
   <td>0.709</td>
   <td>0.722</td>
   <td>0.688</td>
   <td>0.667</td>
</tr>

<tr>
   <td rowspan="4">ResNet50</td>
   <td>Acc / Rec</td>
   <td>0.496</td>
   <td>0.419</td>
   <td>0.466</td>
   <td>0.487</td>
   <td>0.453</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.493</td>
   <td>0.407</td>
   <td>0.448</td>
   <td>0.486</td>
   <td>0.437</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.488</td>
   <td>0.408</td>
   <td>0.450</td>
   <td>0.484</td>
   <td>0.441</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.683</td>
   <td>0.610</td>
   <td>0.675</td>
   <td>0.679</td>
   <td>0.674</td>
</tr>
</table>

##### Subcutaneous emphysema vs No Finding (Ottawa dataset)
<table class="dataintable">
<tr>
   <th>Model</th>
   <th>Metric</th>
   <th>λ = 1</th>
   <th>λ = 0.75</th>
   <th>λ = 0.5</th>
   <th>λ = 0.25</th>
   <th>λ = Adapt</th>
</tr>

<tr>
   <td rowspan="4">PVT</td>
   <td>Acc / Rec</td>
   <td>0.754</td>
   <td>0.735</td>
   <td>0.727</td>
   <td>0.758</td>
   <td>0.739</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.757</td>
   <td>0.740</td>
   <td>0.727</td>
   <td>0.759</td>
   <td>0.740</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.753</td>
   <td>0.733</td>
   <td>0.727</td>
   <td>0.757</td>
   <td>0.738</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.828</td>
   <td>0.810</td>
   <td>0.811</td>
   <td>0.831</td>
   <td>0.799</td>
</tr>

<tr>
   <td rowspan="4">VGG16</td>
   <td>Acc / Rec</td>
   <td>0.750</td>
   <td>0.742</td>
   <td>0.742</td>
   <td>0.731</td>
   <td>0.758</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.751</td>
   <td>0.742</td>
   <td>0.751</td>
   <td>0.731</td>
   <td>0.776</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.750</td>
   <td>0.742</td>
   <td>0.740</td>
   <td>0.731</td>
   <td>0.753</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.790</td>
   <td>0.811</td>
   <td>0.829</td>
   <td>0.813</td>
   <td>0.811</td>
</tr>

<tr>
   <td rowspan="4">ResNet50</td>
   <td>Acc / Rec</td>
   <td>0.667</td>
   <td>0.678</td>
   <td>0.621</td>
   <td>0.652</td>
   <td>0.652</td>
</tr>
<tr>
   <td>Prec</td>
   <td>0.668</td>
   <td>0.683</td>
   <td>0.621</td>
   <td>0.657</td>
   <td>0.652</td>
</tr>
<tr>
   <td>F1</td>
   <td>0.666</td>
   <td>0.676</td>
   <td>0.621</td>
   <td>0.649</td>
   <td>0.651</td>
</tr>
<tr>
   <td>AUC</td>
   <td>0.707</td>
   <td>0.712</td>
   <td>0.667</td>
   <td>0.715</td>
   <td>0.696</td>
</tr>
</table>

For the full version of the results, please refer to the file DM746-Appendix.pdf.


