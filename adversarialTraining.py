import copy
import numpy as np
import os
from pathlib import Path
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchattacks import FGSM, PGD
from torchvision import models, transforms

from utils.balancedDataset import BalancedDataset
from utils.const import *
from utils.helperFunctions import setSeed, getScores, getSubDirs  # Ensure getSubDirs is imported
from utils.tasks import currentTask

import warnings
warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

# ITERABLE PARAMETERS

# Ratio between classes cat and dog
BALANCES = [[50, 50], [40, 60], [30, 70], [20, 80]]

# Models to train
MODEL_NAMES = ["alexnet", "resnet", "vgg"]

# OTHER PARAMETERS
NUM_CLASSES = 2  # Binary Classification
NUM_WORKERS = 0
PIN_MEMORY = False

# Batch size for training (change depending on how much memory you have)
BATCH_SIZE = 64

# Early stopping
NUM_EPOCHS = 500  # Number of epochs to train for
PATIENCE_ES = 10  # Patience for early stopping
DELTA_ES = 0.0001  # Delta for early stopping

# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
FEATURE_EXTRACT = False

LEARNING_RATE = 0.001  # The learning rate of the optimizer
MOMENTUM = 0.9  # The momentum of the optimizer

attacksParams = {
    "math": {
        "FGSM": [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5],
        "PGD": [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
    }
}



ADVERSARIAL_DIR = f"./adversarialSamplesVal/{currentTask}"



def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, delta=0, patience=10, adaptive=False):
    since = time.time()
    last_since = time.time()

    scores_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    best_score = None
    counter = 0

    for epoch in range(num_epochs):
        print('[💪 EPOCH] {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        epoch_score = None

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            labels_outputs = torch.tensor([]).to(DEVICE, non_blocking=True)
            labels_targets = torch.tensor([]).to(DEVICE, non_blocking=True)

            # Iterate over data
            setSeed()
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if adaptive:
                            attacks = [
                                FGSM(model, eps=float(epsilons[epoch])),
                                PGD(model, eps=float(epsilons[epoch])),
                            ]
                            for attack in attacks:
                                adversarial_samples = attack(inputs, labels)
                                adversarial_samples_outputs = model(adversarial_samples)

                                attack_loss = criterion(adversarial_samples_outputs, labels)
                                attack_loss.backward()
                                optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                labels_outputs = torch.cat([labels_outputs, preds], dim=0)
                labels_targets = torch.cat([labels_targets, labels], dim=0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc, epoch_prec, epoch_rec, epoch_f1 = getScores(
                labels_targets, labels_outputs, complete=False)

            print('[🗃️ {}] Loss: {:.4f} Acc: {:.4f} Pre: {:.4f} Rec: {:.4f} F-Score: {:.4f}'.format(
                phase.upper(), epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1))

            time_elapsed = time.time() - last_since
            last_since = time.time()
            print("\t[🕑] {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60))                

            if phase == 'val':
                epoch_score = epoch_f1

                # Deep copy the model
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())

                # Store scores history
                scores_history.append({
                    "loss": epoch_loss,
                    "acc": epoch_acc.cpu().numpy(),
                    "precision": epoch_prec.cpu().numpy(),
                    "recall": epoch_rec.cpu().numpy(),
                    "f1": epoch_f1.cpu().numpy()
                })

        if best_score is None:
            best_score = epoch_score
        elif epoch_score <= best_score + delta:
            counter += 1
            print("\t[⚠️ EARLY STOPPING] {}/{}".format(counter, patience))
            if counter >= patience:
                break
        else:
            best_score = epoch_score
            counter = 0

        print()

    time_elapsed = time.time() - since
    print()
    print('[🕑 TRAINING COMPLETE] {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('[🥇 BEST SCORE] F-Score: {:4f}'.format(best_f1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, scores_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size



############################################################################################################################################




### ITERATING MODELS AND BALANCES ###
setSeed()

torch.cuda.empty_cache()

for dataset_dir in sorted(getSubDirs(DATASETS_DIR)):
    torch.cuda.empty_cache()
    for model_name in sorted(MODEL_NAMES):
        torch.cuda.empty_cache()
        for balance in sorted(BALANCES):
            torch.cuda.empty_cache()

            data_dir = os.path.join(DATASETS_DIR, dataset_dir)
            fgsm_data_dir = os.path.join(ADVERSARIAL_DIR, dataset_dir, 'math', 'FGSM', model_name, "_".join(str(b) for b in balance), '0.2')
            pgd_data_dir = fgsm_data_dir.replace('FGSM', 'PGD')

            current_dir = os.getcwd()
            curr_append = os.path.join(os.path.join(
                ADV_MODELS_DIR, dataset_dir), model_name)

            model_save_path = os.path.join(current_dir, curr_append)
            # if not os.path.exists(model_save_path):
                # os.makedirs(model_save_path)

            model_save_name = "{}_{}".format(
                model_name, "_".join(str(b) for b in balance))
            model_save_path = os.path.join(model_save_path, model_save_name)

            # Initialize the model for this run
            model_ft, input_size = initialize_model(
                model_name, NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)

            # Data resize and normalization
            data_transforms = {
                "train": transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
                ]),
                "val": transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
                ]),
            }

            # Create training and validation datasets
            image_datasets = {x: BalancedDataset(os.path.join(data_dir, x),
                                                 transform=data_transforms[x],
                                                 balance=balance,
                                                 check_images=False,
                                                 use_cache=True) for x in ["train", "val"]}
            fgsm_image_datasets = {x: BalancedDataset(fgsm_data_dir,
                                                 transform=data_transforms[x],
                                                 balance=balance,
                                                 check_images=False,
                                                 use_cache=True) for x in ["train"]}
            pgd_image_datasets = {x: BalancedDataset(pgd_data_dir,
                                                 transform=data_transforms[x],
                                                 balance=balance,
                                                 check_images=False,
                                                 use_cache=True) for x in ["train"]}
           
            fgsm_datasets = {
                "train": torch.utils.data.ConcatDataset([image_datasets["train"], fgsm_image_datasets["train"]]),
                "val": image_datasets["val"]
            }
            pgd_datasets = {
                "train": torch.utils.data.ConcatDataset([image_datasets["train"], pgd_image_datasets["train"]]),
                "val": image_datasets["val"]
            }
            ensamble_datasets = {
                "train": torch.utils.data.ConcatDataset([image_datasets["train"], fgsm_image_datasets["train"], pgd_image_datasets["train"]]),
                "val": image_datasets["val"]
            }

            setSeed()
            fgsm_dict = {x: torch.utils.data.DataLoader(
                fgsm_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY) for x in ["train", "val"]}
            pgd_dict = {x: torch.utils.data.DataLoader(
                pgd_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY) for x in ["train", "val"]}
            ensamble_dict = {x: torch.utils.data.DataLoader(
                ensamble_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY) for x in ["train", "val"]}

            dicts = []
            for d in [fgsm_dict, pgd_dict, ensamble_dict]:
                dicts.append(d)
            
            for i, d in enumerate(dicts):
                
                print(f'\n\n[🤖 MODEL] {"FGSM" if i == 0 else "PGD" if i == 1 else "ensamble"} - {dataset_dir} - {model_name} - {balance}\n\n')

                this_model_save_path = model_save_path.replace(currentTask, f'{currentTask}/FGSM' if i == 0
                                                          else f'{currentTask}/PGD' if i == 1
                                                          else f'{currentTask}/ensamble')

                os.makedirs(os.path.dirname(this_model_save_path), exist_ok=True)
                if os.path.exists(this_model_save_path + ".pt"):
                    print('\t[✅ SKIPPING] ALREADY TRAINED')
                    continue

                # Initialize the model for this run
                model_ft, input_size = initialize_model(
                    model_name, NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)
                model_ft = model_ft.to(DEVICE, non_blocking=True)

                params_to_update = model_ft.parameters()
                if FEATURE_EXTRACT:
                    params_to_update = []
                    for name, param in model_ft.named_parameters():
                        if param.requires_grad == True:
                            params_to_update.append(param)

                # Observe that all parameters are being optimized
                optimizer_ft = optim.SGD(
                    params_to_update, lr=LEARNING_RATE, momentum=MOMENTUM)

                # Setup the loss fxn
                criterion = nn.CrossEntropyLoss()

                # Train and evaluate
                setSeed()
                model_ft, scores_history = train_model(model_ft, d, criterion, optimizer_ft,
                                                    num_epochs=NUM_EPOCHS, is_inception=False,
                                                    delta=DELTA_ES, patience=PATIENCE_ES)
                
                torch.save({
                    'model': model_ft,
                    'task': currentTask,
                    'dataset': dataset_dir,
                    'learning_rate': LEARNING_RATE,
                    'momentum': MOMENTUM,
                    'balance': balance,
                    'model_name': model_name,
                    'batch_size': BATCH_SIZE,
                    'num_epochs': NUM_EPOCHS,
                    'criterion': criterion,
                    'optimizer': optimizer_ft,
                    'scores_history': scores_history,
                    'delta_es': DELTA_ES,
                    'patience_es': PATIENCE_ES
                }, this_model_save_path + ".pt")

                print("[💾 SAVED]", dataset_dir, model_name,
                    "/".join(str(b) for b in balance))
                
                # Free memory explicitly
                del model_ft, optimizer_ft
                torch.cuda.empty_cache()

            ###########################
            ### CURRICULUM TRAINING ###
            ###########################

            epsilons = sorted(os.listdir(os.path.join(ADVERSARIAL_DIR, dataset_dir, 'math', 'FGSM', model_name, "_".join(str(b) for b in balance))))

            # Initialize the model for this run
            model_ft, input_size = initialize_model(
                model_name, NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)
            model_ft = model_ft.to(DEVICE, non_blocking=True)

            params_to_update = model_ft.parameters()
            if FEATURE_EXTRACT:
                params_to_update = []
                for name, param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)

            # Observe that all parameters are being optimized
            optimizer_ft = optim.SGD(
                params_to_update, lr=LEARNING_RATE, momentum=MOMENTUM)

            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()

            # Train and evaluate
            setSeed()

            for i, epsilon in enumerate(epsilons):
                print(f'\n\n[🤖 MODEL] CURRICULUM @ {epsilon} ({i+1}/{len(epsilons)}) - {dataset_dir} - {model_name} - {balance}\n\n')

                this_model_save_path = model_save_path.replace(currentTask, f'{currentTask}/curriculum')
                os.makedirs(os.path.dirname(this_model_save_path), exist_ok=True)
                if os.path.exists(this_model_save_path + ".pt"):
                    print('\t[✅ SKIPPING] ALREADY TRAINED')
                    continue

                fgsm_image_datasets = {x: BalancedDataset(fgsm_data_dir.replace('0.2', epsilon),
                                                    transform=data_transforms[x],
                                                    balance=balance,
                                                    check_images=False,
                                                    use_cache=True) for x in ["train"]}
                pgd_image_datasets = {x: BalancedDataset(pgd_data_dir.replace('0.2', epsilon),
                                                    transform=data_transforms[x],
                                                    balance=balance,
                                                    check_images=False,
                                                    use_cache=True) for x in ["train"]}
            
                ensamble_datasets = {
                    "train": torch.utils.data.ConcatDataset([image_datasets["train"], fgsm_image_datasets["train"], pgd_image_datasets["train"]]),
                    "val": image_datasets["val"]
                }

                dataloaders_dict = {x: torch.utils.data.DataLoader(
                    ensamble_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY) for x in ["train", "val"]}

                model_ft, scores_history = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                                    num_epochs=5, is_inception=False,
                                                    delta=DELTA_ES, patience=3)
                
            if not os.path.exists(this_model_save_path + ".pt"):
                torch.save({
                    'model': model_ft,
                    'task': currentTask,
                    'dataset': dataset_dir,
                    'learning_rate': LEARNING_RATE,
                    'momentum': MOMENTUM,
                    'balance': balance,
                    'model_name': model_name,
                    'batch_size': BATCH_SIZE,
                    'num_epochs': NUM_EPOCHS,
                    'criterion': criterion,
                    'optimizer': optimizer_ft,
                    'scores_history': scores_history,
                    'delta_es': DELTA_ES,
                    'patience_es': PATIENCE_ES
                }, this_model_save_path + ".pt")

            #########################
            ### ADAPTIVE TRAINING ###
            #########################

            epsilons = sorted(os.listdir(os.path.join(ADVERSARIAL_DIR, dataset_dir, 'math', 'FGSM', model_name, "_".join(str(b) for b in balance))))

            print(f'\n\n[🤖 MODEL] ADAPTIVE - {dataset_dir} - {model_name} - {balance}\n\n')

            this_model_save_path = model_save_path.replace(currentTask, f'{currentTask}/adaptive')
            os.makedirs(os.path.dirname(this_model_save_path), exist_ok=True)
            if not os.path.exists(this_model_save_path + ".pt"):
                # continue

                # Initialize the model for this run
                model_ft, input_size = initialize_model(
                    model_name, NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)

                # Create training and validation dataloaders
                setSeed()
                dataloaders_dict = {x: torch.utils.data.DataLoader(
                    image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY) for x in ["train", "val"]}

                model_ft = model_ft.to(DEVICE, non_blocking=True)

                params_to_update = model_ft.parameters()
                if FEATURE_EXTRACT:
                    params_to_update = []
                    for name, param in model_ft.named_parameters():
                        if param.requires_grad == True:
                            params_to_update.append(param)

                # Observe that all parameters are being optimized
                optimizer_ft = optim.SGD(
                    params_to_update, lr=LEARNING_RATE, momentum=MOMENTUM)

                # Setup the loss fxn
                criterion = nn.CrossEntropyLoss()

                # Train and evaluate
                setSeed()
                model_ft, scores_history = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                                    num_epochs=len(epsilons), is_inception=False,
                                                    delta=DELTA_ES, patience=PATIENCE_ES, adaptive=True)

                torch.save({
                    'model': model_ft,
                    'task': currentTask,
                    'dataset': dataset_dir,
                    'learning_rate': LEARNING_RATE,
                    'momentum': MOMENTUM,
                    'balance': balance,
                    'model_name': model_name,
                    'batch_size': BATCH_SIZE,
                    'num_epochs': NUM_EPOCHS,
                    'criterion': criterion,
                    'optimizer': optimizer_ft,
                    'scores_history': scores_history,
                    'delta_es': DELTA_ES,
                    'patience_es': PATIENCE_ES
                }, this_model_save_path + ".pt")

                print("[💾 SAVED]", dataset_dir, model_name,
                    "/".join(str(b) for b in balance))
            else:
                print('\t[✅ SKIPPING] ALREADY TRAINED')

            #########################
            ### NON-MATH TRAINING ###
            #########################

            nonmath = {
                "BoxBlur_5": ("BoxBlur", "5"),
                "GaussianNoise_005": ("GaussianNoise", "0.05"),
                "GreyScale_1": ("GreyScale", "1"),
                "InvertColor_1": ("InvertColor", "1"),
                "RandomBlackBox_100": ("RandomBlackBox", "100"),
                "SaltPepper_005": ("SaltPepper", "0.05"),
                "BoxBlur_1": ("BoxBlur", "1"),
            }

            nonmath_dirs = {
                name: os.path.join(ADVERSARIAL_DIR, dataset_dir, 'nonMath', attack_type, value)
                for name, (attack_type, value) in nonmath.items()
            }

            nonmath_datasets_dirs = {
                name: {x: BalancedDataset(dir_path,
                                        transform=data_transforms[x],
                                        balance=balance,
                                        check_images=False,
                                        use_cache=True) 
                    for x in ["train"]}
                for name, dir_path in nonmath_dirs.items()
            }

            nonmath_datasets = {
                "train": torch.utils.data.ConcatDataset(
                    [image_datasets["train"]] + [nonmath_datasets_dirs[name]["train"] for name in nonmath_datasets_dirs]
                ),
                "val": image_datasets["val"]
            }

            nonmath_dict = {x: torch.utils.data.DataLoader(
                nonmath_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY) for x in ["train", "val"]}

            print(f'\n\n[🤖 MODEL] NON-MATH - {dataset_dir} - {model_name} - {balance}\n\n')

            this_model_save_path = model_save_path.replace(currentTask, f'{currentTask}/nonMath')

            os.makedirs(os.path.dirname(this_model_save_path), exist_ok=True)
            if not os.path.exists(this_model_save_path + ".pt"):
                # continue

                # Initialize the model for this run
                model_ft, input_size = initialize_model(
                    model_name, NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)
                model_ft = model_ft.to(DEVICE, non_blocking=True)

                params_to_update = model_ft.parameters()
                if FEATURE_EXTRACT:
                    params_to_update = []
                    for name, param in model_ft.named_parameters():
                        if param.requires_grad == True:
                            params_to_update.append(param)

                # Observe that all parameters are being optimized
                optimizer_ft = optim.SGD(
                    params_to_update, lr=LEARNING_RATE, momentum=MOMENTUM)

                # Setup the loss fxn
                criterion = nn.CrossEntropyLoss()

                # Train and evaluate
                setSeed()
                model_ft, scores_history = train_model(model_ft, nonmath_dict, criterion, optimizer_ft,
                                                    num_epochs=NUM_EPOCHS, is_inception=False,
                                                    delta=DELTA_ES, patience=PATIENCE_ES)
                
                torch.save({
                    'model': model_ft,
                    'task': currentTask,
                    'dataset': dataset_dir,
                    'learning_rate': LEARNING_RATE,
                    'momentum': MOMENTUM,
                    'balance': balance,
                    'model_name': model_name,
                    'batch_size': BATCH_SIZE,
                    'num_epochs': NUM_EPOCHS,
                    'criterion': criterion,
                    'optimizer': optimizer_ft,
                    'scores_history': scores_history,
                    'delta_es': DELTA_ES,
                    'patience_es': PATIENCE_ES
                }, this_model_save_path + ".pt")

                print("[💾 SAVED]", dataset_dir, model_name,
                    "/".join(str(b) for b in balance))
                
                # Free memory explicitly
                del model_ft, optimizer_ft
                torch.cuda.empty_cache()
            else:
                print('\t[✅ SKIPPING] ALREADY TRAINED')

            ##############################
            ### SALT & PEPPER TRAINING ###
            ##############################

            nonmath = {
                "SaltPepper_005": ("SaltPepper", "0.05")
            }

            nonmath_dirs = {
                name: os.path.join(ADVERSARIAL_DIR, dataset_dir, 'nonMath', attack_type, value)
                for name, (attack_type, value) in nonmath.items()
            }

            nonmath_datasets_dirs = {
                name: {x: BalancedDataset(dir_path,
                                        transform=data_transforms[x],
                                        balance=balance,
                                        check_images=False,
                                        use_cache=True) 
                    for x in ["train"]}
                for name, dir_path in nonmath_dirs.items()
            }

            nonmath_datasets = {
                "train": torch.utils.data.ConcatDataset(
                    [image_datasets["train"]] + [nonmath_datasets_dirs[name]["train"] for name in nonmath_datasets_dirs]
                ),
                "val": image_datasets["val"]
            }

            nonmath_dict = {x: torch.utils.data.DataLoader(
                nonmath_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY) for x in ["train", "val"]}

            print(f'\n\n[🤖 MODEL] SALT & PEPPER - {dataset_dir} - {model_name} - {balance}\n\n')

            this_model_save_path = model_save_path.replace(currentTask, f'{currentTask}/saltPepper')

            os.makedirs(os.path.dirname(this_model_save_path), exist_ok=True)
            if not os.path.exists(this_model_save_path + ".pt"):
                # continue

                # Initialize the model for this run
                model_ft, input_size = initialize_model(
                    model_name, NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)
                model_ft = model_ft.to(DEVICE, non_blocking=True)

                params_to_update = model_ft.parameters()
                if FEATURE_EXTRACT:
                    params_to_update = []
                    for name, param in model_ft.named_parameters():
                        if param.requires_grad == True:
                            params_to_update.append(param)

                # Observe that all parameters are being optimized
                optimizer_ft = optim.SGD(
                    params_to_update, lr=LEARNING_RATE, momentum=MOMENTUM)

                # Setup the loss fxn
                criterion = nn.CrossEntropyLoss()

                # Train and evaluate
                setSeed()
                model_ft, scores_history = train_model(model_ft, nonmath_dict, criterion, optimizer_ft,
                                                    num_epochs=NUM_EPOCHS, is_inception=False,
                                                    delta=DELTA_ES, patience=PATIENCE_ES)
                
                torch.save({
                    'model': model_ft,
                    'task': currentTask,
                    'dataset': dataset_dir,
                    'learning_rate': LEARNING_RATE,
                    'momentum': MOMENTUM,
                    'balance': balance,
                    'model_name': model_name,
                    'batch_size': BATCH_SIZE,
                    'num_epochs': NUM_EPOCHS,
                    'criterion': criterion,
                    'optimizer': optimizer_ft,
                    'scores_history': scores_history,
                    'delta_es': DELTA_ES,
                    'patience_es': PATIENCE_ES
                }, this_model_save_path + ".pt")

                print("[💾 SAVED]", dataset_dir, model_name,
                    "/".join(str(b) for b in balance))
                
                # Free memory explicitly
                del model_ft, optimizer_ft
                torch.cuda.empty_cache()
            else:
                print('\t[✅ SKIPPING] ALREADY TRAINED')

            ###############################
            ### GAUSSIAN NOISE TRAINING ###
            ###############################

            nonmath = {
                "GaussianNoise_005": ("GaussianNoise", "0.05")
            }

            nonmath_dirs = {
                name: os.path.join(ADVERSARIAL_DIR, dataset_dir, 'nonMath', attack_type, value)
                for name, (attack_type, value) in nonmath.items()
            }

            nonmath_datasets_dirs = {
                name: {x: BalancedDataset(dir_path,
                                        transform=data_transforms[x],
                                        balance=balance,
                                        check_images=False,
                                        use_cache=True) 
                    for x in ["train"]}
                for name, dir_path in nonmath_dirs.items()
            }

            nonmath_datasets = {
                "train": torch.utils.data.ConcatDataset(
                    [image_datasets["train"]] + [nonmath_datasets_dirs[name]["train"] for name in nonmath_datasets_dirs]
                ),
                "val": image_datasets["val"]
            }

            nonmath_dict = {x: torch.utils.data.DataLoader(
                nonmath_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY) for x in ["train", "val"]}

            print(f'\n\n[🤖 MODEL] GAUSSIAN - {dataset_dir} - {model_name} - {balance}\n\n')

            this_model_save_path = model_save_path.replace(currentTask, f'{currentTask}/gaussian')

            os.makedirs(os.path.dirname(this_model_save_path), exist_ok=True)
            if not os.path.exists(this_model_save_path + ".pt"):
                # continue

                # Initialize the model for this run
                model_ft, input_size = initialize_model(
                    model_name, NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)
                model_ft = model_ft.to(DEVICE, non_blocking=True)

                params_to_update = model_ft.parameters()
                if FEATURE_EXTRACT:
                    params_to_update = []
                    for name, param in model_ft.named_parameters():
                        if param.requires_grad == True:
                            params_to_update.append(param)

                # Observe that all parameters are being optimized
                optimizer_ft = optim.SGD(
                    params_to_update, lr=LEARNING_RATE, momentum=MOMENTUM)

                # Setup the loss fxn
                criterion = nn.CrossEntropyLoss()

                # Train and evaluate
                setSeed()
                model_ft, scores_history = train_model(model_ft, nonmath_dict, criterion, optimizer_ft,
                                                    num_epochs=NUM_EPOCHS, is_inception=False,
                                                    delta=DELTA_ES, patience=PATIENCE_ES)
                
                torch.save({
                    'model': model_ft,
                    'task': currentTask,
                    'dataset': dataset_dir,
                    'learning_rate': LEARNING_RATE,
                    'momentum': MOMENTUM,
                    'balance': balance,
                    'model_name': model_name,
                    'batch_size': BATCH_SIZE,
                    'num_epochs': NUM_EPOCHS,
                    'criterion': criterion,
                    'optimizer': optimizer_ft,
                    'scores_history': scores_history,
                    'delta_es': DELTA_ES,
                    'patience_es': PATIENCE_ES
                }, this_model_save_path + ".pt")

                print("[💾 SAVED]", dataset_dir, model_name,
                    "/".join(str(b) for b in balance))
                
                # Free memory explicitly
                del model_ft, optimizer_ft
                torch.cuda.empty_cache()
            else:
                print('\t[✅ SKIPPING] ALREADY TRAINED')

            ##########################
            ### SURROGATE TRAINING ###
            ##########################

            surrogate_names = [name for name in MODEL_NAMES if name != model_name]

            surrogate1_fgsm_data_dir = os.path.join(ADVERSARIAL_DIR, dataset_dir, 'math', 'FGSM', surrogate_names[0], "_".join(str(b) for b in balance), '0.2')
            surrogate1_pgd_data_dir = fgsm_data_dir.replace('FGSM', 'PGD')
            surrogate2_fgsm_data_dir = os.path.join(ADVERSARIAL_DIR, dataset_dir, 'math', 'FGSM', surrogate_names[1], "_".join(str(b) for b in balance), '0.2')
            surrogate2_pgd_data_dir = fgsm_data_dir.replace('FGSM', 'PGD')

            surrogate1_fgsm_image_datasets = {x: BalancedDataset(surrogate1_fgsm_data_dir,
                                                    transform=data_transforms[x],
                                                    balance=balance,
                                                    check_images=False,
                                                    use_cache=True) for x in ["train"]}
            surrogate1_pgd_image_datasets = {x: BalancedDataset(surrogate1_pgd_data_dir,
                                                    transform=data_transforms[x],
                                                    balance=balance,
                                                    check_images=False,
                                                    use_cache=True) for x in ["train"]}
            surrogate2_fgsm_image_datasets = {x: BalancedDataset(surrogate2_fgsm_data_dir,
                                                    transform=data_transforms[x],
                                                    balance=balance,
                                                    check_images=False,
                                                    use_cache=True) for x in ["train"]}
            surrogate2_pgd_image_datasets = {x: BalancedDataset(surrogate2_pgd_data_dir,
                                                    transform=data_transforms[x],
                                                    balance=balance,
                                                    check_images=False,
                                                    use_cache=True) for x in ["train"]}
            
            surrogate_datasets = {
                "train": torch.utils.data.ConcatDataset([image_datasets["train"],
                                                         surrogate1_fgsm_image_datasets["train"],
                                                         surrogate1_pgd_image_datasets["train"],
                                                         surrogate2_fgsm_image_datasets["train"],
                                                         surrogate2_pgd_image_datasets["train"]]),
                "val": image_datasets["val"]
            }

            surrogate_dict = {x: torch.utils.data.DataLoader(
                surrogate_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY) for x in ["train", "val"]}
            
            print(f'\n\n[🤖 MODEL] SURROGATE - {dataset_dir} - {model_name} - {balance}\n\n')

            this_model_save_path = model_save_path.replace(currentTask, f'{currentTask}/surrogate')

            os.makedirs(os.path.dirname(this_model_save_path), exist_ok=True)
            if not os.path.exists(this_model_save_path + ".pt"):
                # continue

                # Initialize the model for this run
                model_ft, input_size = initialize_model(
                    model_name, NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)
                model_ft = model_ft.to(DEVICE, non_blocking=True)

                params_to_update = model_ft.parameters()
                if FEATURE_EXTRACT:
                    params_to_update = []
                    for name, param in model_ft.named_parameters():
                        if param.requires_grad == True:
                            params_to_update.append(param)

                # Observe that all parameters are being optimized
                optimizer_ft = optim.SGD(
                    params_to_update, lr=LEARNING_RATE, momentum=MOMENTUM)

                # Setup the loss fxn
                criterion = nn.CrossEntropyLoss()

                # Train and evaluate
                setSeed()
                model_ft, scores_history = train_model(model_ft, surrogate_dict, criterion, optimizer_ft,
                                                    num_epochs=NUM_EPOCHS, is_inception=False,
                                                    delta=DELTA_ES, patience=PATIENCE_ES)
                
                torch.save({
                    'model': model_ft,
                    'task': currentTask,
                    'dataset': dataset_dir,
                    'learning_rate': LEARNING_RATE,
                    'momentum': MOMENTUM,
                    'balance': balance,
                    'model_name': model_name,
                    'batch_size': BATCH_SIZE,
                    'num_epochs': NUM_EPOCHS,
                    'criterion': criterion,
                    'optimizer': optimizer_ft,
                    'scores_history': scores_history,
                    'delta_es': DELTA_ES,
                    'patience_es': PATIENCE_ES
                }, this_model_save_path + ".pt")

                print("[💾 SAVED]", dataset_dir, model_name,
                    "/".join(str(b) for b in balance))
                
                # Free memory explicitly
                del model_ft, optimizer_ft
                torch.cuda.empty_cache()
            else:
                print('\t[✅ SKIPPING] ALREADY TRAINED')