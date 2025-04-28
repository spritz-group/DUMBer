import numpy as np
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchattacks import FGSM, DeepFool, BIM, RFGSM, PGD, Square, TIFGSM
import torchvision
from torchvision import models, transforms

from utils.balancedDataset import BalancedDataset
from utils.const import *
from utils.helperFunctions import *
from utils.nonMathAttacks import NonMathAttacks

import warnings
warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

####################################################################################
####################################################################################
####################################################################################


MODEL_NAMES = ["alexnet", "resnet", "vgg"]

def baselineEvaluateModel(model, dataloader):
    model.eval()
    labelsOutputs = torch.tensor([]).to(DEVICE, non_blocking=True)

    for inputs, labels in dataloader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        labelsOutputs = torch.cat([labelsOutputs, preds], dim=0)

    return labelsOutputs

def evaluateModelF1(model, dataloader):
    model.eval()
    labelsOutputs = torch.tensor([]).to(DEVICE, non_blocking=True)
    labelsTargets = torch.tensor([]).to(DEVICE, non_blocking=True)

    for inputs, labels in dataloader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        labelsOutputs = torch.cat([labelsOutputs, preds], dim=0)
        labelsTargets = torch.cat([labelsTargets, labels], dim=0)

    acc, precision, recall, f1 = getScores(
        labelsTargets, labelsOutputs, complete=False)

    return {
        "acc": acc.cpu().numpy(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "f1": f1.cpu().numpy()
    }

def evaluateModelsOnDatasetBaseline(datasetFolder, datasetInfo, adv=False):
    modelsEvals = []

    # Setup for normalization
    dataTransform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
    ])

    testDataset = BalancedDataset(
        datasetFolder, transform=dataTransform, use_cache=True, check_images=False)

    setSeed()
    testDataLoader = DataLoader(
        testDataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    # Evaluate every model
    if not adv:
        for root, _, fnames in sorted(os.walk(MODELS_DIR, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)

                try:
                    modelData = torch.load(path)
                except:
                    continue

                modelDataset = modelData["dataset"]
                modelName = modelData["model_name"]
                modelPercents = "/".join([str(x)
                                        for x in modelData["balance"]])

                print()
                print("[🧮 EVALUATING] {} - {} {}".format(
                    modelDataset,
                    modelName,
                    modelPercents
                ))

                modelToTest = modelData["model"]
                modelToTest = modelToTest.to(DEVICE, non_blocking=True)

                scores = evaluateModelF1(modelToTest, testDataLoader)

                modelsEvals.append({
                    "source_dataset": datasetInfo["dataset"],
                    "target_model": modelName,
                    "target_dataset": modelDataset,
                    "target_balancing": modelPercents,
                    "baseline_f1": scores["f1"]
                })

                print("\tAcc: {:.4f}".format(scores["acc"]))
                print("\tPre: {:.4f}".format(scores["precision"]))
                print("\tRec: {:.4f}".format(scores["recall"]))
                print("\tF-Score: {:.4f}".format(scores["f1"]))

                torch.cuda.empty_cache()
    else:
        for root, _, fnames in sorted(os.walk(ADV_MODELS_DIR, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)

                try:
                    modelData = torch.load(path)
                except:
                    continue

                modelDataset = modelData["dataset"]
                modelName = modelData["model_name"]
                modelPercents = "/".join([str(x)
                                        for x in modelData["balance"]])
                modelTraining = path.split('/')[-4]

                print()
                print("[🧮 EVALUATING] {} - {} {} {}".format(
                    modelDataset,
                    modelName,
                    modelTraining,
                    modelPercents,
                ))

                modelToTest = modelData["model"]
                modelToTest = modelToTest.to(DEVICE, non_blocking=True)

                scores = evaluateModelF1(modelToTest, testDataLoader)

                modelsEvals.append({
                    "source_dataset": datasetInfo["dataset"],
                    "target_model": modelName,
                    "target_dataset": modelDataset,
                    "target_balancing": modelPercents,
                    "target_adv_training": modelTraining,
                    "baseline_f1": scores["f1"]
                })

                print("\tAcc: {:.4f}".format(scores["acc"]))
                print("\tPre: {:.4f}".format(scores["precision"]))
                print("\tRec: {:.4f}".format(scores["recall"]))
                print("\tF-Score: {:.4f}".format(scores["f1"]))

                torch.cuda.empty_cache()

    return modelsEvals

### GENERATING PREDICTIONS ###
print("\n\n" + "-" * 50)
print("\n[🧠 GENERATING MODEL PREDICTIONS]")

predictions = []

if not os.path.exists(MODEL_PREDICTIONS_PATH):
    for dataset in sorted(getSubDirs(DATASETS_DIR)):
        print("\n" + "-" * 15)
        print("[🗃️ DATASET] {}\n".format(dataset))

        datasetDir = os.path.join(DATASETS_DIR, dataset)
        testDir = os.path.join(datasetDir, "test")

        toTensor = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
        ])

        testDataset = BalancedDataset(
            testDir, transform=toTensor, use_cache=False, check_images=False)

        testDataLoader = DataLoader(testDataset, batch_size=16, shuffle=False)

        for root, _, fnames in sorted(os.walk(MODELS_DIR)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)

                modelData = torch.load(path)

                modelDataset = modelData["dataset"]
                modelName = modelData["model_name"]

                modelBalance = "/".join(str(x) for x in modelData["balance"])

                print("[🎖️ EVALUATING]", modelData["model_name"], modelBalance)

                modelToTest = modelData["model"]
                modelToTest = modelToTest.to(DEVICE, non_blocking=True)

                outputs = baselineEvaluateModel(modelToTest, testDataLoader)

                for (image, label), output in zip(testDataset.imgs, outputs):
                    predictions.append(
                        {
                            "task": currentTask,
                            "model": modelData["model_name"],
                            "model_dataset": modelData["dataset"],
                            "balance": modelBalance,
                            "dataset": dataset,
                            "image": Path(image),
                            "name": Path(image).name,
                            "label": label,
                            "prediction": int(output.cpu().numpy())
                        }
                    )

    predictionsDF = pd.DataFrame(predictions)

    if not os.path.exists(os.path.dirname('/'.join(MODEL_PREDICTIONS_PATH.split('.csv')[0].split('/')[:-1])+'/')):
        os.makedirs(os.path.dirname('/'.join(MODEL_PREDICTIONS_PATH.split('.csv')[0].split('/')[:-1])+'/'))

    predictionsDF.to_csv(MODEL_PREDICTIONS_PATH)
else:
    print(f"[⚔️  ADVERSARIAL] Predictions already generated, skipping...")
    predictionsDF = pd.read_csv(MODEL_PREDICTIONS_PATH)


print("\n\n" + "-" * 50)
print("\n[🧠 MODELS EVALUATION - BASELINE]")

if not os.path.exists(BASELINE_PATH):
    modelsEvals = []

    # Evaluate models on test folders
    for dataset in sorted(getSubDirs(DATASETS_DIR)):
        print("\n" + "-" * 15)
        print("[🗃️ TEST DATASET] {}".format(dataset))

        datasetDir = os.path.join(DATASETS_DIR, dataset)
        testDir = os.path.join(datasetDir, "test")

        advDatasetInfo = {
            "dataset": dataset,
            "math": None,
            "attack": None,
            "balancing": None,
            "model": None,
        }

        evals = evaluateModelsOnDatasetBaseline(testDir, advDatasetInfo)
        modelsEvals.extend(evals)

    modelsEvalsDF = pd.DataFrame(modelsEvals)

    if not os.path.exists(os.path.dirname('/'.join(BASELINE_PATH.split('.csv')[0].split('/')[:-1])+'/')):
        os.makedirs(os.path.dirname('/'.join(BASELINE_PATH.split('.csv')[0].split('/')[:-1])+'/'))

    modelsEvalsDF.to_csv(BASELINE_PATH)
else:
    print(f"[⚔️  ADVERSARIAL] Baseline already evaluated, skipping...")
    modelsEvalsDF = pd.read_csv(BASELINE_PATH)


print("\n\n" + "-" * 50)
print("\n[🧠 ADVERSARIAL MODELS EVALUATION - BASELINE]")

if not os.path.exists(ADV_BASELINE_PATH):
    modelsEvals = []

    # Evaluate models on test folders
    for dataset in sorted(getSubDirs(DATASETS_DIR)):
        print("\n" + "-" * 15)
        print("[🗃️ TEST DATASET] {}".format(dataset))

        datasetDir = os.path.join(DATASETS_DIR, dataset)
        testDir = os.path.join(datasetDir, "test")

        advDatasetInfo = {
            "dataset": dataset,
            "math": None,
            "attack": None,
            "balancing": None,
            "model": None,
        }

        evals = evaluateModelsOnDatasetBaseline(testDir, advDatasetInfo, adv=True)
        modelsEvals.extend(evals)

    modelsEvalsDF = pd.DataFrame(modelsEvals)

    if not os.path.exists(os.path.dirname('/'.join(ADV_BASELINE_PATH.split('.csv')[0].split('/')[:-1])+'/')):
        os.makedirs(os.path.dirname('/'.join(ADV_BASELINE_PATH.split('.csv')[0].split('/')[:-1])+'/'))

    modelsEvalsDF.to_csv(ADV_BASELINE_PATH)
else:
    print(f"[⚔️  ADVERSARIAL] Baseline already evaluated, skipping...")
    modelsEvalsDF = pd.read_csv(ADV_BASELINE_PATH)


### COMPUTING CLASS SIMILARITY ###
print("\n\n" + "-" * 50)
print("\n[🧠 MODELS EVALUATION - CLASS SIMILARITY]")

# Defining clean pre-trained models (not finetuned)
alexnet = models.alexnet(pretrained=True)
resnet = models.resnet18(pretrained=True)
vgg = models.vgg11_bn(pretrained=True)

models = [alexnet, resnet, vgg]

similarities = []

if not os.path.exists(SIMILARITY_PATH):
    for model, name in zip(models, MODEL_NAMES):
        for dataset in ['bing', 'google']:

            print(f'\n[🧮 EVALUATING] {name} - {dataset}')

            # Loading test set
            datasetDir = os.path.join(DATASETS_DIR, dataset)
            testDir = os.path.join(datasetDir, "test")

            toTensor = transforms.Compose([
                transforms.Resize(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
            ])

            testDataset = BalancedDataset(
                testDir, transform=toTensor, use_cache=False, check_images=False)

            setSeed()
            testDataLoader = DataLoader(
                testDataset, batch_size=16, shuffle=False)

            model = model.to(DEVICE, non_blocking=True)

            layer = model._modules.get('avgpool')

            def copy_embeddings(m, i, o):
                """
                Copy embeddings from the avgpool layer.
                """
                o = o[:, :, 0, 0].detach().cpu().numpy().tolist()
                outputs.append(o)

            outputs = []

            # Attach hook to avgpool layer
            _ = layer.register_forward_hook(copy_embeddings)

            model.eval()

            for X, y in testDataLoader:
                X = X.to(DEVICE, non_blocking=True)
                _ = model(X)

            list_embeddings = [item for sublist in outputs for item in sublist]
            embedding_size = len(list_embeddings[0])

            embeddings_0 = list_embeddings[:len(list_embeddings)//2]
            embeddings_1 = list_embeddings[len(list_embeddings)//2:]

            inter = []
            intra0 = []
            intra1 = []

            print(f'\t[⛏️ INTER] ', end='')
            for e0 in embeddings_0:
                for e1 in embeddings_1:
                    dist = np.linalg.norm(np.array(e0) - np.array(e1))
                    inter.append(dist)
            inter_dist = round(np.mean(inter)/embedding_size, 3)
            print(inter_dist)

            print(f'\t[⛏️ INTRA #0] ', end='')
            for i, e0_0 in enumerate(embeddings_0):
                for j, e0_1 in enumerate(embeddings_0):
                    if i != j:
                        dist = np.linalg.norm(np.array(e0_0) - np.array(e0_1))
                        intra0.append(dist)
            intra0_dist = round(np.mean(intra0)/embedding_size, 3)
            print(intra0_dist)

            print(f'\t[⛏️ INTRA #1] ', end='')
            for i, e1_0 in enumerate(embeddings_1):
                for j, e1_1 in enumerate(embeddings_1):
                    if i != j:
                        dist = np.linalg.norm(np.array(e1_0) - np.array(e1_1))
                        intra1.append(dist)
            intra1_dist = round(np.mean(intra1)/embedding_size, 3)
            print(intra1_dist)

            similarities.append({
                'dataset': dataset,
                'model': name,
                'inter': inter_dist,
                'intra0': intra0_dist,
                'intra1': intra1_dist
            })

    df = pd.DataFrame(similarities)
    if not os.path.exists(os.path.dirname('/'.join(SIMILARITY_PATH.split('.csv')[0].split('/')[:-1])+'/')):
        os.makedirs(os.path.dirname('/'.join(SIMILARITY_PATH.split('.csv')[0].split('/')[:-1])+'/'))
    df.to_csv(SIMILARITY_PATH)
else:
    print(f"[⚔️  ADVERSARIAL] Similarity already evaluated, skipping...")
    df = pd.read_csv(SIMILARITY_PATH)

####################################################################################
####################################################################################
####################################################################################

# Parameters

NON_MATH_ATTACKS = NonMathAttacks()

SHUFFLE_DATASET = False  # Shuffle the dataset

# Parameters for best eps estimation
ALPHA = 0.6
BETA = 1 - ALPHA

# If true, maximize gamma function
# If false, take eps which gives maximum ASR when SSIM is over a threshold
useGamma = False
threshold = 0.4

ADVERSARIAL_DIR = f"./adversarialSamplesTest/{currentTask}"

if not os.path.exists(os.path.join(os.getcwd(), ADVERSARIAL_DIR)):
    os.makedirs(os.path.join(os.getcwd(), ADVERSARIAL_DIR))

dfMath = pd.read_csv(MODEL_PREDICTIONS_PATH, index_col=[
                     "task", "model", "model_dataset", "balance", "dataset"]).sort_index()

# Setting seed for reproducibility

setSeed()

# Helper functions


def evaluateModelsOnDataset(datasetFolder, datasetInfo, models_dir=MODELS_DIR, isAdv=False):
    modelsEvals = []

    # Get the images and calculate mean and standard deviation
    imageDataset = torchvision.datasets.ImageFolder(
        datasetFolder, transform=transforms.Compose([transforms.ToTensor()]))

    for cls in imageDataset.classes:
        cls_index = imageDataset.class_to_idx[cls]
        num_cls = np.count_nonzero(
            np.array(imageDataset.targets) == cls_index)

        print("\t[🧮 # ELEMENTS] {}: {}".format(cls, num_cls))

    # Setup for normalization
    dataTransform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
    ])

    testDataset = BalancedDataset(
        datasetFolder, transform=dataTransform, use_cache=False, check_images=False, with_path=True)

    setSeed()
    testDataLoader = DataLoader(
        testDataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    # Evaluate every model
    for root, _, fnames in sorted(os.walk(models_dir, followlinks=True)):
        for fname in sorted(fnames):

            modelPath = os.path.join(root, fname)
            # print(modelPath)

            try:
                modelData = torch.load(modelPath)
            except:
                continue

            modelDataset = modelData["dataset"]
            modelName = modelData["model_name"]
            modelPercents = "/".join([str(x)
                                     for x in modelData["balance"]])

            print()
            if not isAdv:
                print("[🧮 EVALUATING] {} - {} {}".format(
                    modelDataset,
                    modelName,
                    modelPercents
                ))
            else:
                print("[🧮 EVALUATING] {} - {} {} {}".format(
                    modelDataset,
                    modelName,
                    modelPath.split('/')[-4],
                    modelPercents
                ))

            modelToTest = modelData["model"]
            modelToTest = modelToTest.to(DEVICE, non_blocking=True)

            scores = evaluateModel(
                modelToTest, testDataLoader, modelDataset, modelData, dfMath)

            if not isAdv:
                modelsEvals.append({
                    "source_dataset": datasetInfo["dataset"],
                    "isMath": datasetInfo["math"],
                    "attack": datasetInfo["attack"],
                    "source_model": datasetInfo["model"],
                    "source_balancing": datasetInfo["balancing"],

                    "target_model": modelName,
                    "target_dataset": modelDataset,
                    "target_balancing": modelPercents,
                    "asr": scores["asr"],
                    "asr_0": scores["asr_0"],
                    "asr_1": scores["asr_1"]
                })
            else:
                modelsEvals.append({
                    "source_dataset": datasetInfo["dataset"],
                    "isMath": datasetInfo["math"],
                    "attack": datasetInfo["attack"],
                    "source_model": datasetInfo["model"],
                    "source_balancing": datasetInfo["balancing"],
                    "source_adv_training": modelPath.split('/')[-4],

                    "target_model": modelName,
                    "target_dataset": modelDataset,
                    "target_balancing": modelPercents,
                    "asr": scores["asr"],
                    "asr_0": scores["asr_0"],
                    "asr_1": scores["asr_1"]
                })

            print("\t[ASR]: {:.4f}".format(scores["asr"]))
            print("\t\t[ASR_0]: {:.4f}".format(scores["asr_0"]))
            print("\t\t[ASR_1]: {:.4f}\n".format(scores["asr_1"]))

            torch.cuda.empty_cache()

    return modelsEvals


modelsEvals = []

datasetsToGenerate = getSubDirs(DATASETS_DIR)

i = 0

attacks_names = [
    'BIM',
    'BoxBlur',
    'DeepFool',
    'FGSM',
    'GaussianNoise',
    'GreyScale',
    'InvertColor',
    'PGD',
    'RandomBlackBox',
    'RFGSM',
    'SaltPepper',
    'SplitMergeRGB',
    'Square',
    'TIFGSM'
]

attacks_names_math = [
    'BIM',
    'DeepFool',
    'FGSM',
    'PGD',
    'RFGSM',
    'Square',
    'TIFGSM'
]

attack_names_static = [
    'GreyScale',
    'InvertColor',
    'SplitMergeRGB'
]

print("[🧠 GENERATING BEST EPS FOR EACH ATTACK]\n")

best_eps_data = []

eps_df_outputPath = os.path.join(HISTORY_DIR, 'all_eps.csv')

if not os.path.exists(eps_df_outputPath):
    for attack_name in attacks_names:
        for dataset in sorted(datasetsToGenerate):

            print("\n" + "-" * 15)
            print("[🗃️  SOURCE DATASET] {}\n".format(dataset))

            datasetDir = os.path.join(DATASETS_DIR, dataset)
            testDir = os.path.join(datasetDir, "test")

            datasetAdvDir = os.path.join(ADVERSARIAL_DIR, dataset)
            mathAttacksDir = os.path.join(datasetAdvDir, "math")
            nonMathAttacksDir = os.path.join(datasetAdvDir, "nonMath")

            if not os.path.exists(mathAttacksDir):
                os.makedirs(mathAttacksDir)
            if not os.path.exists(nonMathAttacksDir):
                os.makedirs(nonMathAttacksDir)

            toTensor = transforms.Compose([transforms.ToTensor()])
            toNormalizedTensor = transforms.Compose([
                transforms.Resize(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
            ])

            for root, _, fnames in sorted(os.walk(os.path.join(MODELS_DIR, dataset), followlinks=True)):
                for fname in sorted(fnames):
                    effective = False
                    asr_history = []

                    path = os.path.join(root, fname)

                    modelData = torch.load(path, map_location=torch.device('cpu'))

                    modelDataset = modelData["dataset"]
                    modelName = modelData["model_name"]

                    torch.cuda.empty_cache()

                    modelPercents = "_".join([str(x)
                                            for x in modelData["balance"]])
                    model = modelData["model"].to(DEVICE)

                    # Test dataset without normalization (for generating samples)
                    originalTestDataset = BalancedDataset(
                        testDir, transform=toTensor, datasetSize=DATASET_SIZE, use_cache=False, check_images=False, with_path=True)

                    setSeed()
                    originalTestDataLoader = DataLoader(
                        originalTestDataset, batch_size=16, num_workers=0, shuffle=SHUFFLE_DATASET)

                    # Test dataset with normalization (for evaluation)
                    testDataset = BalancedDataset(
                        testDir, transform=toNormalizedTensor, datasetSize=DATASET_SIZE, use_cache=False, check_images=False, with_path=True)

                    setSeed()
                    testDataLoader = DataLoader(
                        testDataset, batch_size=16, num_workers=0, shuffle=SHUFFLE_DATASET)

                    # Loading best epsilon value for this model
                    best_df = pd.read_csv(os.path.join(
                        HISTORY_DIR, attack_name + '.csv'), index_col='Unnamed: 0')

                    df_atk = best_df[best_df['model'] == modelName]
                    df_atk = df_atk[df_atk['dataset'] == modelDataset]
                    df_atk = df_atk[df_atk['balance'] == modelPercents]

                    epss = list(df_atk['eps'])
                    asrs = list(df_atk['asr'])
                    ssims = list(df_atk['ssim'])

                    best = []
                    max_eps_idx = 0
                    for j in range(len(epss)):
                        if useGamma:
                            best.append((ALPHA * asrs[j]) + (BETA * ssims[j]))
                        else:
                            if ssims[j] > threshold and asrs[j] >= asrs[max_eps_idx]:
                                eps = epss[j]
                                max_eps_idx = j

                    if useGamma:
                        maxx = max(best)
                        best_index = best.index(maxx)
                        eps = epss[best_index]

                    attacks = {
                        "BIM": BIM(model, eps=eps),
                        "BoxBlur": NON_MATH_ATTACKS.boxBlur,
                        "FGSM": FGSM(model, eps=eps),
                        "GaussianNoise": NON_MATH_ATTACKS.gaussianNoise,
                        "GreyScale": NON_MATH_ATTACKS.greyscale,
                        "InvertColor": NON_MATH_ATTACKS.invertColor,
                        "DeepFool": DeepFool(model, overshoot=eps),
                        "PGD": PGD(model, eps=eps),
                        "RandomBlackBox": NON_MATH_ATTACKS.randomBlackBox,
                        "RFGSM": RFGSM(model, eps=eps),
                        "SaltPepper": NON_MATH_ATTACKS.saltAndPepper,
                        "SplitMergeRGB": NON_MATH_ATTACKS.splitMergeRGB,
                        "Square": Square(model, eps=eps),
                        "TIFGSM": TIFGSM(model, eps=eps)
                    }

                    for attack in attacks:
                        if attack == attack_name:
                            # Mathematical attacks
                            if attack in attacks_names_math:
                                attacker = attacks[attack]

                                attackDir = os.path.join(
                                    mathAttacksDir, attack)
                                saveDir = os.path.join(
                                    attackDir, modelName + "/" + modelPercents)

                                if not os.path.exists(saveDir):
                                    os.makedirs(saveDir)

                                print("\n[⚔️  ADVERSARIAL] {} @ {} - {} - {} {}".format(
                                    attack,
                                    eps,
                                    modelDataset,
                                    modelName,
                                    modelPercents
                                ))

                                setSeed()
                                saveMathAdversarials(
                                    originalTestDataLoader, originalTestDataset.classes, attacker, saveDir)
                            # Non mathematical attacks of which a parameter have been grid-searched
                            elif attack not in attack_names_static:
                                print("[⚔️  ADVERSARIAL] {} @ {} - {} - {} {}".format(
                                    attack,
                                    eps,
                                    modelDataset,
                                    modelName,
                                    modelPercents
                                ))
                                for path, cls in sorted(testDataset.imgs):
                                    clsName = testDataset.classes[cls]

                                    imageName = os.path.basename(path)

                                    image = Image.open(path).convert("RGB")

                                    attacker = attacks[attack]

                                    attackDir = os.path.join(
                                        nonMathAttacksDir, attack)
                                    saveDir = os.path.join(attackDir, modelName)
                                    saveDir2 = os.path.join(saveDir, modelPercents)
                                    saveDir = os.path.join(saveDir2, clsName)

                                    if not os.path.exists(saveDir):
                                        os.makedirs(saveDir)

                                    outImage = image.copy()
                                    outImage = attacker(outImage, amount=eps)
                                    outImage.save(os.path.join(
                                        saveDir, imageName), "JPEG")

                                print(f"\t[💾 IMAGES SAVED]")

                            best_eps_data.append({
                                'attack': attack_name,
                                'model': modelName,
                                'dataset': modelDataset,
                                'balance': modelPercents,
                                'best_eps': eps
                            })

    eps_df = pd.DataFrame(best_eps_data)
    eps_df.to_csv(eps_df_outputPath)
else:
    print(f"[⚔️  ADVERSARIAL] Best eps already generated, skipping...")
    eps_df = pd.read_csv(eps_df_outputPath)


print("\n\n[🧠 ATTACKS EVALUATION]\n")

modelsEvals = []

for attack in sorted(attacks_names):

    modelsEvals_outputPath = os.path.join(EVALUATIONS_DIR, 'evaluations_' + attack + '.csv')

    if not os.path.exists(modelsEvals_outputPath):
        modelsEvals = []
        # Evaluate models on math attacks folders
        for dataset in sorted(getSubDirs(ADVERSARIAL_DIR)):
            datasetDir = os.path.join(ADVERSARIAL_DIR, dataset)
            mathAdvDir = os.path.join(datasetDir, "math")
            nonMathAdvDir = os.path.join(datasetDir, "nonMath")

            if not os.path.exists(mathAdvDir):
                continue

            if attack in attacks_names_math:
                attackDir = os.path.join(mathAdvDir, attack)
                isMath = True
            else:
                attackDir = os.path.join(nonMathAdvDir, attack)
                isMath = False

            for advModel in sorted(getSubDirs(attackDir)):
                advModelDir = os.path.join(attackDir, advModel)

                for advBalancing in sorted(getSubDirs(advModelDir)):
                    advDatasetDir = os.path.join(advModelDir, advBalancing)

                    print("\n" + "-" * 15)
                    print("[🗃️ ADVERSARIAL DATASET] {}/{}/{}/{}".format(dataset,
                        attack, advModel, advBalancing))

                    advDatasetInfo = {
                        "dataset": dataset,
                        "math": isMath,
                        "attack": attack,
                        "balancing": advBalancing.replace("_", "/"),
                        "model": advModel,
                    }

                    evals = evaluateModelsOnDataset(advDatasetDir, advDatasetInfo)
                    modelsEvals.extend(evals)

        modelsEvalsDF = pd.DataFrame(modelsEvals)

        if not os.path.exists(EVALUATIONS_DIR):
            os.makedirs(EVALUATIONS_DIR)

        modelsEvalsDF.to_csv(modelsEvals_outputPath)
    else:
        print(f"[⚔️  ADVERSARIAL] {attack} already evaluated, skipping...")
        modelsEvalsDF = pd.read_csv(modelsEvals_outputPath)

###########################################################################################################################

print("\n\n[🧠 ATTACKS EVALUATION ON ADVERSARIALLY TRAINED MODELS]\n")

modelsEvals = []

for attack in sorted(attacks_names):

    modelsEvals_outputPath = os.path.join(ADV_EVALUATIONS_DIR, 'evaluations_' + attack + '.csv')

    if not os.path.exists(modelsEvals_outputPath):
        modelsEvals = []
        # Evaluate models on math attacks folders
        for dataset in sorted(getSubDirs(ADVERSARIAL_DIR)):
            datasetDir = os.path.join(ADVERSARIAL_DIR, dataset)
            mathAdvDir = os.path.join(datasetDir, "math")
            nonMathAdvDir = os.path.join(datasetDir, "nonMath")

            if not os.path.exists(mathAdvDir):
                continue

            if attack in attacks_names_math:
                attackDir = os.path.join(mathAdvDir, attack)
                isMath = True
            else:
                attackDir = os.path.join(nonMathAdvDir, attack)
                isMath = False

            for advModel in sorted(getSubDirs(attackDir)):
                advModelDir = os.path.join(attackDir, advModel)

                for advBalancing in sorted(getSubDirs(advModelDir)):
                    advDatasetDir = os.path.join(advModelDir, advBalancing)

                    print("\n" + "-" * 15)
                    print("[🗃️ ADVERSARIAL DATASET] {}/{}/{}/{}".format(dataset,
                        attack, advModel, advBalancing))

                    advDatasetInfo = {
                        "dataset": dataset,
                        "math": isMath,
                        "attack": attack,
                        "balancing": advBalancing.replace("_", "/"),
                        "model": advModel,
                    }

                    evals = evaluateModelsOnDataset(advDatasetDir, advDatasetInfo, models_dir=ADV_MODELS_DIR, isAdv=True)
                    modelsEvals.extend(evals)

        modelsEvalsDF = pd.DataFrame(modelsEvals)

        if not os.path.exists(ADV_EVALUATIONS_DIR):
            os.makedirs(ADV_EVALUATIONS_DIR)

        modelsEvalsDF.to_csv(modelsEvals_outputPath)
    else:
        print(f"[⚔️  ADVERSARIAL] {attack} already evaluated, checking for updates...")
        modelsEvalsDF = pd.read_csv(modelsEvals_outputPath)

        existing_adv_training_types = set(modelsEvalsDF["source_adv_training"].unique())

        all_training_types = set(getSubDirs(ADV_MODELS_DIR))

        new_training_types = all_training_types - existing_adv_training_types

        new_evals = []
        for training_type in sorted(new_training_types):
            new_adv_models_dir = os.path.join(ADV_MODELS_DIR, training_type)
            # Determine which datasets need to be updated
            for dataset in sorted(getSubDirs(ADVERSARIAL_DIR)):
                datasetDir = os.path.join(ADVERSARIAL_DIR, dataset)
                mathAdvDir = os.path.join(datasetDir, "math")
                nonMathAdvDir = os.path.join(datasetDir, "nonMath")

                if not os.path.exists(mathAdvDir):
                    continue

                if attack in attacks_names_math:
                    attackDir = os.path.join(mathAdvDir, attack)
                    isMath = True
                else:
                    attackDir = os.path.join(nonMathAdvDir, attack)
                    isMath = False

                for advModel in sorted(getSubDirs(attackDir)):
                    advModelDir = os.path.join(attackDir, advModel)

                    for advBalancing in sorted(getSubDirs(advModelDir)):
                        advDatasetDir = os.path.join(advModelDir, advBalancing)

                        print("\n" + "-" * 15)
                        print("[🗃️ ADVERSARIAL DATASET] {}/{}/{}/{}".format(dataset,
                            attack, advModel, advBalancing))

                        advDatasetInfo = {
                            "dataset": dataset,
                            "math": isMath,
                            "attack": attack,
                            "balancing": advBalancing.replace("_", "/"),
                            "model": advModel,
                        }

                        evals = evaluateModelsOnDataset(advDatasetDir, advDatasetInfo, models_dir=new_adv_models_dir, isAdv=True)
                        new_evals.extend(evals)

        if new_evals:
            newEvalsDF = pd.DataFrame(new_evals)
            modelsEvalsDF = pd.concat([modelsEvalsDF, newEvalsDF], ignore_index=True)
            modelsEvalsDF.to_csv(modelsEvals_outputPath, index=False)
            print(f"[✅ UPDATED] Added {len(new_evals)} new evaluations to {modelsEvals_outputPath}")
        else:
            print(f"[✔️  UP-TO-DATE] No new training types found for {attack}")