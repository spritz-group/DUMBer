import os

import pandas as pd
from PIL import Image
import sys
import torch
from torch.utils.data import DataLoader
from torchattacks import FGSM, DeepFool, BIM, RFGSM, PGD, Square, TIFGSM
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision import transforms

from utils.balancedDataset import BalancedDataset
from utils.const import *
from utils.helperFunctions import *
from utils.nonMathAttacks import NonMathAttacks
from utils.tasks import currentTask

import warnings
warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

# Ranges and step for attack epsilon

ADVERSARIAL_DIR = f"./adversarialSamplesVal/{currentTask}"
DATASET_SIZE = 250

attacksParams = {
    "math": {
        "BIM": [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5],
        "DeepFool": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "FGSM": [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5],
        "PGD": [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5],
        "RFGSM": [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5],
        "Square": [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5],
        "TIFGSM": [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
    },
    "nonmath": {
        "BoxBlur": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "GaussianNoise": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        "GreyScale": [1],
        "InvertColor": [1],
        "RandomBlackBox": [20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
        "SaltPepper": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        "SplitMergeRGB": [1]
    }
}

nonMathAttacks = NonMathAttacks()

# Parameters

SHUFFLE_DATASET = False  # Shuffle the dataset

if not os.path.exists(os.path.join(os.getcwd(), ADVERSARIAL_DIR)):
    os.makedirs(os.path.join(os.getcwd(), ADVERSARIAL_DIR))

dfMath = pd.read_csv(MODEL_PREDICTIONS_PATH, index_col=[
                     "task", "model", "model_dataset", "balance", "dataset"]).sort_index()

# Helper functions

modelsEvals = []

datasetsToGenerate = getSubDirs(DATASETS_DIR)

i = 0

print("[🧠 MATH ATTACK GENERATION]\n")

for attack_name in attacksParams["math"].keys():
    currentAttackParams = attacksParams["math"][attack_name]

    csv_data = []
    for dataset in sorted(datasetsToGenerate):

        print("\n" + "-" * 15)
        print("[🗃️  SOURCE DATASET] {}\n".format(dataset))

        datasetDir = os.path.join(DATASETS_DIR, dataset)
        testDir = os.path.join(datasetDir, "val")

        datasetAdvDir = os.path.join(ADVERSARIAL_DIR, dataset)
        mathAttacksDir = os.path.join(datasetAdvDir, "math")

        if not os.path.exists(mathAttacksDir):
            os.makedirs(mathAttacksDir)

        toTensor = transforms.Compose([transforms.ToTensor()])
        toNormalizedTensor = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
        ])

        for root, _, fnames in sorted(os.walk(os.path.join(MODELS_DIR, dataset), followlinks=True)):
            for fname in sorted(fnames):
                # eps = currentAttackParams["init"]
                # effective = False
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

                justCreated = False

                # while not effective:
                for eps in currentAttackParams:
                    attacks = {
                        "BIM": BIM(model, eps=eps),
                        "DeepFool": DeepFool(model, overshoot=eps),
                        "FGSM": FGSM(model, eps=eps),
                        "PGD": PGD(model, eps=eps),
                        "RFGSM": RFGSM(model, eps=eps),
                        "Square": Square(model, eps=eps),
                        "TIFGSM": TIFGSM(model, eps=eps)
                    }
                    for attack in attacks:
                        if attack == attack_name:
                            attacker = attacks[attack]

                            attackDir = os.path.join(
                                mathAttacksDir, attack)
                            saveDir = os.path.join(
                                attackDir, modelName + "/" + modelPercents + "/" + str(eps))

                            print("\n[⚔️  ADVERSARIAL] {} @ {} - {} - {} {}".format(
                                attack,
                                eps,
                                modelDataset,
                                modelName,
                                modelPercents
                            ))

                            setSeed()

                            if not os.path.exists(saveDir):
                                os.makedirs(saveDir)
                                saveMathAdversarials(
                                    originalTestDataLoader, originalTestDataset.classes, attacker, saveDir)
                            elif sum([len([f for f in files if f.endswith('.jpg')]) for _, _, files in os.walk(saveDir)]) < (DATASET_SIZE*2):
                                saveMathAdversarials(
                                    originalTestDataLoader, originalTestDataset.classes, attacker, saveDir) 
                            else:
                                print(f"\t[💾 ALREADY SAVED]")
                


print("\n\n[🧠 NON-MATH ATTACK GENERATION]\n")

for attack_name in attacksParams["nonmath"].keys():
    currentAttackParams = attacksParams["nonmath"][attack_name]

    csv_data = []
    for dataset in sorted(datasetsToGenerate):

        print("\n" + "-" * 15)
        print("[🗃️  SOURCE DATASET] {}\n".format(dataset))

        datasetDir = os.path.join(DATASETS_DIR, dataset)
        testDir = os.path.join(datasetDir, "val")

        datasetAdvDir = os.path.join(ADVERSARIAL_DIR, dataset)
        nonMathAttacksDir = os.path.join(datasetAdvDir, "nonMath")

        if not os.path.exists(nonMathAttacksDir):
            os.makedirs(nonMathAttacksDir)

        toTensor = transforms.Compose([transforms.ToTensor()])
        toNormalizedTensor = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
        ])

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

        for eps in currentAttackParams:
            attacks = {
                "BoxBlur": nonMathAttacks.boxBlur,
                "GaussianNoise": nonMathAttacks.gaussianNoise,
                "GreyScale": nonMathAttacks.greyscale,
                "InvertColor": nonMathAttacks.invertColor,
                "RandomBlackBox": nonMathAttacks.randomBlackBox,
                "SaltPepper": nonMathAttacks.saltAndPepper,
                "SplitMergeRGB": nonMathAttacks.splitMergeRGB
            }
            for attack in attacks:
                if attack == attack_name:
                    print("\n[⚔️  ATTACKS] {} @ {}".format(
                        attack,
                        eps
                    ))

                    for path, cls in sorted(testDataset.imgs):
                        clsName = testDataset.classes[cls]
                        # saveDir = os.path.join(saveDir2, clsName)

                        imageName = os.path.basename(path)

                        image = Image.open(path).convert("RGB")

                        attacker = attacks[attack]

                        attackDir = os.path.join(
                            nonMathAttacksDir, attack)
                        saveDir2 = os.path.join(attackDir, str(eps))
                        # saveDir2 = os.path.join(saveDir, modelPercents + '/' + str(eps))
                        saveDir = os.path.join(saveDir2, clsName)

                        if not os.path.exists(saveDir):
                            os.makedirs(saveDir)

                        outImage = image.copy()
                        if attack != 'InvertColor' and attack != 'GreyScale' and attack != 'SplitMergeRGB':
                            outImage = attacker(outImage, amount=eps)
                        else:
                            outImage = attacker(outImage)
                        
                        outImage.save(os.path.join(
                            saveDir, imageName), "JPEG")

                    print(f"\t[💾 IMAGES SAVED]")

                    torch.cuda.empty_cache()