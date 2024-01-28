import math

from einops import rearrange
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.model import Discriminator, Generator
from models.loss import GeneratorLoss

from scripts.dataloader import *
from .configParser import ConfigParser
from .utils import *


torch.autograd.set_detect_anomaly(True)


class SrganSR:
    def __init__(self, configs="train"):
        title("Initialize")
        self.configs = None
        self.epoch = None
        self.initConfigs(configs)
        self.initParams()

    def initConfigs(self, configs):
        self.configs = configs or self.configs
        self.configs = ConfigParser(self.configs).content
        mkdirs([PATHS.MODELS, PATHS.SCRIPTS, PATHS.SCRIPTS, PATHS.CONFIGS, PATHS.SHELLS, PATHS.CHECKPOINTS,
                PATHS.DATASETS])
        createFiles([PATHS.CONFIG_DEFAULT, PATHS.CONFIG_OVERRIDE])
        if self.configs["usegpu"] and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
            warn('Using CPU.')

    def trainEpochs(self, start, end):
        self.epoch = start
        self.endEpoch = end

        if self.pretrain:
            schedulerG = optim.lr_scheduler.StepLR(self.optimizerG, step_size=200, gamma=0.5)
            for i in range(start, end):
                self.epoch = i
                trainLoss, tranCorrect = self.epochAction("train", self.trainloader, generatorOnly=True)
                if (i + 1) % self.configs["saveEvery"] == 0:
                    self.save()
                #validLoss, validCorrect = self.epochAction("valid", self.validloader, generatorOnly=True)
                schedulerG.step()

        self.optimizerG = optim.Adam(self.modelG.parameters(), lr=1/2*self.configs["startLearningRate"])
        self.optimizerD = optim.Adam(self.modelD.parameters(), lr=1/2*self.configs["startLearningRate"])

        for epoch in range(start, end):
            if epoch == 400:
                self.optimizerG = optim.Adam(self.modelG.parameters(), lr=1/20*self.configs["startLearningRate"])
                self.optimizerD = optim.Adam(self.modelD.parameters(), lr=1/20*self.configs["startLearningRate"])
            if epoch == 600:
                self.optimizerG = optim.Adam(self.modelG.parameters(), lr=1/200*self.configs["startLearningRate"])
                self.optimizerD = optim.Adam(self.modelD.parameters(), lr=1/200*self.configs["startLearningRate"])
            self.epoch = epoch
            trainLoss, tranCorrect = self.epochAction("train", self.trainloader)
            self.trainLosses.append(trainLoss.item())
            if (epoch + 1) % self.configs["saveEvery"] == 0:
                self.save()

            validLoss, validCorrect = self.epochAction("valid", self.validloader)
            info(f'Valid loss: {round(validLoss.item(), 3)}')
            self.validLosses.append(validLoss.item())
            self.learningRates.append(self.learningRate)
            if validLoss < self.bestValidLoss:
                self.bestValidLoss = validLoss
                [best.unlink() for best in getFiles(self.getCheckpointFolder(), "best*.pth")]  # remove last best pth
                self.save(f"bestEpoch{epoch + 1}.pth")
                info(f"save best model, valid loss {round(validLoss.item(), 3)}")
            #self.schedulerG.step(validLoss)

    @property
    def learningRate(self):
        return self.optimizerG.param_groups[0]['lr']

    def modelForward(self, x, y):
        device = self.device
        x, y = map(lambda t: rearrange(t.to(device), 'b p c h w -> (b p) c h w'), (x, y))
        out = self.modelG(x)
        loss = self.criterion(out, y)
        return x, y, out, loss

    def epochAction(self, action, loader, generatorOnly=False):
        batch_size = self.batchSize
        running_results = {'batch_sizes': 0, 'd_loss': 0,
                           "g_loss": 0, "d_score": 0, "g_score": 0}
        isBackward = True if action == "train" else False
        GradSelection = Grad if isBackward else torch.no_grad
        totalLoss, totalCorrect, totalLen = 0, 0, 0
        batchLoader = tqdm(loader)

        mse_loss = torch.nn.MSELoss()

        if isBackward:
            self.modelG.train()
            self.modelD.train()
        else:
            self.modelG.eval()
            self.modelD.eval()

        with GradSelection():
            for x, y in batchLoader:
                device = self.device
                x, y = map(lambda t: rearrange(t.to(device), 'b p c h w -> (b p) c h w'), (x, y))

                if not generatorOnly:
                    for d_params in self.modelD.parameters():
                        d_params.requires_grad = True
                    self.modelD.zero_grad(set_to_none=True)
                    hr_d_output = self.modelD(y)
                    d_loss_hr = 1 - hr_d_output.mean()
                    if isBackward:
                        d_loss_hr.backward(retain_graph=True)

                    sr = self.modelG(x)
                    sr_d_output = self.modelD(sr.detach().clone())
                    d_loss_sr = sr_d_output.mean()
                    if isBackward:
                        d_loss_sr.backward()
                        self.optimizerD.step()
                    d_loss = d_loss_hr + d_loss_sr

                    for d_params in self.modelD.parameters():
                        d_params.requires_grad = False

                    self.modelG.zero_grad(set_to_none=True)
                    sr_d_output = self.modelD(sr.detach().clone())
                    g_loss = self.generatorCriterion(sr_d_output, sr, y)
                    if isBackward:
                        g_loss.backward()
                        self.optimizerG.step()

                else:
                    self.modelG.zero_grad(set_to_none=True)
                    sr = self.modelG(x)
                    g_loss = mse_loss(sr, y)
                    if isBackward:
                        g_loss.backward()
                        self.optimizerG.step()

                totalLoss += g_loss
                totalCorrect += torch.sum(y == sr)
                totalLen += len(x)
                running_results['batch_sizes'] += batch_size
                running_results['g_loss'] += g_loss.item() * batch_size

                if not generatorOnly:
                    running_results['d_loss'] += d_loss.item() * batch_size
                    running_results['d_score'] += (1 - d_loss_hr).item() * batch_size
                    running_results['g_score'] += d_loss_sr.item() * batch_size
                    batchLoader.set_description(desc="[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f " % (
                                                self.epoch + 1, self.configs['epochs'],
                                                running_results['d_loss'] / running_results['batch_sizes'],
                                                running_results['g_loss'] / running_results['batch_sizes'],
                                                running_results['d_score'] / running_results['batch_sizes'],
                                                running_results['g_score'] / running_results['batch_sizes']
                                                ) + f"Learning rate: {'%.1f' % (-math.log(self.learningRate, 10))} ")
                else:
                    batchLoader.set_description(desc=f'[{self.epoch}/{self.configs["epochs"]}] '
                        f"Learning rate: {'%.1f' % (-math.log(self.learningRate, 10))} "
                        f'Loss_G: {"%.4f" % (running_results["g_loss"] / running_results["batch_sizes"])}')

        return totalLoss / len(batchLoader), totalCorrect / len(batchLoader)

    def train(self, loader=None):
        title("Train")
        self.trainloader = loader or self.trainloader
        self.load()
        self.trainEpochs(self.startEpoch, self.configs["epochs"])

    def test(self):
        title("Test")
        self.loadBest()

    def saveObject(self, epoch):
        return {
            "epoch": epoch,
            "modelG": self.modelG.state_dict(),
            #"schedulerG": self.schedulerG.state_dict(),
            "optimizerG": self.optimizerG.state_dict(),
            "modelD": self.modelD.state_dict(),
            #"schedulerD": self.schedulerD.state_dict(),
            "optimizerD": self.optimizerD.state_dict(),
            "trainLosses": self.trainLosses,
            "validLosses": self.validLosses,
            "learningRates": self.learningRates
        }

    def getCheckpointFolder(self):
        return PATHS.CHECKPOINTS / f"X{self.configs['scaleFactor']}" / self.getModelName()

    def getModelName(self):
        return f"SRGAN-lr{self.configs['startLearningRate']}-flip{self.configs['randomFlip']}-psize" \
               f"{self.configs['patchSize']}"

    def save(self, fileName=""):
        epoch = self.epoch
        fileName = fileName or f"epoch{epoch + 1}.pth"
        saveFolder = self.getCheckpointFolder()
        mkdir(saveFolder)
        fileName = saveFolder / fileName
        torch.save(self.saveObject(epoch), fileName)

    def load(self):
        saveFolder = self.getCheckpointFolder()
        startEpoch = self.configs["startEpoch"]

        startEpoch = getFinalEpoch(saveFolder) if startEpoch == -1 else startEpoch  # get real last epoch if -1
        self.startEpoch = startEpoch
        if startEpoch == 0:
            return  # if 0 no load (including can't find )

        modelFile = getFile(saveFolder, f"epoch{startEpoch}.pth")
        self.loadParams(modelFile)

    def loadBest(self):
        modelFile = getFile(self.getCheckpointFolder(), "best*.pth")
        if modelFile:
            self.loadParams(modelFile)
        else:
            warn(f"best model not found under {self.getCheckpointFolder()}\nIs 'bestXXX.pth' exist?")
            self.load()

    def loadParams(self, fileP):
        info(f"load model from {fileP.name}")
        saveObject = torch.load(fileP)

        self.modelG.load_state_dict(saveObject["modelG"])
        #self.schedulerG.load_state_dict(saveObject["schedulerG"])
        self.optimizerG.load_state_dict(saveObject["optimizerG"])

        self.modelD.load_state_dict(saveObject["modelD"])
        #self.schedulerD.load_state_dict(saveObject["schedulerD"])
        self.optimizerD.load_state_dict(saveObject["optimizerD"])

        self.validLosses = saveObject["validLosses"]
        self.trainLosses = saveObject["trainLosses"]
        self.learningRates = saveObject["learningRates"]
        self.bestValidLoss = max([*self.validLosses, 0])

    def initParams(self):
        self.criterion = torch.nn.L1Loss()

        self.modelG = Generator(scale_factor=self.configs['scaleFactor'])
        self.modelG = self.modelG.to(self.device)
        self.optimizerG = optim.Adam(self.modelG.parameters(), lr=self.configs["startLearningRate"])
        #self.schedulerG = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG)

        self.modelD = Discriminator()
        self.modelD = self.modelD.to(self.device)
        self.optimizerD = optim.Adam(self.modelD.parameters(), lr=self.configs["startLearningRate"])
        #self.schedulerD = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD)

        self.generatorCriterion = GeneratorLoss()
        self.generatorCriterion = self.generatorCriterion.to(self.device)

        self.pretrain = self.configs["pretrainG"]
        self.trainLosses = []
        self.validLosses = []
        self.learningRates = []
        self.bestValidLoss = float("inf")
        self.batchSize = self.configs["batchSize"]
        self.trainDatasetPath = PATHS.DATASETS / self.configs["datasetPath"]
        self.patchSize = self.configs["patchSize"]

        self.trainDataset = DIV2KDataset(
            root_dir=self.trainDatasetPath,
            lr_scale=self.configs["scaleFactor"],
            crop_size=self.patchSize,
            is_training=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )
        self.validDataset = DIV2KDataset(
            root_dir=self.trainDatasetPath,
            lr_scale=self.configs["scaleFactor"],
            crop_size=self.patchSize,
            is_training=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )
        self.trainloader = DataLoader(
            self.trainDataset, batch_size=self.batchSize, shuffle=True, pin_memory=self.configs["pinMemory"],
            num_workers=self.configs["numWorkers"])
        self.validloader = DataLoader(
            self.validDataset, batch_size=self.batchSize, shuffle=True, pin_memory=self.configs["pinMemory"],
            num_workers=self.configs["numWorkers"])


if __name__ == '__main__':
    a = SrganSR("train")
