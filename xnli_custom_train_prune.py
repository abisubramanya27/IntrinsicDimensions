# Imports
import os
import time
import sys
import wandb
import numpy as np
import math
import torch
import datasets
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
import importlib
from transformers import AdamW
from transformers import get_scheduler
from torch.nn.utils import prune

os.environ["TOKENIZERS_PARALLELISM"] = 'true'
import sys
if '/home/indic-analysis/container/early-stopping-pytorch/' not in sys.path:
    sys.path.append('/home/indic-analysis/container/early-stopping-pytorch/')
EarlyStopping = importlib.import_module("early-stopping-pytorch.pytorchtools")

from torch.utils.cpp_extension import load

fwh_cuda = load(name='fwh_cuda',
                sources=['IntrinsicDimensions/id_fb_test/fwh_extension/fwh_cpp.cpp', 'IntrinsicDimensions/id_fb_test/fwh_extension/fwh_cu.cu'],
                verbose=True)

import torch
from torch.nn import functional as F
from fwh_cuda import fast_walsh_hadamard_transform as fast_walsh_hadamard_transform_cuda


## Fastfood Wrapper
class FastfoodWrap(nn.Module):
    def __init__(self, module, intrinsic_dimension, said=False, device=0):
        """
        Wrapper to estimate the intrinsic dimensionality of the
        objective landscape for a specific task given a specific model using FastFood transform
        :param module: pytorch nn.Module
        :param intrinsic_dimension: dimensionality within which we search for solution
        :param device: cuda device id
        """
        super(FastfoodWrap, self).__init__()

        # Hide this from inspection by get_parameters()
        self.m = [module]

        self.name_base_localname = []

        # Stores the initial value: \theta_{0}^{D}
        self.initial_value = dict()

        # Fastfood parameters
        self.fastfood_params = {}

        # SAID
        self.said = said
        self.said_size = len(list(module.named_parameters()))
        if self.said:
            assert intrinsic_dimension > self.said_size
            intrinsic_dimension -= self.said_size

        # Parameter vector that is updated
        # Initialised with zeros as per text: \theta^{d}
        intrinsic_parameter = nn.Parameter(torch.zeros((intrinsic_dimension)).to(device))
        self.register_parameter("intrinsic_parameter", intrinsic_parameter)
        v_size = (intrinsic_dimension,)

        length = 0
        # Iterate over layers in the module
        for name, param in module.named_parameters():
            # If param requires grad update
            if param.requires_grad:
                length += 1
                # Saves the initial values of the initialised parameters from param.data and sets them to no grad.
                # (initial values are the 'origin' of the search)
                self.initial_value[name] = v0 = (
                    param.clone().detach().requires_grad_(False).to(device)
                )

                # Generate fastfood parameters
                DD = np.prod(v0.size())
                self.fastfood_params[name] = fastfood_vars(DD, device)

                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))
                if "intrinsic_parameter" not in name:
                    param.requires_grad_(False)
            
        if said:
            intrinsic_parameter_said = nn.Parameter(torch.ones((length)).to(device))
            self.register_parameter("intrinsic_parameter_said", intrinsic_parameter_said)
            
        # for name, base, localname in self.name_base_localname:
        #     delattr(base, localname)

    def forward(self, x):
        index = 0
        # Iterate over layers
        for name, base, localname in self.name_base_localname:

            init_shape = self.initial_value[name].size()
            DD = np.prod(init_shape)

            # Fastfood transform te replace dence P
            ray = fastfood_torched(self.intrinsic_parameter, DD, self.fastfood_params[name]).view(
                init_shape
            )
            if self.said:
                ray = ray * self.intrinsic_parameter_said[index]
            param = self.initial_value[name] + ray

            delattr(base, localname)
            setattr(base, localname, param)
            index += 1

        # Pass through the model, by getting hte module from a list self.m
        module = self.m[0]
        x = module(x)
        return x

class FastWalshHadamard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(torch.tensor(
            [1 / np.sqrt(float(input.size(0)))]).to(input))
        if input.is_cuda:
            return fast_walsh_hadamard_transform_cuda(input.float(), False)
        else:
            return fast_walsh_hadamard_torched(input.float(), normalize=False)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if grad_output.is_cuda:
            return input*fast_walsh_hadamard_transform_cuda(grad_output.clone().float(), False).to(grad_output)
        else:
            return input*fast_walsh_hadamard_torched(grad_output.clone().float(), normalize=False).to(grad_output)

def fast_walsh_hadamard_torched(x, axis=0, normalize=False):
    """
    Performs fast Walsh Hadamard transform
    :param x:
    :param axis:
    :param normalize:
    :return:
    """
    orig_shape = x.size()
    assert axis >= 0 and axis < len(orig_shape), (
        "For a vector of shape %s, axis must be in [0, %d] but it is %d"
        % (orig_shape, len(orig_shape) - 1, axis)
    )
    h_dim = orig_shape[axis]
    h_dim_exp = int(round(np.log(h_dim) / np.log(2)))
    assert h_dim == 2 ** h_dim_exp, (
        "hadamard can only be computed over axis with size that is a power of two, but"
        " chosen axis %d has size %d" % (axis, h_dim)
    )

    working_shape_pre = [int(np.prod(orig_shape[:axis]))]  # prod of empty array is 1 :)
    working_shape_post = [
        int(np.prod(orig_shape[axis + 1 :]))
    ]  # prod of empty array is 1 :)
    working_shape_mid = [2] * h_dim_exp
    working_shape = working_shape_pre + working_shape_mid + working_shape_post

    ret = x.view(working_shape)

    for ii in range(h_dim_exp):
        dim = ii + 1
        arrs = torch.chunk(ret, 2, dim=dim)
        assert len(arrs) == 2
        ret = torch.cat((arrs[0] + arrs[1], arrs[0] - arrs[1]), axis=dim)

    if normalize:
        ret = ret / torch.sqrt(float(h_dim))

    ret = ret.view(orig_shape)

    return ret


def fastfood_vars(DD, device=0):
    """
    Returns parameters for fast food transform
    :param DD: desired dimension
    :return:
    """
    ll = int(np.ceil(np.log(DD) / np.log(2)))
    LL = 2 ** ll

    # Binary scaling matrix where $B_{i,i} \in \{\pm 1 \}$ drawn iid
    BB = torch.FloatTensor(LL).uniform_(0, 2).type(torch.LongTensor)
    BB = (BB * 2 - 1).type(torch.FloatTensor).to(device)
    BB.requires_grad = False

    # Random permutation matrix
    Pi = torch.LongTensor(np.random.permutation(LL)).to(device)
    Pi.requires_grad = False

    # Gaussian scaling matrix, whose elements $G_{i,i} \sim \mathcal{N}(0, 1)$
    GG = torch.FloatTensor(LL,).normal_().to(device)
    GG.requires_grad = False

    # Hadamard Matrix
    # HH = torch.tensor(hadamard(LL)).to(device)
    # HH.requirez_grad = False

    divisor = torch.sqrt(LL * torch.sum(torch.pow(GG, 2)))

    # return [BB, Pi, GG, HH, divisor, LL]
    return [BB, Pi, GG, divisor, LL]


def fastfood_torched(x, DD, param_list=None, device=0):
    """
    Fastfood transform
    :param x: array of dd dimension
    :param DD: desired dimension
    :return:
    """
    dd = x.size(0)

    if not param_list:

        BB, Pi, GG, divisor, LL = fastfood_vars(DD, device=device)

    else:

        BB, Pi, GG, divisor, LL = param_list

    # Padd x if needed
    dd_pad = F.pad(x, pad=(0, LL - dd), value=0, mode="constant")

    # From left to right HGPiH(BX), where H is Walsh-Hadamard matrix
    mul_1 = torch.mul(BB, dd_pad)

    # HGPi(HBX)
    mul_2 = fast_walsh_hadamard_torched(mul_1, 0, normalize=False)
    # mul2 = hadamard_torched_matmul(mul_1, 0, normalize=False)
    # mul_2 = torch.mul(HH, mul_1)
    # mul_2 = FastWalshHadamard.apply(mul_1)

    # HG(PiHBX)
    mul_3 = mul_2[Pi]

    # H(GPiHBX)
    mul_4 = torch.mul(mul_3, GG)

    # (HGPiHBX)
    # mul_5 = fast_walsh_hadamard_torched(mul_4, 0, normalize=False)
    mul_5 = FastWalshHadamard.apply(mul_4)

    ret = torch.div(mul_5[:DD], divisor * np.sqrt(float(DD) / LL))

    return ret


# Data
class DatasetBoi:
    def __init__(self, DATASET, CONFIG, MODEL_NAME, BATCH_SIZE, MAX_LENGTH, NUM_TRAIN_SAMPLES=-1, NUM_EVAL_SAMPLES=-1,  NUM_TEST_SAMPLES=-1):
        self.DATASET = DATASET
        self.CONFIG = CONFIG
        self.MODEL_NAME = MODEL_NAME
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_LENGTH = MAX_LENGTH
        self.NUM_TRAIN_SAMPLES = NUM_TRAIN_SAMPLES
        self.NUM_EVAL_SAMPLES = NUM_EVAL_SAMPLES
        self.NUM_TEST_SAMPLES = NUM_TEST_SAMPLES

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.train_dataset, self.eval_dataset, self.test_dataset = self.download_data()
        self.train_dataset, self.eval_dataset, self.test_dataset = self.preprocess_data()
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = self.get_dataloaders()

    def download_data(self):
        # Download data
        data = datasets.load_dataset(self.DATASET, self.CONFIG)
        print(data)
        train_dataset = data['train'].select(range(self.NUM_TRAIN_SAMPLES)) if self.NUM_TRAIN_SAMPLES > 0 else data['train']
        print('Training data length:', len(train_dataset))
        eval_dataset = data['validation'].select(range(self.NUM_EVAL_SAMPLES)) if self.NUM_EVAL_SAMPLES > 0 else data['validation']
        print('Validation data length:', len(eval_dataset))
        test_dataset = data['test'].select(range(self.NUM_TEST_SAMPLES)) if self.NUM_TEST_SAMPLES > 0 else data['test']
        print('Test data length:', len(test_dataset))

        return train_dataset, eval_dataset, test_dataset

    def preprocess_data(self):
        # Preprocessing
        train_dataset = self.train_dataset.map(self._tokenize, batched=True, batch_size=self.BATCH_SIZE)
        eval_dataset = self.eval_dataset.map(self._tokenize, batched=True, batch_size=self.BATCH_SIZE)
        test_dataset = self.test_dataset.map(self._tokenize, batched=True, batch_size=self.BATCH_SIZE)
        train_dataset = self._format_input(train_dataset)
        eval_dataset = self._format_input(eval_dataset)
        test_dataset = self._format_input(test_dataset)

        return train_dataset, eval_dataset, test_dataset


    def get_dataloaders(self):
        # Dataloades
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, num_workers=15, drop_last=False)
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.BATCH_SIZE, num_workers=15, drop_last=False)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, num_workers=15, drop_last=False)

        return train_dataloader, eval_dataloader, test_dataloader

    def _tokenize(self, batch):
        return self.tokenizer(batch['premise'], batch['hypothesis'], padding='max_length', truncation=True, max_length=self.MAX_LENGTH)
    
    def _format_input(self, dataset):
        dataset.set_format(type='torch', columns=['input_ids','label']) # Currently attention_mask and token_type_ids is being removed as fastfoodwrap accept>
        return dataset

# Pruning Function
def prune_model_global_unstructured(model, layer_type, proportion):
    module_tups = []
    for module in model.modules():
        if isinstance(module, layer_type):
            module_tups.append((module, 'weight'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=proportion
    )
    for module, _ in module_tups:
        prune.remove(module, 'weight')
    return model

# Model
class ModelBoi(nn.Module):
    def __init__(self, MODEL_NAME, FREEZE_FRACTION, ID, NUM_LABELS, device, said, prune=0):
        super(ModelBoi,self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, output_loading_info=False)
        print("Before Freezing- ")
        trainable_params, layers = self.trainable_stats()
        print("After Freezing- ")
        self.freeze_layers(layers*FREEZE_FRACTION)
        trainable_params, layers = self.trainable_stats()
        self.model.to(device)
        if prune > 0:
            prune_model_global_unstructured(self.model, torch.nn.Linear, prune)
        if ID:
            self.model = FastfoodWrap(self.model, intrinsic_dimension=ID, said=said, device=device)
            # self.model = intrinsic_dimension(self.model, ID, set())
            print("After fastfood - ")
            trainable_params, layers = self.trainable_stats()
        self.model.to(device)

    def trainable_stats(self):
        trainable_params = 0
        layers = 0
        for name, param in self.model.named_parameters():
            layers+=1
            # print(name, param.size())
            if param.requires_grad :
                trainable_params+=np.prod(param.size())

        print(f"Trainable params : {trainable_params} and layers : {layers}")
        return trainable_params, layers

    def freeze_layers(self, num_layers):
        for layer, (name, param) in enumerate(self.model.named_parameters()):
            if layer < num_layers:
                param.requires_grad = False
            else : break

        return

    def forward(self,inputs):
        outputs = self.model.forward(inputs)
        return outputs

# Data
class DatasetBoi2:
    def __init__(self, DATASET, CONFIG, MODEL_NAME, BATCH_SIZE, MAX_LENGTH, NUM_TRAIN_SAMPLES=-1, NUM_EVAL_SAMPLES=-1,  NUM_TEST_SAMPLES=-1):
        self.DATASET = DATASET
        self.CONFIG = CONFIG
        self.MODEL_NAME = MODEL_NAME
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_LENGTH = MAX_LENGTH
        self.NUM_TRAIN_SAMPLES = NUM_TRAIN_SAMPLES
        self.NUM_EVAL_SAMPLES = NUM_EVAL_SAMPLES
        self.NUM_TEST_SAMPLES = NUM_TEST_SAMPLES

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.eval_dataset, self.test_dataset = self.download_data()
        self.eval_dataset, self.test_dataset = self.preprocess_data()
        self.eval_dataloader, self.test_dataloader = self.get_dataloaders()

    def download_data(self):
        # Download data
        data = datasets.load_dataset(self.DATASET, self.CONFIG, split=['validation', 'test'])
        print(data)
        # train_dataset = data['train'].select(range(self.NUM_TRAIN_SAMPLES)) if self.NUM_TRAIN_SAMPLES > 0 else data['train']
        # print('Training data length:', len(train_dataset))
        eval_dataset = data[0].select(range(self.NUM_EVAL_SAMPLES)) if self.NUM_EVAL_SAMPLES > 0 else data[0]
        print('Validation data length:', len(eval_dataset))
        test_dataset = data[1].select(range(self.NUM_TEST_SAMPLES)) if self.NUM_TEST_SAMPLES > 0 else data[1]
        print('Test data length:', len(test_dataset))

        return eval_dataset, test_dataset

    def preprocess_data(self):
        # Preprocessing
        # train_dataset = self.train_dataset.map(self._tokenize, batched=True, batch_size=self.BATCH_SIZE)
        eval_dataset = self.eval_dataset.map(self._tokenize, batched=True, batch_size=self.BATCH_SIZE)
        test_dataset = self.test_dataset.map(self._tokenize, batched=True, batch_size=self.BATCH_SIZE)
        # train_dataset = self._format_input(train_dataset)
        eval_dataset = self._format_input(eval_dataset)
        test_dataset = self._format_input(test_dataset)

        return eval_dataset, test_dataset


    def get_dataloaders(self):
        # Dataloades
        # train_dataloader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, num_workers=15, drop_last=False)
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.BATCH_SIZE, num_workers=15, drop_last=False)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, num_workers=15, drop_last=False)

        return eval_dataloader, test_dataloader

    def _tokenize(self, batch):
        return self.tokenizer(batch['premise'], batch['hypothesis'], padding='max_length', truncation=True, max_length=self.MAX_LENGTH)
    
    def _format_input(self, dataset):
        dataset.set_format(type='torch', columns=['input_ids','label']) # Currently attention_mask and token_type_ids is being removed as fastfoodwrap accept>
        return dataset


# Training Loop
def train_loop_fn(model, train_dataloader, optimizer, scheduler, loss_fn, device, log_interval=50, wandb_log=False):
    model.train()

    train_loss = 0
    n_correct = 0
    train_start_time = time.time()
    n_samples = 0
    batch_train_loss = 0
    batch_samples = 0

    for batch_idx, batch in enumerate(train_dataloader):

        ## Dont need to send inputs and labels to device while using parallel loader as they are already sent to the right device
        inputs = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        pred = outputs.logits.argmax(dim=1, keepdim=True)
        n_correct += pred.eq(labels.view_as(pred)).sum().item()
        train_loss += loss.item() * len(inputs)
        n_samples += len(inputs)
        batch_train_loss += loss.item() * len(inputs)
        batch_samples += len(inputs)

        if (batch_idx+1)%log_interval == 0:
            # print(f"Batch Number: {batch_idx+1}\t\t Current Loss: {loss.item()}")
            if wandb_log and batch_samples > 0:
                wandb.log({"train_loss_step": batch_train_loss/batch_samples})
                batch_samples = 0
                batch_train_loss = 0

        if scheduler:
            scheduler.step()
    
    train_loss /= n_samples
    train_acc = n_correct*100.0 / n_samples

    if wandb_log and batch_samples > 0:
        wandb.log({"train_loss_step": batch_train_loss/batch_samples})

    return train_loss, train_acc

# Eval Loop
def eval_loop_fn(model, eval_dataloader, device, loss_fn, early_stop_callback):
    model.eval()
    eval_loss = 0
    num_correct = 0
    n_samples = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model.forward(inputs)
            eval_loss += loss_fn(outputs.logits, labels).item() * len(inputs)
            pred = outputs.logits.argmax(dim=1, keepdim=True)
            num_correct += pred.eq(labels.view_as(pred)).sum().item()
            n_samples += len(inputs)

    eval_loss /= n_samples
    eval_acc = num_correct*100 / n_samples

    if early_stop_callback:
        early_stop_callback(eval_loss, model)
        if  early_stop_callback.early_stop:
            print("Early stopping")
            return eval_loss, eval_acc, True

    return eval_loss, eval_acc, False

# Main Function
def main_fn(MODEL_NAME, DATASET, CONFIG, BATCH_SIZE, MAX_LENGTH, NUM_TRAIN_SAMPLES, NUM_EVAL_SAMPLES, NUM_LABELS, NUM_EPOCHS,
            LR, ID=0, said=False, prune=0, wandb_log=True, output_dir=None):
    said_str = "_SAID" if said else ''
    run_name = f"{MODEL_NAME}_ID{ID}_lr{LR}_ml{MAX_LENGTH}_pr{prune}"+said_str if ID>0 else f"{MODEL_NAME}_baseline_lr{LR}_ml{MAX_LENGTH}_pr{prune}"
    beta1, beta2 = 0.9, 0.999
    weight_decay, eps = 0.01, 1e-8
    scheduler_type = 'linear'

    # torch.set_default_tensor_type('torch.FloatTensor')
    device = torch.device('cuda')
    db = DatasetBoi(DATASET, CONFIG, MODEL_NAME, BATCH_SIZE, MAX_LENGTH, NUM_TRAIN_SAMPLES, NUM_EVAL_SAMPLES)
    model = ModelBoi(MODEL_NAME, FREEZE_FRACTION, ID, NUM_LABELS, device, said, prune)
    print(f'done loading model on {device}')

    # Optimizer and LR scheduler
    optimizer = AdamW(model.parameters(), lr=LR, betas = (beta1,beta2), weight_decay=weight_decay, eps=eps)

    # Callbacks
    early_stop_callback = EarlyStopping.EarlyStopping(patience=3,delta=0)

    warmup_steps = math.ceil(len(db.train_dataloader) * NUM_EPOCHS * 0.1)
    num_training_steps = NUM_EPOCHS * len(db.train_dataloader)
    lr_scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    if wandb_log:
        config = {
            'model_name': MODEL_NAME,
            'dataset': DATASET + '/' + CONFIG,
            'batch_size': BATCH_SIZE,
            'max_length': MAX_LENGTH,
            'lr': LR,
            'ID': ID,
            'mode': 'DID' if not said else 'SAID',
            'lr_scheduler': scheduler_type,
            'warmup_steps': warmup_steps,
            'optim': 'Adam',
            'beta1': beta1,
            'beta2': beta2,
            'weight_decay': weight_decay,
            'eps': eps,
            'prune_frac': prune
        }
        run = wandb.init(reinit=True, config=config, project=f'mbert-pruned-{DATASET}-{CONFIG}-{MAX_LENGTH}', entity='iitm-id', name=run_name, resume=None)

    train_start = time.time()
    for epoch in range(NUM_EPOCHS):
        epoch_time = time.time()
        train_loss, train_acc = train_loop_fn(model, db.train_dataloader, optimizer, lr_scheduler, loss_fn, device, 100, wandb_log)

        eval_time = time.time()
        print(f"\n[{round(eval_time-epoch_time,4)}s] Epoch elapsed: {epoch+1}\t\t Train Loss: {train_loss}\t\t Train Accuracy: {train_acc:.2f}%")
        
        eval_loss, eval_acc, early_stop = eval_loop_fn(model, db.eval_dataloader, device, loss_fn, early_stop_callback)
        print(f"[{round(time.time()-eval_time,4)}s] Epoch elapsed: {epoch+1}\t\t Eval Loss: {eval_loss}\t\t Eval Accuracy: {eval_acc:.2f}%")

        test_loss, test_acc, early_stop = eval_loop_fn(model, db.test_dataloader, device, loss_fn, early_stop_callback)
        print(f"[{round(time.time()-eval_time,4)}s] Epoch elapsed: {epoch+1}\t\t Test Loss: {test_loss}\t\t Test Accuracy: {test_acc:.2f}%")

        if wandb_log:
            wandb.log({"Train Loss": train_loss, "Train Accuracy": train_acc, "Eval Loss": eval_loss, "Eval Accuracy": eval_acc, "Test Loss":test_loss, "Test Accuracy":test_acc})

        print(f"Total time taken for epoch {epoch+1}: {round(time.time()-epoch_time,4)}s\n")

        if output_dir!=None and epoch == NUM_EPOCHS-1:
            output_path = os.path.join(output_dir, run_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_path = os.path.join(output_path, f'epoch_{epoch+1}.pth')
            torch.save({
                'model_state_dict': model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'epoch': epoch+1
            }, output_path)
            if wandb_log:
                artifact = wandb.Artifact(run_name, type='model')
                artifact.add_file(output_path, name=f'epoch_{epoch+1}.pth')
                run.log_artifact(artifact)
        
        if early_stop:
            break

    del model
    torch.cuda.empty_cache()
    print(f"Total time taken: {round(time.time()-train_start,4)}s")
    return

# Config
MODEL_NAME = "bert-base-multilingual-cased" #"bert-base-cased"  #"albert-base-v2"  "distilbert-base-multilingual-cased"    "albert-large-v2" "prajjwal1/bert-tiny"
NUM_LABELS = 3
DATASET = "xnli"
CONFIG = "en"
ID = 100
NUM_TRAIN_SAMPLES = -1
NUM_EVAL_SAMPLES = -1
BATCH_SIZE = 80
NUM_EPOCHS = 3
MAX_LENGTH = 256 # only 0.2 % of samples are > 256 size
LR = 1e-5
FREEZE_FRACTION = 0

ID_lr_dict = {
    # 0: 3e-5,
    # 100: 1e-3,
    # 500: 1e-3,
    # 1000: 1e-3,
    # 2000: 1e-3,
    # 5000: 1e-3,
    # 10000: 1e-3,
    # 12000: 1e-3,
    # 15000: 1e-3,
    # 18000: 1e-3,
    # 20000: 1e-3,
    # 35000: 1e-3,
    50000: 1e-3,
    75000: 1e-3,
    100000: 1e-3,
    200000: 5e-4,
    500000: 2e-4
}

# Prune knee point: 0.6
for ID in sorted(ID_lr_dict.keys()):
    main_fn(MODEL_NAME, DATASET, CONFIG, BATCH_SIZE, MAX_LENGTH, NUM_TRAIN_SAMPLES, NUM_EVAL_SAMPLES, NUM_LABELS, NUM_EPOCHS,
            ID_lr_dict[ID], int(ID), said=False, prune=0.6, wandb_log=True, output_dir="/home/indic-analysis/container/checkpoints_mbert_pruned_xnli/")
