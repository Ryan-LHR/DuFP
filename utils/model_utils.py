import time

import numpy as np
import torch
from tqdm import tqdm

from utils.eval import measure_execution_time
from utils.load_data.image.imagenet.convert_1k_to_100 import convert_model_to_im100
from utils.load_data.image.imagenet.in_utils import HFImgClsWrapper


def get_predictions(model, dataloader, return_type='Tensor', is_print=False,
                    data_name=None, device=None):
    """
    Get predictions from the input model

    Args:
        model (Model): the torch model
        dataloader (Dataloader): the dataloader
        return_type (str): ...
        is_print (bool): ...
        data_name (str)

    Returns:
        ...
    """
    start_time = time.time()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    prediction_vectors = []
    prediction_labels = []
    prediction_correct = []

    # Check whether inputs is tensor (for text task)
    input_probe, _ = next(iter(dataloader))
    input_is_tensor = torch.is_tensor(input_probe)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Predicting", total=len(dataloader)):
            if input_is_tensor:  # for general image and numerical tasks
                inputs, labels = inputs.to(device), labels.to(device)
            else:  # for text tasks
                labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            prediction_vectors.extend(outputs.detach().cpu())
            prediction_labels.extend(predicted.detach().cpu())

            correct = (predicted == labels)
            prediction_correct.extend(correct.detach().cpu())

    # print prediction results
    if is_print:
        accuracy = sum(prediction_correct) / len(prediction_correct)
        if data_name != None:
            print(f"\n---------Prediction of {data_name} set---------")
        print(f"\nPrediction vector for first input: {prediction_vectors[0]}"
              f"\nPrediction label for first input: {prediction_labels[0]}"
              f"\nPrediction correctness for first input: {prediction_correct[0]}"
              f"\nNumber of correct predictions: {sum(prediction_correct)}"
              f"\nOverall Accuracy: {accuracy*100:.2f}%"
              )
    end_time = time.time()
    if True:
    # if data_name == 'Test':
        exec_time_dict = measure_execution_time(start_time, end_time, stage='Inference Time')
        print(exec_time_dict)

    if return_type == 'Tensor':
        'return list with Tensor item'
        return (prediction_vectors, prediction_labels, prediction_correct)

    elif return_type == 'List':
        'return list with int/float item'
        prediction_vectors = [t.item() for t in prediction_vectors]
        prediction_labels = [t.item() for t in prediction_labels]
        prediction_correct = [t.item() for t in prediction_correct]

        return (prediction_vectors, prediction_labels, prediction_correct)

    elif return_type == 'Number':
        all_number = len(prediction_correct)
        correct_number = int(sum(prediction_correct))
        wrong_number = all_number - correct_number

        return (all_number, correct_number, wrong_number)


def extract_layer_output(model, dataloader, layer_name, device=None):
    """
    Extract Intermediate output of target layer

    Args:
        layer_name (str): The name of target layer.

    Returns:
        layer_output (np.ndarray): Intermediate output, with shape of (num_samples, num_features)。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_output = []

    # extract features through forward_feature
    if layer_name == "forward_feature":
        return extract_forward_feature(model, dataloader, device)

    def hook_fn(module, input, output):
        if layer_name == "hf_model.vit.layernorm":
            output = output[:, 0, :]  # [B, 768]
        elif layer_name == "hf_model.deit.layernorm":
            out_part1 = output[:, 0, :]
            out_part2 = output[:, 1, :]
            output = torch.cat([out_part1, out_part2], dim=1)  # [B, 768*2]

        layer_output.append(output.detach().cpu().numpy())

    # register hook
    hook = None
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
            break
    if hook is None:
        raise ValueError(f"Layer {layer_name} not found in the model.")

    input_probe, _ = next(iter(dataloader))
    input_is_tensor = torch.is_tensor(input_probe)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
                dataloader,
                total=len(dataloader),
                desc=f"Extracting intermediate output from {layer_name}",
        ):
            inputs, _ = batch
            if input_is_tensor:
                inputs = inputs.to(device)
            _ = model(inputs)

    hook.remove()

    layer_output = np.concatenate(layer_output, axis=0)
    return layer_output


def extract_forward_feature(model, dataloader, device):
    """
    Extract Intermediate output of target layer (by specific forward function)
    """

    if not (hasattr(model, "forward_feature") and callable(getattr(model, "forward_feature"))):
        raise ValueError("This model has not a callable forward_feature function")
    layer_output = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting intermediate output from forward_feature"):
            inputs, _ = batch
            inputs = inputs.to(device)

            features = model.forward_feature(inputs)
            layer_output.append(features.detach().cpu().numpy())

    layer_output = np.concatenate(layer_output, axis=0)
    return layer_output


def flatten_layer_output(x: np.ndarray):
    """
    Flatten Intermediate output.

    Args:
        x (np.ndarray): The intermediate output of a layer.

    Returns:
        layer_feature (np.ndarray): Flattened intermediate output,
            with shape of (num_samples, num_features).
    """
    num_samples = x.shape[0]
    layer_feature = x.reshape(num_samples, -1)
    return layer_feature


def get_model_parameters(model):
    """
    Get the parameter count of model
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def save_state_dict(model, path):
    """
    Save state_dict of given model to given path
    """
    torch.save(model.state_dict(), path)
    # usage: save_state_dict(model, args.model_file)
    return

def load_state_dict(model_class, path, device):
    """
    Load state_dict from path to given model
    """
    model = model_class()
    state_dict = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model

def load_model(model_name, model_class, path, device):
    """
    Load model from state dict or class object
    """

    # save state dict of new model

    if any(name in model_name for name in ["DBPedia", "AGNews"]):
        from transformers import AutoModelForSequenceClassification
        from utils.load_data.text.hg_dataset_utils import HFSeqClsWrapper
        # Load model directly (from local directory)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model = HFSeqClsWrapper(model)
        model = model.to(device)

        # print_forward(model)
        return model


    # load from class object
    if model_name in ['IM100-FastViT_T8', 'IM100-FastViT_S12', 'IM100-FastViT_S12_FT']:
        if model_name == 'IM100-FastViT_S12_FT':
            path = '/root/autodl-tmp/Projects/dimp/data/temp/dimp/retrain/finetuned_vit.pth'
            path = '/root/autodl-tmp/Projects/dimp/data/temp/dimp/retrain/finetuned_vit_neg_lr.pth'
        model = torch.load(path, map_location=torch.device(device))
        return model

    if "IM100Test" in model_name:
        import timm, os
        import transformers
        from transformers import AutoModel
        from transformers import AutoModelForImageClassification

        if model_name in ["IM100Test-deit_base_patch16_224", "IM100-DeiT"]:
            # name = "facebook/deit-base-patch16-224"
            local_dir = "/root/autodl-tmp/Projects/dimp/data/models/IM100Test/deit-base-patch16-224"


        elif model_name == "IM100Test-deit_base_distilled_patch16_224":
            # name = "facebook/deit-base-patch16-224"
            local_dir = "/root/autodl-tmp/Projects/dimp/data/models/IM100Test/deit-base-distilled-patch16-224"

        elif model_name == "IM100Test-deit_base_distilled_patch16_384":
            # name = "facebook/deit-base-patch16-224"
            local_dir = "/root/autodl-tmp/Projects/dimp/data/models/IM100Test/deit-base-distilled-patch16-384"

        elif model_name == "IM100Test-vit_large_patch14_clip_224":
            local_dir = "/root/autodl-tmp/Projects/dimp/data/models/IM100Test/vit_large_patch14_clip_224_openai_ft_in12k_in1k"
        elif model_name == "IM100Test-maxvit_base_512":
            name = "timm/maxvit_base_tf_512.in21k_ft_in1k"
            local_dir = "/root/autodl-tmp/Projects/dimp/data/models/IM100Test/maxvit_base_tf_512_in21k_ft_in1k"
            model = timm.create_model(
                name,
                pretrained=True,
                pretrained_cfg_overlay=dict(
                    file=os.path.join(local_dir, "pytorch_model.bin")  # 或 model.safetensors 对应的权重文件
                ),
            )
        elif model_name == "IM100Test-eva_large_patch14_196":
            local_dir = "/root/autodl-tmp/Projects/dimp/data/models/IM100Test/eva_large_patch14_196_in22k_ft_in22k_in1k"


        else:
            raise ValueError("model not found")

        # 通过transformer加载
        model = AutoModelForImageClassification.from_pretrained(
            local_dir,
            local_files_only=True,
            torch_dtype=torch.float32,
            use_safetensors=False,  # 强制走 pytorch_model.bin
        )
        model = HFImgClsWrapper(model)  # wrap huggingface model for our framework
        model = convert_model_to_im100(model, model_name=model_name)

        model = model.to(device)
        # print_forward(model)

    # load from state dict (for general models, e.g. LeNet5, VGG16, ResNet20)
    else:
        model = model_class()
        # state_dict = torch.load(path, map_location=torch.device(device))
        state_dict = torch.load(
            path,
            map_location=lambda storage, loc: storage.cpu()
        )
        model.load_state_dict(state_dict)
        model = model.to(device)

    # print_forward(model)
    return model


def print_forward(model):
    """
    Print forward function of model
    """
    if hasattr(model, "hf_model"):
        model = model.hf_model
    print(type(model))
    print(model.__class__.__name__)
    import inspect
    print(inspect.getsource(model.forward))

    if hasattr(model, "bert"):
        print(inspect.getsource(model.bert.forward))

    if hasattr(model, "classifier"):
        print(inspect.getsource(model.classifier.forward))

    return
