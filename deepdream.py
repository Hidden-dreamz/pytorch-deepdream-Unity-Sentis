"""
    This file contains the implementation of the DeepDream algorithm,
    with an added option to export the underlying model to ONNX for Unity Sentis.
"""

import os
import argparse
import shutil
import time

import numpy as np
import torch
import cv2 as cv

import utils.utils as utils
from utils.constants import *
import utils.video_utils as video_utils


# Existing gradient ascent and DeepDream functions remain unchanged.
def gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration):
    out = model(input_tensor)
    activations = [out[layer_id_to_use] for layer_id_to_use in layer_ids_to_use]
    losses = []
    for layer_activation in activations:
        loss_component = torch.nn.MSELoss(reduction='mean')(layer_activation, torch.zeros_like(layer_activation))
        losses.append(loss_component)
    loss = torch.mean(torch.stack(losses))
    loss.backward()
    grad = input_tensor.grad.data
    sigma = ((iteration + 1) / config['num_gradient_ascent_iterations']) * 2.0 + config['smoothing_coefficient']
    smooth_grad = utils.CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)
    g_std = torch.std(smooth_grad)
    g_mean = torch.mean(smooth_grad)
    smooth_grad = smooth_grad - g_mean
    smooth_grad = smooth_grad / g_std
    input_tensor.data += config['lr'] * smooth_grad
    input_tensor.grad.data.zero_()
    input_tensor.data = torch.max(torch.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)


def deep_dream_static_image(config, img):
    model = utils.fetch_and_prepare_model(config['model_name'], config['pretrained_weights'], DEVICE)
    try:
        layer_ids_to_use = [model.layer_names.index(layer_name) for layer_name in config['layers_to_use']]
    except Exception as e:
        print(f'Invalid layer names {[layer_name for layer_name in config["layers_to_use"]]}.')
        print(f'Available layers for model {config["model_name"]} are {model.layer_names}.')
        return

    if img is None:
        img_path = utils.parse_input_file(config['input'])
        img = utils.load_image(img_path, target_shape=config['img_width'])
        if config['use_noise']:
            shape = img.shape
            img = np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32)

    img = utils.pre_process_numpy_img(img)
    base_shape = img.shape[:-1]
    for pyramid_level in range(config['pyramid_size']):
        new_shape = utils.get_new_shape(config, base_shape, pyramid_level)
        img = cv.resize(img, (new_shape[1], new_shape[0]))
        input_tensor = utils.pytorch_input_adapter(img, DEVICE)
        for iteration in range(config['num_gradient_ascent_iterations']):
            h_shift, w_shift = np.random.randint(-config['spatial_shift_size'], config['spatial_shift_size'] + 1, 2)
            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift)
            gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration)
            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift, should_undo=True)
        img = utils.pytorch_output_adapter(input_tensor)
    return utils.post_process_numpy_img(img)


def deep_dream_video_ouroboros(config):
    ts = time.time()
    assert any([config['input_name'].lower().endswith(img_ext) for img_ext in SUPPORTED_IMAGE_FORMATS]), \
        f'Expected an image, but got {config["input_name"]}. Supported image formats {SUPPORTED_IMAGE_FORMATS}.'
    utils.print_ouroboros_video_header(config)
    img_path = utils.parse_input_file(config['input'])
    frame = None if config['use_noise'] else utils.load_image(img_path, target_shape=config['img_width'])
    for frame_id in range(config['ouroboros_length']):
        print(f'Ouroboros iteration {frame_id+1}.')
        frame = deep_dream_static_image(config, frame)
        dump_path = utils.save_and_maybe_display_image(config, frame, name_modifier=frame_id)
        print(f'Saved ouroboros frame to: {os.path.relpath(dump_path)}\n')
        frame = utils.transform_frame(config, frame)
    video_utils.create_video_from_intermediate_results(config)
    print(f'time elapsed = {time.time()-ts} seconds.')


def deep_dream_video(config):
    video_path = utils.parse_input_file(config['input'])
    tmp_input_dir = os.path.join(OUT_VIDEOS_PATH, 'tmp_input')
    tmp_output_dir = os.path.join(OUT_VIDEOS_PATH, 'tmp_out')
    config['dump_dir'] = tmp_output_dir
    os.makedirs(tmp_input_dir, exist_ok=True)
    os.makedirs(tmp_output_dir, exist_ok=True)
    metadata = video_utils.extract_frames(video_path, tmp_input_dir)
    config['fps'] = metadata['fps']
    utils.print_deep_dream_video_header(config)
    last_img = None
    for frame_id, frame_name in enumerate(sorted(os.listdir(tmp_input_dir))):
        print(f'Processing frame {frame_id}')
        frame_path = os.path.join(tmp_input_dir, frame_name)
        frame = utils.load_image(frame_path, target_shape=config['img_width'])
        if config['blend'] is not None and last_img is not None:
            frame = utils.linear_blend(last_img, frame, config['blend'])
        dreamed_frame = deep_dream_static_image(config, frame)
        last_img = dreamed_frame
        dump_path = utils.save_and_maybe_display_image(config, dreamed_frame, name_modifier=frame_id)
        print(f'Saved DeepDream frame to: {os.path.relpath(dump_path)}\n')
    video_utils.create_video_from_intermediate_results(config)
    shutil.rmtree(tmp_input_dir)
    print(f'Deleted tmp frame dump directory {tmp_input_dir}.')


# NEW: Function to export the model to ONNX
def export_onnx_model(config):
    # Load model and set to evaluation mode
    model = utils.fetch_and_prepare_model(config['model_name'], config['pretrained_weights'], DEVICE)
    model.eval()
    # Create a dummy input tensor with shape [1, 3, H, W]
    dummy_input = torch.randn(1, 3, config['img_width'], config['img_width'], device=DEVICE)
    # Ensure the dump directory exists
    os.makedirs(config['dump_dir'], exist_ok=True)
    onnx_path = os.path.join(config['dump_dir'], 'deepdream.onnx')
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=11
    )
    print(f"Exported model to ONNX format: {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Common params
    parser.add_argument("--input", type=str, help="Input IMAGE or VIDEO name for dreaming", default='figures.jpg')
    parser.add_argument("--img_width", type=int, help="Resize input image to this width", default=600)
    parser.add_argument("--layers_to_use", type=str, nargs='+', help="Layers to maximize activations", default=['relu4_3'])
    parser.add_argument("--model_name", choices=[m.name for m in SupportedModels],
                        help="Neural network (model) to use", default=SupportedModels.VGG16_EXPERIMENTAL.name)
    parser.add_argument("--pretrained_weights", choices=[pw.name for pw in SupportedPretrainedWeights],
                        help="Pretrained weights to use", default=SupportedPretrainedWeights.IMAGENET.name)
    # Main params for experimentation
    parser.add_argument("--pyramid_size", type=int, help="Number of images in an image pyramid", default=4)
    parser.add_argument("--pyramid_ratio", type=float, help="Ratio of image sizes in the pyramid", default=1.8)
    parser.add_argument("--num_gradient_ascent_iterations", type=int, help="Number of gradient ascent iterations", default=10)
    parser.add_argument("--lr", type=float, help="Learning rate (step size)", default=0.09)
    # Video specific arguments
    parser.add_argument("--create_ouroboros", action='store_true', help="Create Ouroboros video (default False)")
    parser.add_argument("--ouroboros_length", type=int, help="Number of video frames in ouroboros video", default=30)
    parser.add_argument("--fps", type=int, help="Frames per second", default=30)
    parser.add_argument("--frame_transform", choices=[t.name for t in TRANSFORMS],
                        help="Transform used to feed output back to the network", default=TRANSFORMS.ZOOM_ROTATE.name)
    parser.add_argument("--blend", type=float, help="Blend coefficient for video creation", default=0.85)
    # Other common options
    parser.add_argument("--should_display", action='store_true', help="Display intermediate results (default False)")
    parser.add_argument("--spatial_shift_size", type=int, help="Pixels to randomly shift image before grad ascent", default=32)
    parser.add_argument("--smoothing_coefficient", type=float, help="Controls std deviation for gradient smoothing", default=0.5)
    parser.add_argument("--use_noise", action='store_true', help="Use noise as a starting point (default False)")
    # NEW: Option to export the model to ONNX
    parser.add_argument("--export_onnx", action='store_true', help="Export the model to ONNX for Unity Sentis integration")

    args = parser.parse_args()
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    # Determine dump directory based on use-case
    config['dump_dir'] = OUT_VIDEOS_PATH if config.get('create_ouroboros') else OUT_IMAGES_PATH
    config['dump_dir'] = os.path.join(config['dump_dir'], f'{config["model_name"]}_{config["pretrained_weights"]}')
    config['input_name'] = os.path.basename(config['input'])

    # If export flag is set, export the model and exit.
    if config.get("export_onnx"):
        export_onnx_model(config)
    # Otherwise, proceed with the original DeepDream functionalities.
    elif config.get('create_ouroboros'):
        deep_dream_video_ouroboros(config)
    elif any([config['input_name'].lower().endswith(video_ext) for video_ext in SUPPORTED_VIDEO_FORMATS]):
        deep_dream_video(config)
    else:
        print('Dreaming started!')
        img = deep_dream_static_image(config, img=None)
        dump_path = utils.save_and_maybe_display_image(config, img)
        print(f'Saved DeepDream static image to: {os.path.relpath(dump_path)}\n')
