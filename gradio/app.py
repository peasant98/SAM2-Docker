import os
import cv2
import torch
import logging
import numpy as np
import gradio as gr; gr.close_all()

from typing import Tuple, Dict, Any, List
from pathlib import Path
from scipy.stats import mode
from einops import repeat
from skimage import color
from skimage.measure import label

import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor



CHECKPOINTS_TO_CFG = {
    "sam2_hiera_tiny.pt": "sam2_hiera_t.yaml",
    "sam2_hiera_small.pt": "sam2_hiera_s.yaml",
    "sam2_hiera_base_plus.pt": "sam2_hiera_b+.yaml",
    "sam2_hiera_large.pt": "sam2_hiera_l.yaml",
}


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def find_dots(image: np.ndarray, image_with_keypoints: np.ndarray) -> np.ndarray:
    # Check for diffs for any of (R, G, B)
    diff_mask = np.max(image != image_with_keypoints, axis=-1)
    # Assign each connected component as its own label
    diff_labels, num_labels = label(diff_mask, return_num=True)
    points = []
    # Find centre of mass of each keypoint
    for i in range(1, num_labels+1):
        com = np.argwhere(diff_labels == i).mean(axis=0)
        # predictor order is (x, y), numpy order is (y, x)
        points.append([com[1], com[0]])

    return np.array(points)


def create_color_mask(image: np.ndarray, annotations: Dict[str, Any]) -> np.ndarray:
    if len(annotations) == 0:
        return image

    sorted_anns = sorted(annotations, key=(lambda x: x['area']), reverse=True)
    mask = np.stack([x["segmentation"] for x in sorted_anns])
    mask = np.argmax(mask, axis=0)

    color_mask = color.label2rgb(mask, image)
    return color_mask


def extract_rgba_masks(
    image: np.ndarray,
    annotations: Dict[str, Any],
    mask_filter_area: float,
) -> List[np.ndarray]:
    image_segments = []
    for ann in annotations:
        if np.sum(ann["segmentation"]) >= mask_filter_area:
            segment = repeat(ann["segmentation"].astype(np.uint8) * 255, "h w -> h w 1")
            image_segment = np.concatenate([image, segment], axis=-1)
            image_segments.append(image_segment)

    return image_segments

@torch.no_grad()
def generate_sam2(
    predictor,
    image: np.ndarray,
    points_per_side: int,
    points_per_batch: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    stability_score_offset: float,
    box_nms_thresh: float,
    crop_n_layers: int,
    crop_nms_thresh: float,
    crop_overlap_ratio: float,
    crop_n_points_downscale_factor: float,
    min_mask_region_area: float,
    display_rgba_segments: bool,
    mask_filter_area: float,
    progress=gr.Progress(),
)   -> Tuple[np.ndarray, np.ndarray]:
    
    logger.info("Generating SAM2 masks")
    
    generator = SAM2AutomaticMaskGenerator(
        model=predictor.value["model"],
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=crop_overlap_ratio,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )
    
    progress(0, "Running inference")
    
    annotations = generator.generate(image)
    progress(0.5, "Making color mask")
    color_mask = create_color_mask(image, annotations)
    
    annotation_masks = []
    
    if display_rgba_segments:
        progress(0.75, "Extracting RGBA segments")
        annotation_masks = extract_rgba_masks(image, annotations, mask_filter_area)
        
    progress(1, "Returning masks")
    return color_mask, annotation_masks


def load_sam2_model(
    model_checkpoint: str,
    model_cfg: str,
    device: str,
):
    sam2 = build_sam2(model_cfg, model_checkpoint, device, apply_postprocessing=False)



def display_detected_keypoints(
    image: np.ndarray,
    fg_keypoints: np.ndarray,
    bg_keypoints: np.ndarray,
    bbox: np.ndarray,
):
    pos_diff_mask = np.max(image != fg_keypoints, axis=-1)
    _, num_positive = label(pos_diff_mask, return_num=True)

    neg_diff_mask = np.max(image != bg_keypoints, axis=-1)
    _, num_negative = label(neg_diff_mask, return_num=True)

    sections = [
        (pos_diff_mask, "Positive"),
        (neg_diff_mask, "Negative"),
    ]

    box = detect_boxes(image, bbox)
    if box is not None:
        sections.append((tuple(box), "Bounding Box"))

    return ((image, sections), num_positive, num_negative)


def detect_boxes(image: np.ndarray, image_with_boxes: np.ndarray) -> np.ndarray:
    diff_mask = np.sum(image != image_with_boxes, axis=-1) > 0
    if np.max(diff_mask) == 0:
        return None
    box_colors = image_with_boxes[diff_mask]

    # Get counts of each pixel color in the diff, select those only with a count above a threshold
    # The diff mask will contain colors that are anti-aliased, we want to ignore these
    box_color = mode(box_colors, axis=0, keepdims=True)[0][0]

    color_mask = np.sum(image_with_boxes == box_color, axis=-1) == 3
    bounding_mask = ((color_mask & diff_mask)).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(bounding_mask)
    box = np.array([x, y, x + w, y + h])
    return box


@torch.no_grad()
def guided_sam2_prediction(
    predictor,
    low_res_mask: np.ndarray,
    image: np.ndarray,
    fg_canvas: np.ndarray,
    bg_canvas: np.ndarray,
    box_canvas: np.ndarray,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[np.ndarray, np.ndarray]:
    
    
    progress(0, "Finding foreground keypoints", total=4)
    fg_points = find_dots(image, fg_canvas)
    progress(0.25, "Finding background keypoints", total=4)
    bg_points = find_dots(image, bg_canvas)
    progress(0.5, "Detecting bounding boxes", total=4)
    boxes = detect_boxes(image, box_canvas)

    if fg_points.size == 0:
        if bg_points.size == 0:
            point_coords = None
            point_labels = None
        else:
            point_coords = bg_points
            point_labels = np.zeros(len(bg_points))
    elif bg_points.size == 0:
        point_coords = fg_points
        point_labels = np.ones(len(fg_points))
    else:
        point_coords = np.concatenate([fg_points, bg_points])
        point_labels = np.concatenate([np.ones(len(fg_points)), np.zeros(len(bg_points))])

    progress(0.75, "Predicting masks", total=4)
    masks, scores, low_res_mask = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=low_res_mask,
        box=boxes,
        multimask_output=True,
    )

    # Mask input is 1 x H x W
    low_res_mask = low_res_mask[[np.argmax(scores)]]

    masks = masks.astype(int)
    colors = [[(1, 0, 0)], [(0, 1, 0)], [(0, 0, 1)]]

    color_masks = [color.label2rgb(masks[i], image, c) for i, c in enumerate(colors)]
    progress(1, "Returning masks", total=4)

    return color_masks, low_res_mask


@torch.no_grad()
def batch_sam2_predict(
    predictor,
    input_folder: str,
    dest_folder: str, 
    mask_suffix: str,
    points_per_side: int,
    points_per_batch: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    stability_score_offset: float,
    box_nms_thresh: float,
    crop_n_layers: int,
    crop_nms_thresh: float,
    crop_overlap_ratio: float,
    crop_n_points_downscale_factor: float,
    min_mask_region_area: float,
    progress: gr.Progress = gr.Progress(),
) -> List[np.ndarray]:
    
    
    if dest_folder is not None:
        Path(dest_folder).mkdir(exist_ok=True)
    image_files = [
        p.resolve() for p in Path(input_folder).rglob("**/*")
        if p.suffix in {".jpeg", ".png", ".jpg"}
    ]
    
    generator = SAM2AutomaticMaskGenerator(
        model=predictor.value["model"],
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=crop_overlap_ratio,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )

    masks = []
    for i, image_path in enumerate(image_files):
        progress(i / len(image_files), desc=f"Predicting {str(image_path)}", total=len(image_files))
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        annotations = generator.generate(image)
        color_mask = create_color_mask(image, annotations)
        if dest_folder is not None:
            dest_file = Path(dest_folder) / f"{image_path.stem}_{mask_suffix or ''}{image_path.suffix}"
            cv2.imwrite(str(dest_file), color_mask * 255)
        masks.append(color_mask)

    return masks


def set_sketch_images(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return image, image, image


def compute_image_embedding(predictor, image: np.ndarray):
    
    model = predictor.value["model"]
    model.set_image(image)
    # Reset best mask
    return None

def test():
    logger.info("Testing")
    return "Test function called!"


root_path = '/home/user/segment-anything-2'
checkpoints_dir = f"{root_path}/checkpoints"

available_models = sorted([x for x in os.listdir(checkpoints_dir) if x.endswith(".pt")])
default_model = available_models[1]

sam2_checkpoint = f"{root_path}/checkpoints/{default_model}"
model_cfg = CHECKPOINTS_TO_CFG[default_model]

device = "cuda" if torch.cuda.is_available() else "cpu"
default_predictor = build_sam2(model_cfg, sam2_checkpoint, device=device)

with gr.Blocks() as application:
    gr.Markdown(value="# Segment Anything 2 At Home")
    selected_model = gr.Dropdown(choices=available_models, label="Model",
                                 value=default_model, interactive=True)
    state = gr.State(value={"model": default_predictor})
    
    selected_model.change(lambda x: build_sam2(model_cfg, x, device), inputs=[
                          selected_model], outputs=[state], show_progress=True)
    with gr.Tab("Automatic Segmentor"):
        with gr.Row():
            with gr.Column():

                with gr.Accordion("Discrete Settings"):
                    with gr.Row():
                        points_per_side = gr.Number(label="points_per_side", value=32, precision=0)
                        points_per_batch = gr.Number(label="points_per_batch", value=64, precision=0)
                        stability_score_offset = gr.Number(label="stability_score_offset", value=1)
                        crop_n_layers = gr.Number(label="crop_n_layers", precision=0)
                        crop_n_points_downscale_factor = gr.Number(
                            label="crop_n_points_downscale_factor", value=1, precision=0)
                        min_mask_region_area = gr.Number(label="min_mask_region_area", precision=0, value=0)

                with gr.Accordion("Threshold Settings"):
                    pred_iou_thresh = gr.Slider(label="pred_iou_thresh", minimum=0, maximum=1, value=0.88, step=0.01)
                    stability_score_thresh = gr.Slider(label="stability_score_thresh",
                                                       minimum=0, maximum=1, value=0.95, step=0.01)
                    box_nms_thresh = gr.Slider(label="box_nms_thresh", minimum=0, maximum=1, value=0.7)
                    crop_nms_thresh = gr.Slider(label="crop_nms_thresh", minimum=0, maximum=1, value=0.7)
                    crop_overlap_ratio = gr.Slider(label="crop_overlap_ratio", minimum=0,
                                                   maximum=1, value=512 / 1500, step=0.01)

            with gr.Column():
                with gr.Tab("Single Prediction"):
                    # image = gr.Image(
                    #     source="upload",
                    #     label="Input Image",
                    #     elem_id="image",
                    #     brush_radius=20,
                    # )
                    with gr.Accordion("Instance Segment Export Settings"):
                        display_rgba_segments = gr.Checkbox(label="Extract RGBA image for each mask")
                        mask_filter_area = gr.Number(label="Segment Mask Area Filter", precision=0, value=0)

                    submit = gr.Button("Single Prediction")
                    output = gr.Image(interactive=False, label="Segmentation Map")
                    annotation_masks = gr.Gallery(label="Segment Images")
                    test_output = gr.Textbox(label="Test Output")  # New output component
                    # output = gr.Image(interactive=False, label="Segmentation Map")
                    # annotation_masks = gr.Gallery(label="Segment Images")
                    submit.click(test, inputs=[], outputs=[test_output])  # Updated outputs
                    # submit.click(generate_sam2, inputs=[
                    #     state,
                    #     image,
                    #     points_per_side,
                    #     points_per_batch,
                    #     pred_iou_thresh,
                    #     stability_score_thresh,
                    #     stability_score_offset,
                    #     box_nms_thresh,
                    #     crop_n_layers,
                    #     crop_nms_thresh,
                    #     crop_overlap_ratio,
                    #     crop_n_points_downscale_factor,
                    #     min_mask_region_area,
                    #     display_rgba_segments,
                    #     mask_filter_area,
                    # ], outputs=[output, annotation_masks])

                with gr.Tab("Batch Prediction"):
                    input_folder = gr.Textbox(label="Image Folder")
                    dest_folder = gr.Textbox(label="Output Folder")
                    mask_suffix = gr.Textbox(label="Mask Suffix", value="seg")
                    batch_predict_button = gr.Button(value="Batch Predict")
                    batch_outputs = gr.Gallery(label="Batch Outputs")
                    batch_predict_button.click(batch_sam2_predict, inputs=[
                        state,
                        input_folder,
                        dest_folder,
                        mask_suffix,
                        points_per_side,
                        points_per_batch,
                        pred_iou_thresh,
                        stability_score_thresh,
                        stability_score_offset,
                        box_nms_thresh,
                        crop_n_layers,
                        crop_nms_thresh,
                        crop_overlap_ratio,
                        crop_n_points_downscale_factor,
                        min_mask_region_area,
                    ], outputs=[batch_outputs])

    with gr.Tab("Predictor"):
        pred_instructions = """
        ## Instructions
        The predictor tab uses the `Sam2Predictor` class interface for doing inference. \
        This exists to allow users to insert foreground/background keypoints into the \
        image to guide the segmentation.

        1. Upload the image using the `Base Image` canvas. This will auto-populate the \
        foreground/background canvas
        2. Add Foreground/Background keypoints by placing dots on the respective canvases.
        3. (Optional) Draw a crude bounding box in the experimental box around an object. \
        The UI will determine a bounding rectangle which is then used for the `predict` interface. \
        **Note: currently supports only a single bounding box for now.**
        4. Click on the predict button.

        **Notes: Ensure that the color is not exactly the same as the pixels on the base image \
        where you are placing points. Any color or even multiple colors are allowed, so long \
        as it's different from the point on the base image. There's a heuristic algorithm that \
        locates the points by computing the diff of the base/keypoint images, taking the centre \
        of mass of each label. As such brush strokes/more complex shapes are treated as a single point.**
        """
        gr.Markdown(pred_instructions)

        with gr.Row():
            base_image = gr.Image(label="Base Image")

        with gr.Row():
            fg_canvas = gr.Image(label="Foreground Keypoints", interactive=True)
            bg_canvas = gr.Image(label="Background Keypoints", interactive=True)
            box_canvas = gr.Image(label="Box Canvas (Experimental)", interactive=True)

        with gr.Row():
            num_fg_keypoints = gr.Text(label="# Detected Foreground Keypoints", interactive=False)
            num_bg_keypoints = gr.Text(label="# Detected Background Keypoints", interactive=False)

        with gr.Row():
            annotated_canvas = gr.AnnotatedImage(label="Annotated Canvas",
                color_map={"Positive": "#46ff33", "Negative": "#ff3333", "Bounding Box": "#3361ff"}
            )
            output_masks = gr.Gallery(label="Output Masks", preview=True)

        predict = gr.Button("Predict")

        previous_mask = gr.State()
        base_image.upload(compute_image_embedding, inputs=[state, base_image], outputs=previous_mask)
        base_image.upload(set_sketch_images, inputs=[base_image], outputs=[fg_canvas, bg_canvas, box_canvas])

        for elem in [fg_canvas, bg_canvas, box_canvas]:
            elem.change(
                display_detected_keypoints,
                inputs=[base_image, fg_canvas, bg_canvas, box_canvas],
                outputs=[annotated_canvas, num_fg_keypoints, num_bg_keypoints],
            )
            elem.change(
                guided_sam2_prediction,
                inputs=[state, previous_mask, base_image, fg_canvas, bg_canvas, box_canvas],
                outputs=[output_masks, previous_mask],
            )

application.queue()

logging.info("Starting server")
application.launch(server_name="0.0.0.0", server_port=7860)
