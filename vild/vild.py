from easydict import EasyDict
import numpy as np
import clip
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from PIL import Image
from scipy.special import softmax
import tensorflow.compat.v1 as tf

from utils.text_utils import single_template, multiple_templates, article, processed_name, build_text_embedding
from utils.func_utils import nms
from utils.vis_utils import *

def vild(image_path, 
        category_name_string, 
        params, 
        session, 
        clip_model,
        numbered_category_indices,
        FLAGS, 
        overall_fig_size, 
        save_path,
        ):
    #################################################################
    # Preprocessing categories and get params
    category_names = [x.strip() for x in category_name_string.split(';')]
    category_names = ['background'] + category_names
    categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(category_names)]
    category_indices = {cat['id']: cat for cat in categories}
    
    max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area = params
    fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)


    #################################################################
    # Obtain results and read image
    roi_boxes, roi_scores, detection_boxes, scores_unused, box_outputs, detection_masks, visual_features, image_info = session.run(
            ['RoiBoxes:0', 'RoiScores:0', '2ndStageBoxes:0', '2ndStageScoresUnused:0', 'BoxOutputs:0', 'MaskOutputs:0', 'VisualFeatOutputs:0', 'ImageInfo:0'],
            feed_dict={'Placeholder:0': [image_path,]})
    
    roi_boxes = np.squeeze(roi_boxes, axis=0)  # squeeze
    # no need to clip the boxes, already done
    roi_scores = np.squeeze(roi_scores, axis=0)

    detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
    scores_unused = np.squeeze(scores_unused, axis=0)
    box_outputs = np.squeeze(box_outputs, axis=0)
    detection_masks = np.squeeze(detection_masks, axis=0)
    visual_features = np.squeeze(visual_features, axis=0)

    image_info = np.squeeze(image_info, axis=0)  # obtain image info
    image_scale = np.tile(image_info[2:3, :], (1, 2))
    image_height = int(image_info[0, 0])
    image_width = int(image_info[0, 1])

    rescaled_detection_boxes = detection_boxes / image_scale # rescale

    # Read image
    image = np.asarray(Image.open(open(image_path, 'rb')).convert("RGB"))
    assert image_height == image.shape[0]
    assert image_width == image.shape[1]


    #################################################################
    # Filter boxes

    # Apply non-maximum suppression to detected boxes with nms threshold.
    nmsed_indices = nms(
        detection_boxes,
        roi_scores,
        thresh=nms_threshold
        )

    # Compute RPN box size.
    box_sizes = (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * (rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])

    # Filter out invalid rois (nmsed rois)
    valid_indices = np.where(
        np.logical_and(
            np.isin(np.arange(len(roi_scores), dtype=np.int32), nmsed_indices),
            np.logical_and(
                np.logical_not(np.all(roi_boxes == 0., axis=-1)),
                np.logical_and(
                roi_scores >= min_rpn_score_thresh,
                box_sizes > min_box_area
                )
            )    
        )
    )[0]
    print('number of valid indices', len(valid_indices))

    detection_roi_scores = roi_scores[valid_indices][:max_boxes_to_draw, ...]
    detection_boxes = detection_boxes[valid_indices][:max_boxes_to_draw, ...]
    detection_masks = detection_masks[valid_indices][:max_boxes_to_draw, ...]
    detection_visual_feat = visual_features[valid_indices][:max_boxes_to_draw, ...]
    rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][:max_boxes_to_draw, ...]


    #################################################################
    # Compute text embeddings and detection scores, and rank results
    text_features = build_text_embedding(categories, FLAGS, clip_model)
    raw_scores = detection_visual_feat.dot(text_features.T)
    if FLAGS.use_softmax:
        scores_all = softmax(FLAGS.temperature * raw_scores, axis=-1)
    else:
        scores_all = raw_scores

    # indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores
    # indices_fg = np.array([i for i in indices if np.argmax(scores_all[i]) != 0])
    indices_fg = np.array([np.argmax(scores_all[:,i]) for i in range(1, len(category_names))])
    # indices_fg = np.array([1])

    #################################################################
    # Plot detected boxes on the input image.
    ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
    processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
    segmentations = paste_instance_masks(detection_masks, processed_boxes, image_height, image_width)

    if len(indices_fg) == 0:
        display_image(np.array(image), size=overall_fig_size)
        print('ViLD does not detect anything belong to the given category')

    else:
        image_with_detections = visualize_boxes_and_labels_on_image_array(
            np.array(image),
            rescaled_detection_boxes[indices_fg],
            valid_indices[:max_boxes_to_draw][indices_fg],
            detection_roi_scores[indices_fg],    
            numbered_category_indices,
            instance_masks=segmentations[indices_fg],
            use_normalized_coordinates=False,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_rpn_score_thresh,
            skip_scores=False,
            skip_labels=True)

        plt.figure(figsize=overall_fig_size)
        plt.imshow(image_with_detections)
        plt.axis('off')
        plt.title('Detected objects and RPN scores')
        plt.show()
        plt.savefig(save_path)

    return valid_indices[:max_boxes_to_draw][indices_fg], rescaled_detection_boxes[indices_fg], segmentations[indices_fg]

    # #################################################################
    # # Plot
    # cnt = 0
    # raw_image = np.array(image)
    # n_boxes = rescaled_detection_boxes.shape[0]

    # for anno_idx in indices[0:int(n_boxes)]:
    #   rpn_score = detection_roi_scores[anno_idx]
    #   bbox = rescaled_detection_boxes[anno_idx]
    #   scores = scores_all[anno_idx]
    #   if np.argmax(scores) == 0:
    #     continue
        
    #   y1, x1, y2, x2 = int(np.floor(bbox[0])), int(np.floor(bbox[1])), int(np.ceil(bbox[2])), int(np.ceil(bbox[3]))
    #   img_w_mask = plot_mask(mask_color, alpha, raw_image, segmentations[anno_idx])
    #   crop_w_mask = img_w_mask[y1:y2, x1:x2, :]


    #   fig, axs = plt.subplots(1, 4, figsize=(fig_size_w, fig_size_h), gridspec_kw={'width_ratios': [3, 1, 1, 2]}, constrained_layout=True)

    #   # Draw bounding box.
    #   rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=line_thickness, edgecolor='r', facecolor='none')
    #   axs[0].add_patch(rect)

    #   axs[0].set_xticks([])
    #   axs[0].set_yticks([])
    #   axs[0].set_title(f'bbox: {y1, x1, y2, x2} area: {(y2 - y1) * (x2 - x1)} rpn score: {rpn_score:.4f}')
    #   axs[0].imshow(raw_image)

    #   # Draw image in a cropped region.
    #   crop = np.copy(raw_image[y1:y2, x1:x2, :])
    #   axs[1].set_xticks([])
    #   axs[1].set_yticks([])
        
    #   axs[1].set_title(f'predicted: {category_names[np.argmax(scores)]}')
    #   axs[1].imshow(crop)

    #   # Draw segmentation inside a cropped region.
    #   axs[2].set_xticks([])
    #   axs[2].set_yticks([])
    #   axs[2].set_title('mask')
    #   axs[2].imshow(crop_w_mask)

    #   # Draw category scores.
    #   fontsize = max(min(fig_size_h / float(len(category_names)) * 45, 20), 8)
    #   for cat_idx in range(len(category_names)):
    #     axs[3].barh(cat_idx, scores[cat_idx], 
    #                 color='orange' if scores[cat_idx] == max(scores) else 'blue')
    #   axs[3].invert_yaxis()
    #   axs[3].set_axisbelow(True)
    #   axs[3].set_xlim(0, 1)
    #   plt.xlabel("confidence score")
    #   axs[3].set_yticks(range(len(category_names)))
    #   axs[3].set_yticklabels(category_names, fontdict={
    #       'fontsize': fontsize})
        
    #   cnt += 1
    #   # fig.tight_layout()


    # print('Detection counts:', cnt)

def detect(image_path, category_name, save_path):
    #Define hyperparameters
    FLAGS = {
        'prompt_engineering': False,
        'this_is': True,
        
        'temperature': 100.0,
        'use_softmax': False,
    }
    FLAGS = EasyDict(FLAGS)


    # Global matplotlib settings
    SMALL_SIZE = 16#10
    MEDIUM_SIZE = 18#12
    BIGGER_SIZE = 20#14

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    # Parameters for drawing figure.
    display_input_size = (10, 10)
    overall_fig_size = (18, 24)

    line_thickness = 2
    fig_size_w = 35
    # fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)
    mask_color =   'red'
    alpha = 0.5
    
    
    # load clip model
    clip.available_models()
    model, preprocess = clip.load("ViT-B/32",download_root='/home/wutong/ComputerVision/vild/clip_model')   # TODO: path to clip model
    
    # # config vild gpu 
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    #     except RuntimeError as e:
    #         print(e)
    # load vild model
    saved_model_dir = '/home/wutong/ComputerVision/vild/image_path_v2' # TODO: path to vild model
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    session = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options))
    _ = tf.saved_model.loader.load(session, ["serve"], saved_model_dir)

    numbered_categories = [{"name": str(idx), "id": idx,} for idx in range(50)]
    numbered_category_indices = {cat["id"]: cat for cat in numbered_categories}
    
    # # set image path
    # image_path = './examples/rubbish_front.png'  #@param {type:"string"}
    # display_image(image_path, size=display_input_size)
    
    ## vild
    # set the category
    # category_name_string = ';'.join(['tomato'])
    # set the params
    max_boxes_to_draw = 25 #@param {type:"integer"}
    nms_threshold = 0.3 #@param {type:"slider", min:0, max:0.9, step:0.05}
    min_rpn_score_thresh = 0.9  #@param {type:"slider", min:0, max:1, step:0.01}
    min_box_area = 100 #@param {type:"slider", min:0, max:10000, step:1.0}


    params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area
    indices, boxes, masks = vild(image_path, category_name, params, session, model, numbered_category_indices, FLAGS, overall_fig_size, save_path)

    return boxes[0]

    