# Graph Report - RTV  (2026-10-18)

## Corpus Check
- 237 files · ~1,478,676 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 12 file(s) not represented in the graph (top: .obj 6, .npz 3, (none) 2)

## Summary
- 2517 nodes · 5584 edges · 156 communities (67 shown, 89 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 326 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d223428a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- embedder.py
- DensePose Data Build
- SMPLModel
- SMPLModel
- DensePoseExtractor
- pix2pixHD Inference Models
- DensePose Trainer & Eval
- lvis.py
- Demo & Garment Removal
- video_keyframe_dataset.py
- Pixel Alignment & Landmarks
- viton_upperbody.py
- typing
- numpy
- torch
- BEV Dataset Generation
- pix2pixHD Base Model
- DensePoseDataRelative
- .__call__
- DensePose Garment Utils
- Garment Dataset & Options
- Docs, Papers & Training Pipeline
- Combined Data Loader
- HRNet Backbone
- Multithread Video IO
- DensePose Vis Extractors
- pix2pixHD Interactive UI
- BilinearInterpolationHelper
- apply_net Visualizers
- DensePose DB Helper
- DensePoseResultsVisualizer
- ASAPNet Model
- apply_net Inference Action
- DensePose Modeling Build
- Human Pose Estimation
- pix2pixHD Networks
- predictor_output_with_coarse_segm_to_mask
- chart_output_to_chart_result.py
- Inference-Based Loader
- .forward
- Deeplab
- coco.py
- Densepose Coco Evaluation
- Mesh
- Random Affine Batch
- densepose/__init__.py
- To Chart Result
- samplers/__init__.py
- os
- converters/__init__.py
- Densepose Base
- Pix2PixHD_RGBA
- Networks
- Apply Net
- Evaluator
- Cycle Shape2Shape
- SegmentationLoss
- MatrixVisualizer
- Base 2
- Utransformer Model
- flat_renderer.py
- Tensor Storage
- Pose Filter
- Convlstm
- Chart With Confidences
- Video Loader
- extract_data_for_mask_loss_from_matches
- HTML
- ._lookup_converter
- Evaluator 2
- MyBEV
- decorate_predictor_output_class_with_confidences
- DensePoseCheckpointer
- Trainer
- Upperbody Garment
- Densepose Coco Evaluation 2
- DensePoseList
- DensePoseResultExtractor
- ImageResizeTransform
- Densepose Cse Base
- predictors/chart_confidence.py
- Cse Confidence
- data/build.py
- Jin 00 White Bg
- normalize_embeddings
- f
- squared_euclidean_distance_matrix
- Image List Dataset
- V1Convx
- save_obj
- Error Measure
- Tensor Storage 3
- Weak Perspective
- Densepose Coco Evaluation 5
- torch
- Pix2PixHD_RNN_RGBA
- Screenshot
- Networks 2
- Pose Filter 2
- Trans Smooth
- Densepose Coco Evaluation 6
- create_embedder
- Large Video Loader
- Han
- DensePoseChartPredictorOutput
- Networks 3
- Image2Video
- ._extract_iuv
- .reset_parameters
- Claude
- Lazygit
- Install
- Upperbody Training
- .__init__
- advect_img.py
- _MeshCatalog
- SMPL_Renderer

## God Nodes (most connected - your core abstractions)
1. `BaseModel` - 35 edges
2. `Pix2PixHDModel` - 32 edges
3. `DensePoseExtractor` - 31 edges
4. `DensePoseCocoEval` - 30 edges
5. `ImagePool` - 29 edges
6. `SMPL_Regressor` - 27 edges
7. `SMPLModel` - 25 edges
8. `DensePoseDataRelative` - 24 edges
9. `create_model()` - 24 edges
10. `ASAPNetModel` - 22 edges

## Surprising Connections (you probably didn't know these)
- `Asymmetric Pose Guidance (30 paired poses, one arm raised vs both)` --conceptually_related_to--> `Step 1: Record 4K video of person wearing garment`  [INFERRED]
  assets/pose_guidance/asymmetric.pdf → training_instructions.md
- `AmputatedSMPL` --uses--> `FlatRenderer`  [INFERRED]
  SMPL/amputated_smpl.py → OffscreenRenderer/flat_renderer.py
- `FullBodyFrameProcessor` --uses--> `DensePoseExtractor`  [INFERRED]
  VITON/viton_fullbody.py → model/DensePose/densepose_extractor.py
- `FullBodySeqFrameProcessor` --uses--> `DensePoseExtractor`  [INFERRED]
  VITON/viton_fullbody_seq.py → model/DensePose/densepose_extractor.py
- `VitonThread` --uses--> `FrameProcessor`  [INFERRED]
  rtl_demo.py → VITON/viton_upperbody.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **DensePose R-CNN FPN configuration stack** — model_densepose_configs_densepose_rcnn_r_50_fpn_s1x_config, model_densepose_configs_base_densepose_rcnn_fpn_config, model_densepose_configs_base_densepose_rcnn_fpn_generalized_rcnn, model_densepose_configs_base_densepose_rcnn_fpn_densepose_roi_heads [EXTRACTED 1.00]
- **Per-garment checkpoint creation pipeline (record -> segment -> dataset -> train -> infer)** — training_instructions_video_recording, training_instructions_garment_segmentation, training_instructions_upperbody_dataset_generation, training_instructions_upperbody_training, training_instructions_upperbody_inference [EXTRACTED 1.00]
- **Garment data preparation: SAM2 garment masking, SMPL UV atlas, texture pattern board** — assets_demo_screenshot_sam2_mask_tool_ui, assets_smpl_amputated_smpl_m_zero_uv_smpl_torso_uv_map, assets_color_pattern_board_300x300_texture_correspondence_pattern [INFERRED 0.65]
- **Try-on garment input set: both series share the white-bg headless upper-body cutout format** — assets_garment_images_jin_00_white_bg_jin_garment_series, assets_garment_images_lab_00_white_bg_lab_garment_series, assets_garment_images_jin_00_white_bg_white_background_garment_input, assets_garment_images_jin_00_white_bg_upper_body_garment [INFERRED 0.85]
- **Pose-guided low-barrier dataset capture** — assets_pose_guidance_symmetric_pose_guide, assets_pose_guidance_asymmetric_pose_guide, training_instructions_video_recording, readme_paper_low_barrier_dataset_collection [INFERRED 0.85]
- **Real-time try-on display with garment selection including 'none' option** — assets_output_mirror_display_demo, assets_output_garment_selection_sidebar, assets_none_no_garment_icon [INFERRED 0.85]

## Communities (156 total, 89 thin omitted)

### Community 1 - "DensePose Data Build"
Cohesion: 0.09
Nodes (22): _add_category_info_to_bootstrapping_metadata(), _add_category_maps_to_metadata(), _add_category_whitelists_to_metadata(), build_bootstrap_dataset(), build_detection_test_loader(), build_detection_train_loader(), _get_test_keep_instance_predicate(), _get_train_keep_instance_predicate() (+14 more)

### Community 4 - "DensePoseExtractor"
Cohesion: 0.07
Nodes (14): gen_dataset(), make_video_loader(), process_video(), VitonThread, FlatRenderer, FullBodySMPL, SMPL_Regressor, UpperBodySMPL (+6 more)

### Community 5 - "pix2pixHD Inference Models"
Cohesion: 0.10
Nodes (12): create_model(), InferenceModel_Align, Pix2PixHDModel_Align, InferenceModel, Pix2PixHD_DUAL_RGBA, InferenceModel_Inpaint, Pix2PixHDModel_Inpaint, InferenceModel_Mask (+4 more)

### Community 7 - "lvis.py"
Cohesion: 0.17
Nodes (12): _add_categories_metadata(), _combine_images_with_annotations(), _load_lvis_annotations(), load_lvis_json(), _maybe_add_bbox(), _maybe_add_densepose(), _maybe_add_keypoints(), _maybe_add_segm() (+4 more)

### Community 8 - "Demo & Garment Removal"
Cohesion: 0.05
Nodes (12): _blur(), _dilate(), _lut(), WornGarmentRemover, MultiHeadCrossAttention, MultiHeadSelfAttention, DualEncoderGenerator, ResnetBlock (+4 more)

### Community 9 - "video_keyframe_dataset.py"
Cohesion: 0.06
Nodes (12): build_frame_selector(), build_transform(), build_video_list_dataset(), FirstKFramesSelector, FrameSelectionStrategy, LastKFramesSelector, RandomKFramesSelector, list_keyframes() (+4 more)

### Community 10 - "Pixel Alignment & Landmarks"
Cohesion: 0.08
Nodes (18): get_max_preds(), GetLandmarkCoord(), ImageCaption(), LandmarkAnnotation(), AABB, compute_iou(), get_threshold_mask(), joint_align() (+10 more)

### Community 11 - "viton_upperbody.py"
Cohesion: 0.10
Nodes (24): erode(), naive_overlay(), naive_overlay_alpha(), TestOptions, blur_image(), blur_image_by_mask(), poisson_blend(), poisson_blend_np() (+16 more)

### Community 12 - "typing"
Cohesion: 0.09
Nodes (7): EmbeddingLoss, CseAnnotationsAccumulator, PackedCseAnnotations, SoftEmbeddingLoss, AnnotationsAccumulator, extract_packed_annotations_from_matches(), resample_data()

### Community 13 - "numpy"
Cohesion: 0.10
Nodes (17): dilate(), get_smooth_2d(), make_kernel(), multiple_random_dilation(), random_dilate(), get_laplacian_matrix(), ImageCaption(), ImageCaptionBottom() (+9 more)

### Community 14 - "torch"
Cohesion: 0.36
Nodes (8): hsl2rgb_torch(), hsv2rgb_torch(), image_quantization(), img2tensor(), measurement_quantization(), rgb2hsl_torch(), rgb2hsv_torch(), tensor2img()

### Community 15 - "BEV Dataset Generation"
Cohesion: 0.20
Nodes (3): projection2screencoord(), MyROMP, load_obj()

### Community 20 - "DensePose Garment Utils"
Cohesion: 0.15
Nodes (19): crop(), main(), main(), crop(), IUV2HandHeadMask(), IUV2Img(), IUV2SDP(), IUV2SSDP() (+11 more)

### Community 21 - "Garment Dataset & Options"
Cohesion: 0.11
Nodes (9): UpperBodyGarment, BaseOptions, TrainOptions, lcm(), main(), mkdir(), mkdirs(), save_image() (+1 more)

### Community 22 - "Docs, Papers & Training Pipeline"
Cohesion: 0.10
Nodes (26): Asymmetric Pose Guidance (30 paired poses, one arm raised vs both), Symmetric Pose Guidance (16 predefined poses), Base-DensePose-RCNN-FPN config, DensePose COCO 2014 datasets, DensePoseROIHeads / DensePoseV1ConvXHead, GeneralizedRCNN with ResNet-FPN backbone, densepose_rcnn_R_50_FPN_s1x config (ResNet-50, 130k iters), detectron2 (+18 more)

### Community 23 - "Combined Data Loader"
Cohesion: 0.13
Nodes (8): _BootstrapDatasetFactoryCatalog, register_dataset(), DatasetType, MeshInfo, register_mesh(), register_meshes(), is_relative_local_path(), maybe_prepend_base_path()

### Community 24 - "HRNet Backbone"
Cohesion: 0.06
Nodes (8): build_hrfpn_backbone(), HRFPN, BasicBlock, Bottleneck, build_pose_hrnet_backbone(), conv3x3(), HighResolutionModule, PoseHigherResolutionNet

### Community 25 - "Multithread Video IO"
Cohesion: 0.12
Nodes (6): BaseOptions, process_video(), MultithreadVideoLoader, MultithreadVideoWriter, compute_confidence(), improve_masks_with_optical_flow()

### Community 26 - "DensePose Vis Extractors"
Cohesion: 0.11
Nodes (8): BoundingBoxExtractor, CompoundExtractor, create_extractor(), extract_boxes_xywh_from_instances(), extract_scores_from_instances(), NmsFilteredExtractor, ScoredBoundingBoxExtractor, ScoreThresholdedExtractor

### Community 27 - "pix2pixHD Interactive UI"
Cohesion: 0.14
Nodes (6): UIModel, Colorize, labelcolormap(), tensor2im(), tensor2label(), uint82bin()

### Community 28 - "BilinearInterpolationHelper"
Cohesion: 0.09
Nodes (6): DensePoseChartLoss, DensePoseChartWithConfidenceLoss, BilinearInterpolationHelper, ChartBasedAnnotationsAccumulator, PackedChartBasedAnnotations, sample_random_indices()

### Community 29 - "apply_net Visualizers"
Cohesion: 0.13
Nodes (6): create_argument_parser(), main(), get_palette(), verbosity_to_level(), dump_esample.sh script, example.sh script

### Community 30 - "DensePose DB Helper"
Cohesion: 0.12
Nodes (5): AllEntrySelector, EntrySelector, FieldEntrySelector, _FieldEntryRangePredicate, _FieldEntryValuePredicate

### Community 31 - "DensePoseResultsVisualizer"
Cohesion: 0.08
Nodes (11): DensePoseMaskedColormapResultsVisualizer, DensePoseResultsCustomContourVisualizer, DensePoseResultsFineSegmentationVisualizer, DensePoseResultsMplContourVisualizer, DensePoseResultsUVisualizer, DensePoseResultsVisualizer, DensePoseResultsVVisualizer, _extract_i_from_iuvarr() (+3 more)

### Community 32 - "ASAPNet Model"
Cohesion: 0.12
Nodes (5): ASAPNetModel, loss_filter(), InferenceModel, ASAPNet_RGBA, InferenceModel

### Community 33 - "apply_net Inference Action"
Cohesion: 0.16
Nodes (3): InferenceAction, register_action(), ShowAction

### Community 34 - "DensePose Modeling Build"
Cohesion: 0.07
Nodes (12): build_densepose_data_filter(), build_densepose_embedder(), build_densepose_head(), build_densepose_losses(), build_densepose_predictor(), DensePoseConfidenceModelConfig, DensePoseSegmConfidenceConfig, DensePoseUVConfidenceConfig (+4 more)

### Community 35 - "Human Pose Estimation"
Cohesion: 0.09
Nodes (8): HeatmapGenerator, parse_args(), HumanJointHeatmap, HumanPose, parse_args(), GetJointCoord(), SkeletonAnnotation(), SkeletonTensor()

### Community 36 - "pix2pixHD Networks"
Cohesion: 0.22
Nodes (7): define_D(), define_G(), Encoder, GANLoss, get_norm_layer(), VGGLoss, weights_init()

### Community 37 - "predictor_output_with_coarse_segm_to_mask"
Cohesion: 0.18
Nodes (5): predictor_output_with_coarse_segm_to_mask(), predictor_output_with_fine_and_coarse_segm_to_mask(), resample_coarse_segm_tensor_to_bbox(), resample_fine_and_coarse_segm_tensors_to_bbox(), resample_fine_and_coarse_segm_to_bbox()

### Community 38 - "chart_output_to_chart_result.py"
Cohesion: 0.16
Nodes (7): make_int_box(), densepose_chart_predictor_output_to_result(), densepose_chart_predictor_output_to_result_with_confidences(), resample_confidences_to_bbox(), resample_uv_tensors_to_bbox(), resample_uv_to_bbox(), DensePoseChartResultWithConfidences

### Community 39 - "Inference-Based Loader"
Cohesion: 0.12
Nodes (6): build_data_filter(), build_inference_based_loader(), build_inference_based_loaders(), _grouper(), InferenceBasedLoader, ScoreBasedFilter

### Community 40 - ".forward"
Cohesion: 0.12
Nodes (6): DensePoseCseLoss, _create_pixel_dist_matrix(), PixToShapeCycleLoss, _sample_fg_pixels_multinomial(), _sample_fg_pixels_randperm(), MaskOrSegmentationLoss

### Community 41 - "Deeplab"
Cohesion: 0.11
Nodes (6): ASPP, ASPPConv, ASPPPooling, DensePoseDeepLabHead, NONLocalBlock2D, _NonLocalBlockND

### Community 42 - "coco.py"
Cohesion: 0.17
Nodes (15): _add_categories_metadata(), CocoDatasetInfo, _combine_images_with_annotations(), create_video_frame_mapping(), get_metadata(), _load_coco_annotations(), load_coco_json(), _maybe_add_bbox() (+7 more)

### Community 44 - "Mesh"
Cohesion: 0.21
Nodes (4): load_mesh_auxiliary_data(), load_mesh_data(), _maybe_copy_to_device(), Mesh

### Community 45 - "Random Affine Batch"
Cohesion: 0.17
Nodes (5): compute_inv_trans(), compute_trans(), RandomAffineBatch, RandomAffineBatchNumpy, RandomAffineMatrix

### Community 46 - "densepose/__init__.py"
Cohesion: 0.05
Nodes (16): add_bootstrap_config(), add_dataset_category_config(), add_densepose_config(), add_densepose_head_config(), add_densepose_head_cse_config(), add_evaluation_config(), add_hrnet_config(), get_bootstrap_dataset_config() (+8 more)

### Community 48 - "samplers/__init__.py"
Cohesion: 0.08
Nodes (8): build_data_sampler(), DensePoseConfidenceBasedSampler, DensePoseCSEConfidenceBasedSampler, DensePoseCSEUniformSampler, DensePoseUniformSampler, MaskFromDensePoseSampler, PredictionToGroundTruthSampler, _Sampler

### Community 49 - "os"
Cohesion: 0.09
Nodes (10): BaseOptions, CameraApp, get_camera_name_linux(), list_available_cameras(), main(), check_type(), get_file_path_list(), download_gdrive_file() (+2 more)

### Community 50 - "converters/__init__.py"
Cohesion: 0.26
Nodes (5): BaseConverter, HFlipConverter, ToChartResultConverter, ToChartResultConverterWithConfidences, ToMaskConverter

### Community 52 - "Pix2PixHD_RGBA"
Cohesion: 0.17
Nodes (3): loss_filter(), InferenceModel, Pix2PixHD_RGBA

### Community 53 - "Networks"
Cohesion: 0.14
Nodes (4): GlobalGenerator, GlobalGenerator_Mask, LocalEnhancer, ResnetBlock

### Community 55 - "Evaluator"
Cohesion: 0.17
Nodes (3): maybe_filter_and_map_categories_cocoapi(), DensePoseCOCOEvaluator, MeshAlignmentEvaluator

### Community 58 - "MatrixVisualizer"
Cohesion: 0.06
Nodes (16): MatrixVisualizer, PointsVisualizer, _densepose_data_i_for_cmap(), _densepose_data_u_for_cmap(), _densepose_data_v_for_cmap(), DensePoseDataCoarseSegmentationVisualizer, DensePoseDataPointsIVisualizer, DensePoseDataPointsUVisualizer (+8 more)

### Community 59 - "Base 2"
Cohesion: 0.16
Nodes (5): CompoundVisualizer, RectangleVisualizer, TextVisualizer, BoundingBoxVisualizer, ScoredBoundingBoxVisualizer

### Community 61 - "flat_renderer.py"
Cohesion: 0.08
Nodes (12): BaseRenderer, _create_glfw_context(), create_initialized_headless_egl_display(), create_opengl_context(), destroy_opengl_context(), init_frame_buffer(), generate_vao(), read_texture() (+4 more)

### Community 63 - "Pose Filter"
Cohesion: 0.18
Nodes (3): Average, PoseFilter, slerp()

### Community 64 - "Convlstm"
Cohesion: 0.20
Nodes (3): ConvLSTM, ConvLSTMCell, RNNGenerator

### Community 67 - "extract_data_for_mask_loss_from_matches"
Cohesion: 0.22
Nodes (3): DataForMaskLoss, extract_data_for_mask_loss_from_matches(), MaskLoss

### Community 70 - "Evaluator 2"
Cohesion: 0.14
Nodes (7): _maybe_add_iscrowd_annotations(), densepose_chart_predictions_to_storage_dict(), densepose_cse_predictions_to_dict(), _derive_results_from_coco_eval(), _evaluate_predictions_on_coco(), _get_densepose_metrics(), prediction_to_dict()

### Community 74 - "decorate_predictor_output_class_with_confidences"
Cohesion: 0.18
Nodes (4): decorate_predictor_output_class_with_confidences(), PredictorOutput_getitem(), PredictorOutput_to(), slice_if_not_none()

### Community 76 - "Trainer"
Cohesion: 0.19
Nodes (4): get_contiguous_id_to_category_id_map(), maybe_filter_categories_cocoapi(), Trainer, Detectron2COCOEvaluatorAdapter

### Community 80 - "DensePoseResultExtractor"
Cohesion: 0.20
Nodes (4): DumpAction, DensePoseExtractor, DensePoseOutputsExtractor, DensePoseResultExtractor

### Community 83 - "predictors/chart_confidence.py"
Cohesion: 0.07
Nodes (7): DensePoseChartConfidencePredictorMixin, DensePoseChartPredictor, DensePoseChartWithConfidencePredictor, DensePoseEmbeddingConfidencePredictorMixin, DensePoseEmbeddingPredictor, DensePoseEmbeddingWithConfidencePredictor, initialize_module_params()

### Community 84 - "Cse Confidence"
Cohesion: 0.18
Nodes (4): decorate_cse_predictor_output_class_with_confidences(), PredictorOutput_getitem(), PredictorOutput_to(), slice_if_not_none()

### Community 85 - "data/build.py"
Cohesion: 0.11
Nodes (11): _add_category_id_to_contiguous_id_maps_to_metadata(), build_combined_loader(), combine_detection_dataset_dicts(), _compute_num_images_per_worker(), _DatasetCategory, _map_category_id_to_contiguous_id(), _maybe_filter_and_map_categories(), _merge_categories() (+3 more)

### Community 86 - "Jin 00 White Bg"
Cohesion: 0.35
Nodes (11): Jin Garment Series (jin_00..jin_26, 27 images), Upper-Body Garment Category, White-Background Garment Input Convention (headless front-view cutout, 1024x1024), jin_07 Light-Blue Pussy-Bow Blouse, jin_13 White Polka-Dot Short-Sleeve Shirt, Knit Sweater, Collared Button-Up Shirt / Blouse, lab_00 White Graphic Tee ('Full-time' print) (+3 more)

### Community 98 - "Error Measure"
Cohesion: 0.23
Nodes (3): get_texture_atlases(), ErrorMeasure, json_wrtiter()

### Community 102 - "Tensor Storage 3"
Cohesion: 0.14
Nodes (13): build_densepose_evaluator_storage(), _calculate_record_field_size_b(), _calculate_record_field_sizes_b(), _calculate_record_size_b(), _file_storage_gather(), MultiProcessFileTensorStorage, MultiProcessRamTensorStorage, _ram_storage_gather() (+5 more)

### Community 106 - "Densepose Coco Evaluation 5"
Cohesion: 0.39
Nodes (5): _summarize(), _summarizeDets(), _summarizeKps(), _summarizeUvs(), _summarizeUvsOld()

### Community 107 - "torch"
Cohesion: 0.07
Nodes (13): compress_quantized_densepose_chart_result(), decompress_compressed_densepose_chart_result(), DensePoseChartResult, DensePoseChartResultCompressed, DensePoseChartResultQuantized, quantize_densepose_chart_result(), DensePoseEmbeddingPredictorOutput, create_mesh() (+5 more)

### Community 109 - "Pix2PixHD_RNN_RGBA"
Cohesion: 0.15
Nodes (3): InferenceModel, Pix2PixHD_RNN_RGBA, OpticalFlow

### Community 110 - "Screenshot"
Cohesion: 0.29
Nodes (7): board_300x300.png: Random Color Noise Pattern Board, Random Color Texture Pattern for Dense Correspondence / UV Calibration, Garment Mask Buffers (4 buffers, Save/Clear, history, prompt store, track), SAM 2.1 Hiera Large checkpoint (cuda/bfloat16), Screenshot.png: SAM2 Interactive Garment Masking/Tracking UI, SMPL Torso + Upper-arm UV Atlas (front, back, two sleeves), amputated_smpl_m_zero_uv.png: Amputated SMPL Upper-body UV Layout

### Community 124 - "Han"
Cohesion: 0.40
Nodes (5): Full-body Garment Replacement (traditional Hanfu over casual outfit), Han.gif: Hanfu Virtual Try-on Before/After Demo, none.png: 'None' / No-garment Prohibition Icon, Garment Selection Sidebar (icons incl. none option), output.gif: Real-time Virtual Try-on Mirror Display Demo

### Community 132 - "DensePoseChartPredictorOutput"
Cohesion: 0.11
Nodes (5): densepose_chart_predictor_output_hflip(), _flip_iuv_semantics_tensor(), _flip_segm_semantics_tensor(), DensePoseChartPredictorOutput, DensePoseTransformData

### Community 169 - "advect_img.py"
Cohesion: 0.53
Nodes (4): advect_numpy_image(), advect_numpy_image2(), advect_tensor_image(), mask_smooth()

## Ambiguous Edges - Review These
- `Garment Mask Buffers (4 buffers, Save/Clear, history, prompt store, track)` → `SMPL Torso + Upper-arm UV Atlas (front, back, two sleeves)`  [AMBIGUOUS]
  assets/demo/Screenshot.png · relation: conceptually_related_to

## Knowledge Gaps
- **17 isolated node(s):** `Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting Garments (Wu, Shen, Igarashi, CGF 2025, arXiv 2506.12348)`, `SAM 2.1 Hiera Large checkpoint (cuda/bfloat16)`, `board_300x300.png: Random Color Noise Pattern Board`, `amputated_smpl_m_zero_uv.png: Amputated SMPL Upper-body UV Layout`, `Full-body Garment Replacement (traditional Hanfu over casual outfit)` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 843 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **89 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DensePoseChartLoss` connect `BilinearInterpolationHelper` to `.forward`, `DensePose Chart Loss`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **What connects `Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting Garments (Wu, Shen, Igarashi, CGF 2025, arXiv 2506.12348)`, `SAM 2.1 Hiera Large checkpoint (cuda/bfloat16)`, `board_300x300.png: Random Color Noise Pattern Board` to the rest of the system?**
  _17 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `DensePose Data Build` be split into smaller, more focused modules?**
  _Cohesion score 0.08677098150782361 - nodes in this community are weakly interconnected._
- **Why does `initialize_module_params()` connect `predictors/chart_confidence.py` to `DensePose Modeling Build`, `V1Convx`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Should `SMPLModel` be split into smaller, more focused modules?**
  _Cohesion score 0.14666666666666667 - nodes in this community are weakly interconnected._
- **Should `DensePoseExtractor` be split into smaller, more focused modules?**
  _Cohesion score 0.06594071385359952 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Garment Mask Buffers (4 buffers, Save/Clear, history, prompt store, track)` and `SMPL Torso + Upper-arm UV Atlas (front, back, two sleeves)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._