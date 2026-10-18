# Graph Report - RTV  (2026-10-18)

## Corpus Check
- 243 files · ~1,480,881 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 12 file(s) not represented in the graph (top: .obj 6, .npz 3, (none) 2)

## Summary
- 2559 nodes · 5681 edges · 154 communities (76 shown, 78 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 339 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d6a58934`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- embedder.py
- DensePose Data Build
- SMPLModel
- process_video
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
- DensePoseChartLoss
- .__call__
- DensePose Garment Utils
- Garment Dataset & Options
- Docs, Papers & Training Pipeline
- Combined Data Loader
- HRNet Backbone
- upperbody_dataset_generation.py
- DensePose Vis Extractors
- pix2pixHD Interactive UI
- ChartBasedAnnotationsAccumulator
- apply_net Visualizers
- DensePose DB Helper
- DensePoseResultsVisualizer
- ASAPNet Model
- apply_net Inference Action
- DensePose Modeling Build
- Human Pose Estimation
- pix2pixHD Networks
- resample_coarse_segm_tensor_to_bbox
- chart_output_to_chart_result.py
- Inference-Based Loader
- torch_nn
- Deeplab
- coco.py
- DensePoseCocoEval
- Mesh
- Random Affine Batch
- test_time_augmentation.py
- To Chart Result
- build_data_sampler
- viton_upperbody.py
- converters/__init__.py
- Densepose Base
- detectron2_config
- Networks
- Apply Net
- Evaluator
- Cycle Shape2Shape
- SegmentationLoss
- densepose_data_points.py
- Base 2
- Utransformer Model
- flat_renderer.py
- tensor_storage.py
- frame_selector.py
- Convlstm
- densepose_outputs_vertex.py
- Video Loader
- losses/cse.py
- HTML
- ._lookup_converter
- Evaluator 2
- upperbody_inference.py
- BilinearInterpolationHelper
- densepose/__init__.py
- DensePoseChartConfidencePredictorMixin
- DensePoseCheckpointer
- Trainer
- Upperbody Garment
- MatrixVisualizer
- DensePoseList
- DensePoseResultExtractor
- DensePoseROIHeads
- Densepose Cse Base
- predictors/__init__.py
- Cse Confidence
- combine_detection_dataset_dicts
- Jin 00 White Bg
- densepose_outputs_iuv.py
- Detectron2COCOEvaluatorAdapter
- DensePoseChartPredictor
- UpperBodyGarment
- squared_euclidean_distance_matrix
- Image List Dataset
- V1Convx
- DensePoseEmbeddingPredictor
- object
- Error Measure
- DensePoseEmbeddingPredictorOutput
- samplers/__init__.py
- CombinedDataLoader
- Tensor Storage 3
- Weak Perspective
- DensePoseResultsVisualizerWithTexture
- detectron2_data
- structures/__init__.py
- .register
- Pix2PixHD_RNN_RGBA
- Screenshot
- DensePoseConfidenceBasedSampler
- Densepose Cse Confidence Based
- Networks 2
- MaskFromDensePoseSampler
- decorate_predictor_output_class_with_confidences
- Trans Smooth
- Encoder
- Large Video Loader
- Han
- DensePoseTransformData
- Networks 3
- Image2Video
- .reset_parameters
- Claude
- Lazygit
- Install
- Upperbody Training
- .__init__
- _MeshCatalog

## God Nodes (most connected - your core abstractions)
1. `BaseModel` - 35 edges
2. `Pix2PixHDModel` - 32 edges
3. `DensePoseExtractor` - 31 edges
4. `DensePoseCocoEval` - 30 edges
5. `ImagePool` - 29 edges
6. `SMPL_Regressor` - 27 edges
7. `SMPLModel` - 25 edges
8. `DensePoseDataRelative` - 24 edges
9. `create_model()` - 23 edges
10. `FrameProcessor` - 22 edges

## Surprising Connections (you probably didn't know these)
- `Asymmetric Pose Guidance (30 paired poses, one arm raised vs both)` --conceptually_related_to--> `Step 1: Record 4K video of person wearing garment`  [INFERRED]
  assets/pose_guidance/asymmetric.pdf → training_instructions.md
- `AmputatedSMPL` --uses--> `FlatRenderer`  [INFERRED]
  SMPL/amputated_smpl.py → OffscreenRenderer/flat_renderer.py
- `UpperBodySMPL` --uses--> `FlatRenderer`  [INFERRED]
  SMPL/upperbody_smpl/UpperBody.py → OffscreenRenderer/flat_renderer.py
- `FrameProcessor` --uses--> `SMPL_Regressor`  [INFERRED]
  VITON/viton_upperbody.py → SMPL/smpl_regressor.py
- `FrameProcessor` --uses--> `UpperBodySMPL`  [INFERRED]
  VITON/viton_upperbody.py → SMPL/upperbody_smpl/UpperBody.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **DensePose R-CNN FPN configuration stack** — model_densepose_configs_densepose_rcnn_r_50_fpn_s1x_config, model_densepose_configs_base_densepose_rcnn_fpn_config, model_densepose_configs_base_densepose_rcnn_fpn_generalized_rcnn, model_densepose_configs_base_densepose_rcnn_fpn_densepose_roi_heads [EXTRACTED 1.00]
- **Per-garment checkpoint creation pipeline (record -> segment -> dataset -> train -> infer)** — training_instructions_video_recording, training_instructions_garment_segmentation, training_instructions_upperbody_dataset_generation, training_instructions_upperbody_training, training_instructions_upperbody_inference [EXTRACTED 1.00]
- **Garment data preparation: SAM2 garment masking, SMPL UV atlas, texture pattern board** — assets_demo_screenshot_sam2_mask_tool_ui, assets_smpl_amputated_smpl_m_zero_uv_smpl_torso_uv_map, assets_color_pattern_board_300x300_texture_correspondence_pattern [INFERRED 0.65]
- **Try-on garment input set: both series share the white-bg headless upper-body cutout format** — assets_garment_images_jin_00_white_bg_jin_garment_series, assets_garment_images_lab_00_white_bg_lab_garment_series, assets_garment_images_jin_00_white_bg_white_background_garment_input, assets_garment_images_jin_00_white_bg_upper_body_garment [INFERRED 0.85]
- **Pose-guided low-barrier dataset capture** — assets_pose_guidance_symmetric_pose_guide, assets_pose_guidance_asymmetric_pose_guide, training_instructions_video_recording, readme_paper_low_barrier_dataset_collection [INFERRED 0.85]
- **Real-time try-on display with garment selection including 'none' option** — assets_output_mirror_display_demo, assets_output_garment_selection_sidebar, assets_none_no_garment_icon [INFERRED 0.85]

## Communities (154 total, 78 thin omitted)

### Community 0 - "embedder.py"
Cohesion: 0.15
Nodes (3): create_embedder(), Embedder, EmbedderType

### Community 1 - "DensePose Data Build"
Cohesion: 0.08
Nodes (24): _add_category_info_to_bootstrapping_metadata(), _add_category_maps_to_metadata(), _add_category_whitelists_to_metadata(), build_combined_loader(), build_detection_test_loader(), build_detection_train_loader(), build_transform(), _compute_num_images_per_worker() (+16 more)

### Community 2 - "SMPLModel"
Cohesion: 0.05
Nodes (6): f(), SMPLModel, mannequin2SMPL(), SMPLModel, SMPL_Renderer, save_obj()

### Community 3 - "process_video"
Cohesion: 0.06
Nodes (7): process_video(), CameraApp, VitonThread, FrameTiming, LatencyRecorder, _NullTiming, MultithreadVideoWriter

### Community 4 - "DensePoseExtractor"
Cohesion: 0.08
Nodes (9): CameraApp, VitonThread, FlatRenderer, FullBodySMPL, SMPL_Regressor, im2tensor(), FullBodyFrameProcessor, make_pix2pix_model() (+1 more)

### Community 5 - "pix2pixHD Inference Models"
Cohesion: 0.07
Nodes (15): create_model(), InferenceModel_Align, Pix2PixHDModel_Align, InferenceModel, Pix2PixHD_DUAL_RGBA, InferenceModel_Inpaint, Pix2PixHDModel_Inpaint, InferenceModel_Mask (+7 more)

### Community 7 - "lvis.py"
Cohesion: 0.15
Nodes (13): build_augmentation(), _add_categories_metadata(), _combine_images_with_annotations(), _load_lvis_annotations(), load_lvis_json(), _maybe_add_bbox(), _maybe_add_densepose(), _maybe_add_keypoints() (+5 more)

### Community 8 - "Demo & Garment Removal"
Cohesion: 0.12
Nodes (8): _blur(), _dilate(), _lut(), WornGarmentRemover, MultiHeadCrossAttention, MultiHeadSelfAttention, DualEncoderGenerator, ResnetBlock

### Community 9 - "video_keyframe_dataset.py"
Cohesion: 0.11
Nodes (5): list_keyframes(), read_keyframe_helper_data(), read_keyframes(), video_list_from_file(), VideoKeyframeDataset

### Community 10 - "Pixel Alignment & Landmarks"
Cohesion: 0.06
Nodes (20): get_max_preds(), GetLandmarkCoord(), ImageCaption(), LandmarkAnnotation(), AABB, compute_iou(), get_threshold_mask(), JointAlignment (+12 more)

### Community 11 - "viton_upperbody.py"
Cohesion: 0.08
Nodes (26): erode(), naive_overlay(), naive_overlay_alpha(), get_camera_name_linux(), list_available_cameras(), main(), IUV2HandHeadMask(), blur_image() (+18 more)

### Community 12 - "typing"
Cohesion: 0.09
Nodes (3): AnnotationsAccumulator, resample_data(), sample_random_indices()

### Community 13 - "numpy"
Cohesion: 0.10
Nodes (24): advect_numpy_image(), advect_numpy_image2(), advect_tensor_image(), mask_smooth(), dilate(), get_smooth_2d(), make_kernel(), multiple_random_dilation() (+16 more)

### Community 14 - "torch"
Cohesion: 0.28
Nodes (9): diff_tex_sampling(), hsl2rgb_torch(), hsv2rgb_torch(), image_quantization(), img2tensor(), measurement_quantization(), rgb2hsl_torch(), rgb2hsv_torch() (+1 more)

### Community 15 - "BEV Dataset Generation"
Cohesion: 0.11
Nodes (4): MyBEV, projection2screencoord(), MyROMP, load_obj()

### Community 18 - "DensePoseChartLoss"
Cohesion: 0.08
Nodes (4): DensePoseChartLoss, DensePoseChartWithConfidenceLoss, IIDIsotropicGaussianUVLoss, IndepAnisotropicGaussianUVLoss

### Community 20 - "DensePose Garment Utils"
Cohesion: 0.14
Nodes (18): crop(), main(), main(), crop(), IUV2Img(), IUV2SDP(), IUV2SSDP(), IUV2SSDP_deprecated() (+10 more)

### Community 21 - "Garment Dataset & Options"
Cohesion: 0.10
Nodes (11): write_meta(), BaseOptions, TrainOptions, lcm(), main(), download_gdrive_file(), TemporalVarianceHeatmap, mkdir() (+3 more)

### Community 22 - "Docs, Papers & Training Pipeline"
Cohesion: 0.10
Nodes (26): Asymmetric Pose Guidance (30 paired poses, one arm raised vs both), Symmetric Pose Guidance (16 predefined poses), Base-DensePose-RCNN-FPN config, DensePose COCO 2014 datasets, DensePoseROIHeads / DensePoseV1ConvXHead, GeneralizedRCNN with ResNet-FPN backbone, densepose_rcnn_R_50_FPN_s1x config (ResNet-50, 130k iters), detectron2 (+18 more)

### Community 23 - "Combined Data Loader"
Cohesion: 0.20
Nodes (7): register_dataset(), DatasetType, MeshInfo, register_mesh(), register_meshes(), is_relative_local_path(), maybe_prepend_base_path()

### Community 24 - "HRNet Backbone"
Cohesion: 0.06
Nodes (8): build_hrfpn_backbone(), HRFPN, BasicBlock, Bottleneck, build_pose_hrnet_backbone(), conv3x3(), HighResolutionModule, PoseHigherResolutionNet

### Community 25 - "upperbody_dataset_generation.py"
Cohesion: 0.11
Nodes (10): BaseOptions, gen_dataset(), make_video_loader(), process_video(), UpperBodySMPL, check_type(), get_file_path_list(), MultithreadVideoLoader (+2 more)

### Community 26 - "DensePose Vis Extractors"
Cohesion: 0.20
Nodes (4): CompoundExtractor, create_extractor(), extract_boxes_xywh_from_instances(), extract_scores_from_instances()

### Community 27 - "pix2pixHD Interactive UI"
Cohesion: 0.14
Nodes (6): UIModel, Colorize, labelcolormap(), tensor2im(), tensor2label(), uint82bin()

### Community 28 - "ChartBasedAnnotationsAccumulator"
Cohesion: 0.12
Nodes (3): ChartBasedAnnotationsAccumulator, extract_packed_annotations_from_matches(), PackedChartBasedAnnotations

### Community 29 - "apply_net Visualizers"
Cohesion: 0.14
Nodes (5): create_argument_parser(), main(), verbosity_to_level(), dump_esample.sh script, example.sh script

### Community 30 - "DensePose DB Helper"
Cohesion: 0.12
Nodes (5): AllEntrySelector, EntrySelector, FieldEntrySelector, _FieldEntryRangePredicate, _FieldEntryValuePredicate

### Community 31 - "DensePoseResultsVisualizer"
Cohesion: 0.10
Nodes (10): DensePoseMaskedColormapResultsVisualizer, DensePoseResultsCustomContourVisualizer, DensePoseResultsFineSegmentationVisualizer, DensePoseResultsMplContourVisualizer, DensePoseResultsUVisualizer, DensePoseResultsVisualizer, DensePoseResultsVVisualizer, _extract_i_from_iuvarr() (+2 more)

### Community 32 - "ASAPNet Model"
Cohesion: 0.12
Nodes (5): ASAPNetModel, loss_filter(), InferenceModel, ASAPNet_RGBA, InferenceModel

### Community 33 - "apply_net Inference Action"
Cohesion: 0.15
Nodes (3): InferenceAction, ShowAction, CompoundVisualizer

### Community 34 - "DensePose Modeling Build"
Cohesion: 0.09
Nodes (8): build_densepose_data_filter(), build_densepose_embedder(), build_densepose_head(), build_densepose_losses(), build_densepose_predictor(), DensePoseDataFilter, densepose_inference(), Decoder

### Community 35 - "Human Pose Estimation"
Cohesion: 0.13
Nodes (5): HeatmapGenerator, parse_args(), HumanJointHeatmap, HumanPose, parse_args()

### Community 36 - "pix2pixHD Networks"
Cohesion: 0.27
Nodes (6): define_D(), define_G(), GANLoss, get_norm_layer(), VGGLoss, weights_init()

### Community 38 - "chart_output_to_chart_result.py"
Cohesion: 0.12
Nodes (9): densepose_chart_predictor_output_hflip(), _flip_iuv_semantics_tensor(), _flip_segm_semantics_tensor(), densepose_chart_predictor_output_to_result(), densepose_chart_predictor_output_to_result_with_confidences(), resample_confidences_to_bbox(), resample_uv_tensors_to_bbox(), resample_uv_to_bbox() (+1 more)

### Community 39 - "Inference-Based Loader"
Cohesion: 0.10
Nodes (7): build_bootstrap_dataset(), build_data_filter(), build_inference_based_loader(), build_inference_based_loaders(), _grouper(), InferenceBasedLoader, ScoreBasedFilter

### Community 40 - "torch_nn"
Cohesion: 0.08
Nodes (5): _create_pixel_dist_matrix(), _sample_fg_pixels_multinomial(), _sample_fg_pixels_randperm(), DataForMaskLoss, extract_data_for_mask_loss_from_matches()

### Community 41 - "Deeplab"
Cohesion: 0.11
Nodes (6): ASPP, ASPPConv, ASPPPooling, DensePoseDeepLabHead, NONLocalBlock2D, _NonLocalBlockND

### Community 42 - "coco.py"
Cohesion: 0.17
Nodes (15): _add_categories_metadata(), CocoDatasetInfo, _combine_images_with_annotations(), create_video_frame_mapping(), get_metadata(), _load_coco_annotations(), load_coco_json(), _maybe_add_bbox() (+7 more)

### Community 43 - "DensePoseCocoEval"
Cohesion: 0.05
Nodes (10): DensePoseCocoEval, _checkIgnore(), _summarize(), _summarizeDets(), _summarizeKps(), _summarizeUvs(), _summarizeUvsOld(), DensePoseDataMode (+2 more)

### Community 44 - "Mesh"
Cohesion: 0.20
Nodes (5): load_mesh_auxiliary_data(), load_mesh_data(), load_mesh_symmetry(), _maybe_copy_to_device(), Mesh

### Community 45 - "Random Affine Batch"
Cohesion: 0.17
Nodes (5): compute_inv_trans(), compute_trans(), RandomAffineBatch, RandomAffineBatchNumpy, RandomAffineMatrix

### Community 46 - "test_time_augmentation.py"
Cohesion: 0.06
Nodes (7): DensePoseGeneralizedRCNNWithTTA, _inverse_rotation(), rotate_box_inverse(), Average, OfflineFilter, PoseFilter, slerp()

### Community 48 - "build_data_sampler"
Cohesion: 0.15
Nodes (3): build_data_sampler(), PredictionToGroundTruthSampler, _Sampler

### Community 49 - "viton_upperbody.py"
Cohesion: 0.11
Nodes (12): GarmentCompositor, make_pix2pix_model(), read_meta(), TestOptions, get_inverse_trans(), TemporalSmoothing, find_module(), module_device() (+4 more)

### Community 50 - "converters/__init__.py"
Cohesion: 0.14
Nodes (9): BaseConverter, make_int_box(), HFlipConverter, predictor_output_with_coarse_segm_to_mask(), predictor_output_with_fine_and_coarse_segm_to_mask(), resample_fine_and_coarse_segm_to_bbox(), ToChartResultConverter, ToChartResultConverterWithConfidences (+1 more)

### Community 52 - "detectron2_config"
Cohesion: 0.12
Nodes (5): DensePoseConfidenceModelConfig, DensePoseSegmConfidenceConfig, DensePoseUVConfidenceConfig, DensePoseUVConfidenceType, initialize_module_params()

### Community 53 - "Networks"
Cohesion: 0.14
Nodes (4): GlobalGenerator, GlobalGenerator_Mask, LocalEnhancer, ResnetBlock

### Community 54 - "Apply Net"
Cohesion: 0.15
Nodes (4): Action, DumpAction, register_action(), get_palette()

### Community 55 - "Evaluator"
Cohesion: 0.17
Nodes (3): maybe_filter_and_map_categories_cocoapi(), DensePoseCOCOEvaluator, MeshAlignmentEvaluator

### Community 58 - "densepose_data_points.py"
Cohesion: 0.22
Nodes (8): PointsVisualizer, _densepose_data_i_for_cmap(), _densepose_data_u_for_cmap(), _densepose_data_v_for_cmap(), DensePoseDataPointsIVisualizer, DensePoseDataPointsUVisualizer, DensePoseDataPointsVisualizer, DensePoseDataPointsVVisualizer

### Community 59 - "Base 2"
Cohesion: 0.23
Nodes (4): RectangleVisualizer, TextVisualizer, BoundingBoxVisualizer, ScoredBoundingBoxVisualizer

### Community 61 - "flat_renderer.py"
Cohesion: 0.08
Nodes (12): BaseRenderer, _create_glfw_context(), create_initialized_headless_egl_display(), create_opengl_context(), destroy_opengl_context(), init_frame_buffer(), generate_vao(), read_texture() (+4 more)

### Community 62 - "tensor_storage.py"
Cohesion: 0.14
Nodes (8): _file_storage_gather(), MultiProcessFileTensorStorage, MultiProcessRamTensorStorage, MultiProcessTensorStorage, _ram_storage_gather(), SingleProcessFileTensorStorage, SingleProcessTensorStorage, storage_gather()

### Community 63 - "frame_selector.py"
Cohesion: 0.11
Nodes (6): build_frame_selector(), build_video_list_dataset(), FirstKFramesSelector, FrameSelectionStrategy, LastKFramesSelector, RandomKFramesSelector

### Community 64 - "Convlstm"
Cohesion: 0.20
Nodes (3): ConvLSTM, ConvLSTMCell, RNNGenerator

### Community 65 - "densepose_outputs_vertex.py"
Cohesion: 0.14
Nodes (6): create_mesh(), DensePoseOutputsTextureVisualizer, DensePoseOutputsVertexVisualizer, get_texture_atlases(), get_xyz_vertex_embedding(), get_texture_atlas()

### Community 67 - "losses/cse.py"
Cohesion: 0.14
Nodes (5): DensePoseCseLoss, PixToShapeCycleLoss, CseAnnotationsAccumulator, MaskLoss, MaskOrSegmentationLoss

### Community 70 - "Evaluator 2"
Cohesion: 0.23
Nodes (6): densepose_chart_predictions_to_storage_dict(), densepose_cse_predictions_to_dict(), _derive_results_from_coco_eval(), _evaluate_predictions_on_coco(), _get_densepose_metrics(), prediction_to_dict()

### Community 71 - "upperbody_inference.py"
Cohesion: 0.15
Nodes (4): BaseOptions, require_cuda(), cuda_sync(), now()

### Community 72 - "BilinearInterpolationHelper"
Cohesion: 0.10
Nodes (4): EmbeddingLoss, PackedCseAnnotations, SoftEmbeddingLoss, BilinearInterpolationHelper

### Community 73 - "densepose/__init__.py"
Cohesion: 0.27
Nodes (9): add_bootstrap_config(), add_dataset_category_config(), add_densepose_config(), add_densepose_head_config(), add_densepose_head_cse_config(), add_evaluation_config(), add_hrnet_config(), get_bootstrap_dataset_config() (+1 more)

### Community 80 - "DensePoseResultExtractor"
Cohesion: 0.24
Nodes (3): DensePoseExtractor, DensePoseOutputsExtractor, DensePoseResultExtractor

### Community 83 - "predictors/__init__.py"
Cohesion: 0.26
Nodes (3): DensePoseChartWithConfidencePredictor, DensePoseEmbeddingConfidencePredictorMixin, DensePoseEmbeddingWithConfidencePredictor

### Community 84 - "Cse Confidence"
Cohesion: 0.18
Nodes (4): decorate_cse_predictor_output_class_with_confidences(), PredictorOutput_getitem(), PredictorOutput_to(), slice_if_not_none()

### Community 85 - "combine_detection_dataset_dicts"
Cohesion: 0.23
Nodes (7): _add_category_id_to_contiguous_id_maps_to_metadata(), combine_detection_dataset_dicts(), _DatasetCategory, _map_category_id_to_contiguous_id(), _maybe_filter_and_map_categories(), _merge_categories(), _warn_if_merged_different_categories()

### Community 86 - "Jin 00 White Bg"
Cohesion: 0.35
Nodes (11): Jin Garment Series (jin_00..jin_26, 27 images), Upper-Body Garment Category, White-Background Garment Input Convention (headless front-view cutout, 1024x1024), jin_07 Light-Blue Pussy-Bow Blouse, jin_13 White Polka-Dot Short-Sleeve Shirt, Knit Sweater, Collared Button-Up Shirt / Blouse, lab_00 White Graphic Tee ('Full-time' print) (+3 more)

### Community 88 - "densepose_outputs_iuv.py"
Cohesion: 0.23
Nodes (4): DensePoseOutputsFineSegmentationVisualizer, DensePoseOutputsUVisualizer, DensePoseOutputsVisualizer, DensePoseOutputsVVisualizer

### Community 89 - "Detectron2COCOEvaluatorAdapter"
Cohesion: 0.24
Nodes (4): get_contiguous_id_to_category_id_map(), maybe_filter_categories_cocoapi(), Detectron2COCOEvaluatorAdapter, _maybe_add_iscrowd_annotations()

### Community 92 - "squared_euclidean_distance_matrix"
Cohesion: 0.24
Nodes (3): get_closest_vertices_mask_from_ES(), normalize_embeddings(), squared_euclidean_distance_matrix()

### Community 96 - "object"
Cohesion: 0.18
Nodes (4): BoundingBoxExtractor, NmsFilteredExtractor, ScoredBoundingBoxExtractor, ScoreThresholdedExtractor

### Community 102 - "Tensor Storage 3"
Cohesion: 0.22
Nodes (6): build_densepose_evaluator_storage(), _calculate_record_field_size_b(), _calculate_record_field_sizes_b(), _calculate_record_size_b(), SingleProcessRamTensorStorage, SizeData

### Community 107 - "structures/__init__.py"
Cohesion: 0.13
Nodes (7): compress_quantized_densepose_chart_result(), decompress_compressed_densepose_chart_result(), DensePoseChartResult, DensePoseChartResultCompressed, DensePoseChartResultQuantized, DensePoseChartResultWithConfidences, quantize_densepose_chart_result()

### Community 109 - "Pix2PixHD_RNN_RGBA"
Cohesion: 0.15
Nodes (3): InferenceModel, Pix2PixHD_RNN_RGBA, OpticalFlow

### Community 110 - "Screenshot"
Cohesion: 0.29
Nodes (7): board_300x300.png: Random Color Noise Pattern Board, Random Color Texture Pattern for Dense Correspondence / UV Calibration, Garment Mask Buffers (4 buffers, Save/Clear, history, prompt store, track), SAM 2.1 Hiera Large checkpoint (cuda/bfloat16), Screenshot.png: SAM2 Interactive Garment Masking/Tracking UI, SMPL Torso + Upper-arm UV Atlas (front, back, two sleeves), amputated_smpl_m_zero_uv.png: Amputated SMPL Upper-body UV Layout

### Community 115 - "decorate_predictor_output_class_with_confidences"
Cohesion: 0.40
Nodes (4): decorate_predictor_output_class_with_confidences(), PredictorOutput_getitem(), PredictorOutput_to(), slice_if_not_none()

### Community 124 - "Han"
Cohesion: 0.40
Nodes (5): Full-body Garment Replacement (traditional Hanfu over casual outfit), Han.gif: Hanfu Virtual Try-on Before/After Demo, none.png: 'None' / No-garment Prohibition Icon, Garment Selection Sidebar (icons incl. none option), output.gif: Real-time Virtual Try-on Mirror Display Demo

## Ambiguous Edges - Review These
- `Garment Mask Buffers (4 buffers, Save/Clear, history, prompt store, track)` → `SMPL Torso + Upper-arm UV Atlas (front, back, two sleeves)`  [AMBIGUOUS]
  assets/demo/Screenshot.png · relation: conceptually_related_to

## Knowledge Gaps
- **17 isolated node(s):** `Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting Garments (Wu, Shen, Igarashi, CGF 2025, arXiv 2506.12348)`, `SAM 2.1 Hiera Large checkpoint (cuda/bfloat16)`, `board_300x300.png: Random Color Noise Pattern Board`, `amputated_smpl_m_zero_uv.png: Amputated SMPL Upper-body UV Layout`, `Full-body Garment Replacement (traditional Hanfu over casual outfit)` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 852 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **78 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DensePoseCocoEval` connect `DensePoseCocoEval` to `resample_coarse_segm_tensor_to_bbox`, `Evaluator 2`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **What connects `Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting Garments (Wu, Shen, Igarashi, CGF 2025, arXiv 2506.12348)`, `SAM 2.1 Hiera Large checkpoint (cuda/bfloat16)`, `board_300x300.png: Random Color Noise Pattern Board` to the rest of the system?**
  _17 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `DensePose Data Build` be split into smaller, more focused modules?**
  _Cohesion score 0.07764705882352942 - nodes in this community are weakly interconnected._
- **Why does `DensePoseChartLoss` connect `DensePoseChartLoss` to `BilinearInterpolationHelper`, `losses/cse.py`, `ChartBasedAnnotationsAccumulator`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Should `SMPLModel` be split into smaller, more focused modules?**
  _Cohesion score 0.05480225988700565 - nodes in this community are weakly interconnected._
- **Why does `SMPLModel` connect `SMPLModel` to `upperbody_dataset_generation.py`, `DensePoseExtractor`, `BEV Dataset Generation`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Garment Mask Buffers (4 buffers, Save/Clear, history, prompt store, track)` and `SMPL Torso + Upper-arm UV Atlas (front, back, two sleeves)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._