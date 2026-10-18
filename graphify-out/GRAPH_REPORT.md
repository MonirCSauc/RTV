# Graph Report - RTV  (2026-10-18)

## Corpus Check
- Large corpus: 287 files · ~1,478,446 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 2515 nodes · 5573 edges · 165 communities (73 shown, 92 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 327 edges (avg confidence: 0.88)
- Token cost: 259,245 input · 0 output

## Community Hubs (Navigation)
- DensePose CSE Embedder & Mesh
- DensePose Data Build
- SMPL NumPy Model
- DensePose Chart Results
- Realtime Worker Threads
- pix2pixHD Inference Models
- DensePose Trainer & Eval
- COCO/LVIS Datasets
- Demo & Garment Removal
- Video Frame Selection
- Pixel Alignment & Landmarks
- Image Warp & Overlay
- DensePose Module Imports
- Human Parsing Utilities
- ConvLSTM & Quantization
- BEV Dataset Generation
- pix2pixHD Base Model
- DensePose Data Relative
- DensePose Chart Loss
- DensePose CSE Loss
- DensePose Garment Utils
- Garment Dataset & Options
- Docs, Papers & Training Pipeline
- Combined Data Loader
- HRNet Backbone
- Multithread Video IO
- DensePose Vis Extractors
- pix2pixHD Interactive UI
- Loss Annotation Accumulator
- apply_net Visualizers
- DensePose DB Helper
- DensePose Results Visualizer
- ASAPNet Model
- apply_net Inference Action
- DensePose Modeling Build
- Human Pose Estimation
- pix2pixHD Networks
- Segm-to-Mask Converter
- Chart Output Converter
- Inference-Based Loader
- Mask
- Deeplab
- Viton Upperbody
- Densepose Coco Evaluation
- Hrnet
- Random Affine Batch
- Config
- To Chart Result
- Prediction To Gt
- Demo
- To Chart Result 2
- Densepose Base
- Densepose Data Points
- Networks
- Apply Net
- Evaluator
- Cycle Shape2Shape
- Segm
- Base
- Base 2
- Utransformer Model
- Off Screen Render
- Tensor Storage
- Pose Filter
- Convlstm
- Chart With Confidences
- Video Loader
- Gl Utils
- Tensor Storage 2
- Roi Head
- Evaluator 2
- Confidence
- Embed
- Soft Embed
- Chart Confidence
- Densepose Outputs Iuv
- Test Time Augmentation
- Upperbody Garment
- Densepose Coco Evaluation 2
- Densepose Cse Uniform
- Test Time Augmentation 2
- Chart Output Hflip
- Densepose Cse Base
- Chart With Confidence
- Cse Confidence
- Pix2Pixhd Align
- Jin 00 White Bg
- Html2
- Smpl Renderer
- Build
- Chart
- Cse
- Densepose Outputs Vertex
- Image List Dataset
- V1Convx
- Densepose Results
- Gl Utils 2
- Inference
- Error Measure
- Base 3
- Image
- Densepose Coco Evaluation 3
- Tensor Storage 3
- Weak Perspective
- Densepose Coco Evaluation 5
- Densepose Results Textures
- Optical Flow
- Screenshot
- Densepose Confidence Based
- Networks 2
- Base Renderer
- Uv Renderer
- Pose Filter 2
- Trans Smooth
- Densepose Coco Evaluation 6
- Deeplab 2
- Chart Confidence 2
- Transform
- Large Video Loader
- Han
- Options
- Base Options
- Utils
- Roi Head 2
- Tensor Storage 4
- Networks 3
- Image2Video
- Gdrive Downloader
- Networks 4
- Claude
- Lazygit
- Install
- Upperbody Training

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
- `AmputatedSMPL` --uses--> `BaseRenderer`  [INFERRED]
  SMPL/amputated_smpl.py → OffscreenRenderer/base_renderer.py
- `AmputatedSMPL` --uses--> `FlatRenderer`  [INFERRED]
  SMPL/amputated_smpl.py → OffscreenRenderer/flat_renderer.py
- `FullBodySMPL` --uses--> `FlatRenderer`  [INFERRED]
  SMPL/fullbody_smpl/FullBody.py → OffscreenRenderer/flat_renderer.py
- `AmputatedSMPL` --uses--> `UVRenderer`  [INFERRED]
  SMPL/amputated_smpl.py → OffscreenRenderer/uv_renderer.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Per-garment checkpoint creation pipeline (record -> segment -> dataset -> train -> infer)** — training_instructions_video_recording, training_instructions_garment_segmentation, training_instructions_upperbody_dataset_generation, training_instructions_upperbody_training, training_instructions_upperbody_inference [EXTRACTED 1.00]
- **Pose-guided low-barrier dataset capture** — assets_pose_guidance_symmetric_pose_guide, assets_pose_guidance_asymmetric_pose_guide, training_instructions_video_recording, readme_paper_low_barrier_dataset_collection [INFERRED 0.85]
- **DensePose R-CNN FPN configuration stack** — model_densepose_configs_densepose_rcnn_r_50_fpn_s1x_config, model_densepose_configs_base_densepose_rcnn_fpn_config, model_densepose_configs_base_densepose_rcnn_fpn_generalized_rcnn, model_densepose_configs_base_densepose_rcnn_fpn_densepose_roi_heads [EXTRACTED 1.00]
- **Real-time try-on display with garment selection including 'none' option** — assets_output_mirror_display_demo, assets_output_garment_selection_sidebar, assets_none_no_garment_icon [INFERRED 0.85]
- **Garment data preparation: SAM2 garment masking, SMPL UV atlas, texture pattern board** — assets_demo_screenshot_sam2_mask_tool_ui, assets_smpl_amputated_smpl_m_zero_uv_smpl_torso_uv_map, assets_color_pattern_board_300x300_texture_correspondence_pattern [INFERRED 0.65]
- **Try-on garment input set: both series share the white-bg headless upper-body cutout format** — assets_garment_images_jin_00_white_bg_jin_garment_series, assets_garment_images_lab_00_white_bg_lab_garment_series, assets_garment_images_jin_00_white_bg_white_background_garment_input, assets_garment_images_jin_00_white_bg_upper_body_garment [INFERRED 0.85]

## Communities (165 total, 92 thin omitted)

### Community 0 - "DensePose CSE Embedder & Mesh"
Cohesion: 0.05
Nodes (10): create_embedder(), Embedder, EmbedderType, VertexDirectEmbedder, VertexFeatureEmbedder, load_mesh_auxiliary_data(), load_mesh_data(), load_mesh_symmetry() (+2 more)

### Community 1 - "DensePose Data Build"
Cohesion: 0.07
Nodes (32): _add_category_id_to_contiguous_id_maps_to_metadata(), _add_category_info_to_bootstrapping_metadata(), _add_category_maps_to_metadata(), _add_category_whitelists_to_metadata(), build_bootstrap_dataset(), build_combined_loader(), build_detection_test_loader(), build_detection_train_loader() (+24 more)

### Community 2 - "SMPL NumPy Model"
Cohesion: 0.06
Nodes (5): SMPLModel, mannequin2SMPL(), SMPLModel, load_obj(), save_obj()

### Community 3 - "DensePose Chart Results"
Cohesion: 0.07
Nodes (12): compress_quantized_densepose_chart_result(), decompress_compressed_densepose_chart_result(), DensePoseChartResult, DensePoseChartResultCompressed, DensePoseChartResultQuantized, quantize_densepose_chart_result(), DensePoseEmbeddingPredictorOutput, create_mesh() (+4 more)

### Community 4 - "Realtime Worker Threads"
Cohesion: 0.07
Nodes (11): VitonThread, DensePoseExtractor, DensePoseOutputsExtractor, DensePoseResultExtractor, FullBodySMPL, SMPL_Regressor, get_inverse_trans(), TemporalSmoothing (+3 more)

### Community 5 - "pix2pixHD Inference Models"
Cohesion: 0.07
Nodes (16): create_model(), InferenceModel_Align, InferenceModel, Pix2PixHD_DUAL_RGBA, InferenceModel_Inpaint, Pix2PixHDModel_Inpaint, InferenceModel_Mask, Pix2PixHDModel_Mask (+8 more)

### Community 6 - "DensePose Trainer & Eval"
Cohesion: 0.05
Nodes (9): get_contiguous_id_to_category_id_map(), maybe_filter_categories_cocoapi(), SampleCountingLoader, SampleCountMetricPrinter, Trainer, Detectron2COCOEvaluatorAdapter, _maybe_add_iscrowd_annotations(), DensePoseCheckpointer (+1 more)

### Community 7 - "COCO/LVIS Datasets"
Cohesion: 0.08
Nodes (28): build_augmentation(), _add_categories_metadata(), CocoDatasetInfo, _combine_images_with_annotations(), create_video_frame_mapping(), get_metadata(), _load_coco_annotations(), load_coco_json() (+20 more)

### Community 8 - "Demo & Garment Removal"
Cohesion: 0.06
Nodes (10): _blur(), _dilate(), _lut(), WornGarmentRemover, MultiHeadCrossAttention, MultiHeadSelfAttention, DualEncoderGenerator, ResnetBlock (+2 more)

### Community 9 - "Video Frame Selection"
Cohesion: 0.06
Nodes (11): build_frame_selector(), build_video_list_dataset(), FirstKFramesSelector, FrameSelectionStrategy, LastKFramesSelector, RandomKFramesSelector, list_keyframes(), read_keyframe_helper_data() (+3 more)

### Community 10 - "Pixel Alignment & Landmarks"
Cohesion: 0.06
Nodes (20): get_max_preds(), GetLandmarkCoord(), ImageCaption(), LandmarkAnnotation(), AABB, compute_iou(), get_threshold_mask(), JointAlignment (+12 more)

### Community 11 - "Image Warp & Overlay"
Cohesion: 0.11
Nodes (25): erode(), naive_overlay(), naive_overlay_alpha(), TestOptions, blur_image(), blur_image_by_mask(), poisson_blend(), poisson_blend_np() (+17 more)

### Community 12 - "DensePose Module Imports"
Cohesion: 0.13
Nodes (3): PixToShapeCycleLoss, PackedCseAnnotations, resample_data()

### Community 13 - "Human Parsing Utilities"
Cohesion: 0.09
Nodes (19): dilate(), get_smooth_2d(), make_kernel(), multiple_random_dilation(), random_dilate(), get_laplacian_matrix(), ImageCaption(), ImageCaptionBottom() (+11 more)

### Community 14 - "ConvLSTM & Quantization"
Cohesion: 0.09
Nodes (19): get_closest_vertices_mask_from_ES(), normalize_embeddings(), squared_euclidean_distance_matrix(), _create_pixel_dist_matrix(), _sample_fg_pixels_multinomial(), _sample_fg_pixels_randperm(), advect_numpy_image(), advect_numpy_image2() (+11 more)

### Community 15 - "BEV Dataset Generation"
Cohesion: 0.09
Nodes (9): gen_dataset(), make_video_loader(), process_video(), AmputatedSMPL, MyBEV, projection2screencoord(), MyROMP, check_type() (+1 more)

### Community 17 - "DensePose Data Relative"
Cohesion: 0.08
Nodes (3): _MeshCatalog, DensePoseDataRelative, DensePoseList

### Community 18 - "DensePose Chart Loss"
Cohesion: 0.11
Nodes (4): DensePoseChartLoss, DensePoseChartWithConfidenceLoss, BilinearInterpolationHelper, _linear_interpolation_utilities()

### Community 19 - "DensePose CSE Loss"
Cohesion: 0.08
Nodes (3): DensePoseCseLoss, CseAnnotationsAccumulator, MaskOrSegmentationLoss

### Community 20 - "DensePose Garment Utils"
Cohesion: 0.16
Nodes (18): crop(), main(), main(), crop(), IUV2Img(), IUV2SDP(), IUV2SSDP(), IUV2SSDP_deprecated() (+10 more)

### Community 21 - "Garment Dataset & Options"
Cohesion: 0.11
Nodes (9): UpperBodyGarment, BaseOptions, TrainOptions, lcm(), main(), mkdir(), mkdirs(), save_image() (+1 more)

### Community 22 - "Docs, Papers & Training Pipeline"
Cohesion: 0.10
Nodes (27): Asymmetric Pose Guidance (30 paired poses, one arm raised vs both), Symmetric Pose Guidance (16 predefined poses), Base-DensePose-RCNN-FPN config, DensePose COCO 2014 datasets, DensePoseROIHeads / DensePoseV1ConvXHead, GeneralizedRCNN with ResNet-FPN backbone, densepose_rcnn_R_50_FPN_s1x config (ResNet-50, 130k iters), detectron2 (+19 more)

### Community 23 - "Combined Data Loader"
Cohesion: 0.12
Nodes (8): CombinedDataLoader, _pooled_next(), register_dataset(), MeshInfo, register_mesh(), register_meshes(), is_relative_local_path(), maybe_prepend_base_path()

### Community 24 - "HRNet Backbone"
Cohesion: 0.09
Nodes (6): build_hrfpn_backbone(), HRFPN, BasicBlock, Bottleneck, build_pose_hrnet_backbone(), conv3x3()

### Community 25 - "Multithread Video IO"
Cohesion: 0.12
Nodes (5): process_video(), MultithreadVideoLoader, MultithreadVideoWriter, compute_confidence(), improve_masks_with_optical_flow()

### Community 26 - "DensePose Vis Extractors"
Cohesion: 0.11
Nodes (8): BoundingBoxExtractor, CompoundExtractor, create_extractor(), extract_boxes_xywh_from_instances(), extract_scores_from_instances(), NmsFilteredExtractor, ScoredBoundingBoxExtractor, ScoreThresholdedExtractor

### Community 27 - "pix2pixHD Interactive UI"
Cohesion: 0.14
Nodes (6): UIModel, Colorize, labelcolormap(), tensor2im(), tensor2label(), uint82bin()

### Community 28 - "Loss Annotation Accumulator"
Cohesion: 0.10
Nodes (5): AnnotationsAccumulator, ChartBasedAnnotationsAccumulator, extract_packed_annotations_from_matches(), PackedChartBasedAnnotations, sample_random_indices()

### Community 29 - "apply_net Visualizers"
Cohesion: 0.14
Nodes (5): create_argument_parser(), main(), verbosity_to_level(), dump_esample.sh script, example.sh script

### Community 30 - "DensePose DB Helper"
Cohesion: 0.12
Nodes (5): AllEntrySelector, EntrySelector, FieldEntrySelector, _FieldEntryRangePredicate, _FieldEntryValuePredicate

### Community 31 - "DensePose Results Visualizer"
Cohesion: 0.13
Nodes (5): DensePoseResultsCustomContourVisualizer, DensePoseResultsMplContourVisualizer, DensePoseResultsVisualizer, _extract_u_from_iuvarr(), _extract_v_from_iuvarr()

### Community 32 - "ASAPNet Model"
Cohesion: 0.12
Nodes (5): ASAPNetModel, loss_filter(), InferenceModel, ASAPNet_RGBA, InferenceModel

### Community 33 - "apply_net Inference Action"
Cohesion: 0.13
Nodes (4): InferenceAction, register_action(), ShowAction, CompoundVisualizer

### Community 34 - "DensePose Modeling Build"
Cohesion: 0.16
Nodes (6): build_densepose_data_filter(), build_densepose_embedder(), build_densepose_head(), build_densepose_losses(), build_densepose_predictor(), DensePoseDataFilter

### Community 35 - "Human Pose Estimation"
Cohesion: 0.13
Nodes (5): HeatmapGenerator, parse_args(), HumanJointHeatmap, HumanPose, parse_args()

### Community 36 - "pix2pixHD Networks"
Cohesion: 0.27
Nodes (6): define_D(), define_G(), GANLoss, get_norm_layer(), VGGLoss, weights_init()

### Community 37 - "Segm-to-Mask Converter"
Cohesion: 0.17
Nodes (6): make_int_box(), predictor_output_with_coarse_segm_to_mask(), predictor_output_with_fine_and_coarse_segm_to_mask(), resample_coarse_segm_tensor_to_bbox(), resample_fine_and_coarse_segm_tensors_to_bbox(), resample_fine_and_coarse_segm_to_bbox()

### Community 38 - "Chart Output Converter"
Cohesion: 0.17
Nodes (7): densepose_chart_predictor_output_to_result(), densepose_chart_predictor_output_to_result_with_confidences(), resample_confidences_to_bbox(), resample_uv_tensors_to_bbox(), resample_uv_to_bbox(), DensePoseChartPredictorOutput, DensePoseChartResultWithConfidences

### Community 39 - "Inference-Based Loader"
Cohesion: 0.13
Nodes (4): build_data_filter(), _grouper(), InferenceBasedLoader, ScoreBasedFilter

### Community 40 - "Mask"
Cohesion: 0.13
Nodes (3): DataForMaskLoss, extract_data_for_mask_loss_from_matches(), MaskLoss

### Community 41 - "Deeplab"
Cohesion: 0.15
Nodes (4): ASPP, ASPPConv, ASPPPooling, DensePoseDeepLabHead

### Community 42 - "Viton Upperbody"
Cohesion: 0.17
Nodes (4): FlatRenderer, UpperBodySMPL, FrameProcessor, soft_alpha()

### Community 45 - "Random Affine Batch"
Cohesion: 0.17
Nodes (5): compute_inv_trans(), compute_trans(), RandomAffineBatch, RandomAffineBatchNumpy, RandomAffineMatrix

### Community 46 - "Config"
Cohesion: 0.27
Nodes (9): add_bootstrap_config(), add_dataset_category_config(), add_densepose_config(), add_densepose_head_config(), add_densepose_head_cse_config(), add_evaluation_config(), add_hrnet_config(), get_bootstrap_dataset_config() (+1 more)

### Community 48 - "Prediction To Gt"
Cohesion: 0.13
Nodes (4): build_data_sampler(), DensePoseCSEConfidenceBasedSampler, PredictionToGroundTruthSampler, _Sampler

### Community 49 - "Demo"
Cohesion: 0.18
Nodes (4): CameraApp, get_camera_name_linux(), list_available_cameras(), main()

### Community 50 - "To Chart Result 2"
Cohesion: 0.27
Nodes (5): BaseConverter, HFlipConverter, ToChartResultConverter, ToChartResultConverterWithConfidences, ToMaskConverter

### Community 52 - "Densepose Data Points"
Cohesion: 0.22
Nodes (8): PointsVisualizer, _densepose_data_i_for_cmap(), _densepose_data_u_for_cmap(), _densepose_data_v_for_cmap(), DensePoseDataPointsIVisualizer, DensePoseDataPointsUVisualizer, DensePoseDataPointsVisualizer, DensePoseDataPointsVVisualizer

### Community 53 - "Networks"
Cohesion: 0.14
Nodes (4): GlobalGenerator, GlobalGenerator_Mask, LocalEnhancer, ResnetBlock

### Community 54 - "Apply Net"
Cohesion: 0.18
Nodes (3): Action, DumpAction, get_palette()

### Community 55 - "Evaluator"
Cohesion: 0.18
Nodes (3): maybe_filter_and_map_categories_cocoapi(), DensePoseCOCOEvaluator, MeshAlignmentEvaluator

### Community 59 - "Base 2"
Cohesion: 0.24
Nodes (4): RectangleVisualizer, TextVisualizer, BoundingBoxVisualizer, ScoredBoundingBoxVisualizer

### Community 61 - "Off Screen Render"
Cohesion: 0.36
Nodes (4): generate_vao(), read_texture(), render(), set_shader_params()

### Community 63 - "Pose Filter"
Cohesion: 0.18
Nodes (3): Average, PoseFilter, slerp()

### Community 64 - "Convlstm"
Cohesion: 0.20
Nodes (3): ConvLSTM, ConvLSTMCell, RNNGenerator

### Community 68 - "Tensor Storage 2"
Cohesion: 0.24
Nodes (6): _file_storage_gather(), MultiProcessFileTensorStorage, MultiProcessRamTensorStorage, _ram_storage_gather(), SingleProcessFileTensorStorage, storage_gather()

### Community 70 - "Evaluator 2"
Cohesion: 0.21
Nodes (6): densepose_chart_predictions_to_storage_dict(), densepose_cse_predictions_to_dict(), _derive_results_from_coco_eval(), _evaluate_predictions_on_coco(), _get_densepose_metrics(), prediction_to_dict()

### Community 71 - "Confidence"
Cohesion: 0.23
Nodes (4): DensePoseConfidenceModelConfig, DensePoseSegmConfidenceConfig, DensePoseUVConfidenceConfig, DensePoseUVConfidenceType

### Community 75 - "Densepose Outputs Iuv"
Cohesion: 0.23
Nodes (4): DensePoseOutputsFineSegmentationVisualizer, DensePoseOutputsUVisualizer, DensePoseOutputsVisualizer, DensePoseOutputsVVisualizer

### Community 76 - "Test Time Augmentation"
Cohesion: 0.18
Nodes (3): DensePoseDatasetMapperTTA, _inverse_rotation(), rotate_box_inverse()

### Community 77 - "Upperbody Garment"
Cohesion: 0.21
Nodes (3): get_random_affine_params(), RandomAffineMatrix, f()

### Community 79 - "Densepose Cse Uniform"
Cohesion: 0.27
Nodes (3): DensePoseCSEUniformSampler, DensePoseUniformSampler, MaskFromDensePoseSampler

### Community 81 - "Chart Output Hflip"
Cohesion: 0.23
Nodes (4): densepose_chart_predictor_output_hflip(), _flip_iuv_semantics_tensor(), _flip_segm_semantics_tensor(), DensePoseTransformData

### Community 83 - "Chart With Confidence"
Cohesion: 0.26
Nodes (3): DensePoseChartWithConfidencePredictor, DensePoseEmbeddingConfidencePredictorMixin, DensePoseEmbeddingWithConfidencePredictor

### Community 84 - "Cse Confidence"
Cohesion: 0.18
Nodes (4): decorate_cse_predictor_output_class_with_confidences(), PredictorOutput_getitem(), PredictorOutput_to(), slice_if_not_none()

### Community 86 - "Jin 00 White Bg"
Cohesion: 0.35
Nodes (11): Jin Garment Series (jin_00..jin_26, 27 images), Upper-Body Garment Category, White-Background Garment Input Convention (headless front-view cutout, 1024x1024), jin_07 Light-Blue Pussy-Bow Blouse, jin_13 White Polka-Dot Short-Sleeve Shirt, Knit Sweater, Collared Button-Up Shirt / Blouse, lab_00 White Graphic Tee ('Full-time' print) (+3 more)

### Community 95 - "Densepose Results"
Cohesion: 0.31
Nodes (5): DensePoseMaskedColormapResultsVisualizer, DensePoseResultsFineSegmentationVisualizer, DensePoseResultsUVisualizer, DensePoseResultsVVisualizer, _extract_i_from_iuvarr()

### Community 96 - "Gl Utils 2"
Cohesion: 0.24
Nodes (3): create_initialized_headless_egl_display(), create_opengl_context(), init_frame_buffer()

### Community 102 - "Tensor Storage 3"
Cohesion: 0.39
Nodes (4): _calculate_record_field_size_b(), _calculate_record_field_sizes_b(), _calculate_record_size_b(), SizeData

### Community 106 - "Densepose Coco Evaluation 5"
Cohesion: 0.39
Nodes (5): _summarize(), _summarizeDets(), _summarizeKps(), _summarizeUvs(), _summarizeUvsOld()

### Community 110 - "Screenshot"
Cohesion: 0.29
Nodes (7): board_300x300.png: Random Color Noise Pattern Board, Random Color Texture Pattern for Dense Correspondence / UV Calibration, Garment Mask Buffers (4 buffers, Save/Clear, history, prompt store, track), SAM 2.1 Hiera Large checkpoint (cuda/bfloat16), Screenshot.png: SAM2 Interactive Garment Masking/Tracking UI, SMPL Torso + Upper-arm UV Atlas (front, back, two sleeves), amputated_smpl_m_zero_uv.png: Amputated SMPL Upper-body UV Layout

### Community 121 - "Chart Confidence 2"
Cohesion: 0.40
Nodes (4): decorate_predictor_output_class_with_confidences(), PredictorOutput_getitem(), PredictorOutput_to(), slice_if_not_none()

### Community 124 - "Han"
Cohesion: 0.40
Nodes (5): Full-body Garment Replacement (traditional Hanfu over casual outfit), Han.gif: Hanfu Virtual Try-on Before/After Demo, none.png: 'None' / No-garment Prohibition Icon, Garment Selection Sidebar (icons incl. none option), output.gif: Real-time Virtual Try-on Mirror Display Demo

## Ambiguous Edges - Review These
- `Apache 2.0-based Non-Commercial License` → `rtv_ckpts Model Card (apache-2.0)`  [AMBIGUOUS]
  rtv_ckpts/README.md · relation: conceptually_related_to
- `Garment Mask Buffers (4 buffers, Save/Clear, history, prompt store, track)` → `SMPL Torso + Upper-arm UV Atlas (front, back, two sleeves)`  [AMBIGUOUS]
  assets/demo/Screenshot.png · relation: conceptually_related_to

## Knowledge Gaps
- **16 isolated node(s):** `Graphify Knowledge Graph Usage Rules`, `graphify update (AST-only rebuild)`, `Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting Garments (Wu, Shen, Igarashi, CGF 2025, arXiv 2506.12348)`, `simple_romp (ROMP fork)`, `muggled_sam segmentation tool` (+11 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 842 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **92 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DensePoseCocoEval` connect `Densepose Coco Evaluation` to `Densepose Coco Evaluation 3`, `Evaluator 2`, `Densepose Coco Evaluation 4`, `Densepose Coco Evaluation 5`, `Densepose Coco Evaluation 2`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **What connects `Graphify Knowledge Graph Usage Rules`, `graphify update (AST-only rebuild)`, `Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting Garments (Wu, Shen, Igarashi, CGF 2025, arXiv 2506.12348)` to the rest of the system?**
  _16 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `DensePose CSE Embedder & Mesh` be split into smaller, more focused modules?**
  _Cohesion score 0.05136612021857923 - nodes in this community are weakly interconnected._
- **Why does `BilinearInterpolationHelper` connect `DensePose Chart Loss` to `Embed`, `Soft Embed`, `DensePose Module Imports`, `DensePose CSE Loss`, `Loss Annotation Accumulator`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Should `DensePose Data Build` be split into smaller, more focused modules?**
  _Cohesion score 0.07380520266182698 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Apache 2.0-based Non-Commercial License` and `rtv_ckpts Model Card (apache-2.0)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Garment Mask Buffers (4 buffers, Save/Clear, history, prompt store, track)` and `SMPL Torso + Upper-arm UV Atlas (front, back, two sleeves)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._