# Graph Report - RTV  (2026-10-18)

## Corpus Check
- 237 files · ~1,478,608 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 12 file(s) not represented in the graph (top: .obj 6, .npz 3, (none) 2)

## Summary
- 2516 nodes · 5582 edges · 174 communities (81 shown, 93 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 326 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `880c8c09`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- embedder.py
- DensePose Data Build
- SMPL NumPy Model
- typing
- DensePoseExtractor
- pix2pixHD Inference Models
- DensePose Trainer & Eval
- lvis.py
- Demo & Garment Removal
- data/build.py
- Pixel Alignment & Landmarks
- viton_upperbody.py
- DensePose Module Imports
- cv2
- torch
- BEV Dataset Generation
- pix2pixHD Base Model
- DensePoseDataRelative
- DensePose Chart Loss
- DensePoseCseLoss
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
- chart_output_to_chart_result.py
- Inference-Based Loader
- .forward
- Deeplab
- coco.py
- Densepose Coco Evaluation
- Mesh
- Random Affine Batch
- densepose/__init__.py
- samplers/__init__.py
- rtl_demo.py
- BaseConverter
- Densepose Base
- Densepose Data Points
- Networks
- Apply Net
- Evaluator
- Cycle Shape2Shape
- MaskOrSegmentationLoss
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
- extract_packed_annotations_from_matches
- Evaluator 2
- Confidence
- Embed
- Soft Embed
- Chart Confidence
- Densepose Outputs Iuv
- Trainer
- Upperbody Garment
- Densepose Coco Evaluation 2
- DensePoseList
- DensePoseResultExtractor
- Chart Output Hflip
- Densepose Cse Base
- Chart With Confidence
- Cse Confidence
- combine_detection_dataset_dicts
- Jin 00 White Bg
- CseAnnotationsAccumulator
- Build
- Chart
- create_mesh
- Image List Dataset
- V1Convx
- Densepose Results
- Gl Utils 2
- Detectron2COCOEvaluatorAdapter
- Error Measure
- DensePoseEmbeddingPredictorOutput
- DensePoseResultsVisualizer
- Tensor Storage 3
- Weak Perspective
- object
- Densepose Coco Evaluation 5
- DensePoseChartResult
- Densepose Results Textures
- Pix2PixHD_RNN_RGBA
- Screenshot
- read_keyframes
- Densepose Cse Confidence Based
- Networks 2
- Base Renderer
- Uv Renderer
- Pose Filter 2
- Trans Smooth
- skeleton_annotator.py
- Densepose Coco Evaluation 6
- combined_loader.py
- Chart Confidence 2
- Embedder
- Large Video Loader
- Han
- Options
- Base Options
- predictors/chart_confidence.py
- read_keyframe_helper_data
- DumpAction
- DensePoseChartPredictorOutput
- Networks 3
- Image2Video
- Gdrive Downloader
- ._extract_iuv
- Claude
- Lazygit
- Install
- Upperbody Training
- ScaleTransOptimizer
- build_bootstrap_dataset
- advect_img.py
- JointAlignment
- _MeshCatalog
- SMPL_Renderer
- _load_coco_annotations

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
- `UpperBodySMPL` --uses--> `FlatRenderer`  [INFERRED]
  SMPL/upperbody_smpl/UpperBody.py → OffscreenRenderer/flat_renderer.py
- `AmputatedSMPL` --uses--> `UVRenderer`  [INFERRED]
  SMPL/amputated_smpl.py → OffscreenRenderer/uv_renderer.py
- `FullBodySeqFrameProcessor` --uses--> `FullBodySMPL`  [INFERRED]
  VITON/viton_fullbody_seq.py → SMPL/fullbody_smpl/FullBody.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **DensePose R-CNN FPN configuration stack** — model_densepose_configs_densepose_rcnn_r_50_fpn_s1x_config, model_densepose_configs_base_densepose_rcnn_fpn_config, model_densepose_configs_base_densepose_rcnn_fpn_generalized_rcnn, model_densepose_configs_base_densepose_rcnn_fpn_densepose_roi_heads [EXTRACTED 1.00]
- **Per-garment checkpoint creation pipeline (record -> segment -> dataset -> train -> infer)** — training_instructions_video_recording, training_instructions_garment_segmentation, training_instructions_upperbody_dataset_generation, training_instructions_upperbody_training, training_instructions_upperbody_inference [EXTRACTED 1.00]
- **Garment data preparation: SAM2 garment masking, SMPL UV atlas, texture pattern board** — assets_demo_screenshot_sam2_mask_tool_ui, assets_smpl_amputated_smpl_m_zero_uv_smpl_torso_uv_map, assets_color_pattern_board_300x300_texture_correspondence_pattern [INFERRED 0.65]
- **Try-on garment input set: both series share the white-bg headless upper-body cutout format** — assets_garment_images_jin_00_white_bg_jin_garment_series, assets_garment_images_lab_00_white_bg_lab_garment_series, assets_garment_images_jin_00_white_bg_white_background_garment_input, assets_garment_images_jin_00_white_bg_upper_body_garment [INFERRED 0.85]
- **Pose-guided low-barrier dataset capture** — assets_pose_guidance_symmetric_pose_guide, assets_pose_guidance_asymmetric_pose_guide, training_instructions_video_recording, readme_paper_low_barrier_dataset_collection [INFERRED 0.85]
- **Real-time try-on display with garment selection including 'none' option** — assets_output_mirror_display_demo, assets_output_garment_selection_sidebar, assets_none_no_garment_icon [INFERRED 0.85]

## Communities (174 total, 93 thin omitted)

### Community 0 - "embedder.py"
Cohesion: 0.14
Nodes (5): create_embedder(), EmbedderType, normalize_embeddings(), VertexDirectEmbedder, VertexFeatureEmbedder

### Community 1 - "DensePose Data Build"
Cohesion: 0.11
Nodes (19): _add_category_maps_to_metadata(), _add_category_whitelists_to_metadata(), build_combined_loader(), build_detection_test_loader(), build_detection_train_loader(), _compute_num_images_per_worker(), _get_test_keep_instance_predicate(), _get_train_keep_instance_predicate() (+11 more)

### Community 2 - "SMPL NumPy Model"
Cohesion: 0.06
Nodes (4): SMPLModel, mannequin2SMPL(), SMPLModel, save_obj()

### Community 4 - "DensePoseExtractor"
Cohesion: 0.07
Nodes (12): naive_overlay_alpha(), gen_dataset(), process_video(), DensePoseExtractor, SMPL_Regressor, UpperBodySMPL, check_type(), get_file_path_list() (+4 more)

### Community 5 - "pix2pixHD Inference Models"
Cohesion: 0.07
Nodes (15): create_model(), InferenceModel_Align, Pix2PixHDModel_Align, InferenceModel, Pix2PixHD_DUAL_RGBA, InferenceModel_Inpaint, Pix2PixHDModel_Inpaint, InferenceModel_Mask (+7 more)

### Community 6 - "DensePose Trainer & Eval"
Cohesion: 0.08
Nodes (4): SampleCountingLoader, SampleCountMetricPrinter, DensePoseCheckpointer, _rename_HRNet_weights()

### Community 7 - "lvis.py"
Cohesion: 0.17
Nodes (12): _add_categories_metadata(), _combine_images_with_annotations(), _load_lvis_annotations(), load_lvis_json(), _maybe_add_bbox(), _maybe_add_densepose(), _maybe_add_keypoints(), _maybe_add_segm() (+4 more)

### Community 8 - "Demo & Garment Removal"
Cohesion: 0.06
Nodes (10): _blur(), _dilate(), _lut(), WornGarmentRemover, MultiHeadCrossAttention, MultiHeadSelfAttention, DualEncoderGenerator, ResnetBlock (+2 more)

### Community 9 - "data/build.py"
Cohesion: 0.09
Nodes (9): build_frame_selector(), build_transform(), build_video_list_dataset(), ImageResizeTransform, FirstKFramesSelector, FrameSelectionStrategy, LastKFramesSelector, RandomKFramesSelector (+1 more)

### Community 10 - "Pixel Alignment & Landmarks"
Cohesion: 0.08
Nodes (28): erode(), dilate(), get_smooth_2d(), make_kernel(), multiple_random_dilation(), random_dilate(), get_laplacian_matrix(), ImageCaption() (+20 more)

### Community 11 - "viton_upperbody.py"
Cohesion: 0.13
Nodes (17): naive_overlay(), FlatRenderer, TestOptions, FullBodySMPL, get_inverse_trans(), TemporalSmoothing, center_crop(), rotate_image() (+9 more)

### Community 12 - "DensePose Module Imports"
Cohesion: 0.12
Nodes (4): DensePoseChartWithConfidenceLoss, PackedCseAnnotations, BilinearInterpolationHelper, resample_data()

### Community 13 - "cv2"
Cohesion: 0.09
Nodes (16): get_camera_name_linux(), list_available_cameras(), main(), blur_image(), blur_image_by_mask(), poisson_blend(), poisson_blend_np(), quantimage() (+8 more)

### Community 14 - "torch"
Cohesion: 0.19
Nodes (9): diff_tex_sampling(), hsl2rgb_torch(), hsv2rgb_torch(), image_quantization(), img2tensor(), measurement_quantization(), rgb2hsl_torch(), rgb2hsv_torch() (+1 more)

### Community 15 - "BEV Dataset Generation"
Cohesion: 0.13
Nodes (4): MyBEV, projection2screencoord(), MyROMP, load_obj()

### Community 20 - "DensePose Garment Utils"
Cohesion: 0.15
Nodes (19): crop(), main(), main(), crop(), IUV2HandHeadMask(), IUV2Img(), IUV2SDP(), IUV2SSDP() (+11 more)

### Community 21 - "Garment Dataset & Options"
Cohesion: 0.07
Nodes (12): BaseOptions, TrainOptions, lcm(), main(), HTML, Colorize, labelcolormap(), mkdir() (+4 more)

### Community 22 - "Docs, Papers & Training Pipeline"
Cohesion: 0.10
Nodes (26): Asymmetric Pose Guidance (30 paired poses, one arm raised vs both), Symmetric Pose Guidance (16 predefined poses), Base-DensePose-RCNN-FPN config, DensePose COCO 2014 datasets, DensePoseROIHeads / DensePoseV1ConvXHead, GeneralizedRCNN with ResNet-FPN backbone, densepose_rcnn_R_50_FPN_s1x config (ResNet-50, 130k iters), detectron2 (+18 more)

### Community 23 - "Combined Data Loader"
Cohesion: 0.14
Nodes (8): register_dataset(), DatasetType, MeshInfo, register_mesh(), register_meshes(), is_relative_local_path(), maybe_prepend_base_path(), video_list_from_file()

### Community 24 - "HRNet Backbone"
Cohesion: 0.06
Nodes (8): build_hrfpn_backbone(), HRFPN, BasicBlock, Bottleneck, build_pose_hrnet_backbone(), conv3x3(), HighResolutionModule, PoseHigherResolutionNet

### Community 25 - "Multithread Video IO"
Cohesion: 0.12
Nodes (6): make_video_loader(), process_video(), MultithreadVideoLoader, MultithreadVideoWriter, compute_confidence(), improve_masks_with_optical_flow()

### Community 26 - "DensePose Vis Extractors"
Cohesion: 0.20
Nodes (4): CompoundExtractor, create_extractor(), extract_boxes_xywh_from_instances(), extract_scores_from_instances()

### Community 27 - "pix2pixHD Interactive UI"
Cohesion: 0.22
Nodes (3): UIModel, tensor2im(), tensor2label()

### Community 28 - "Loss Annotation Accumulator"
Cohesion: 0.13
Nodes (3): ChartBasedAnnotationsAccumulator, PackedChartBasedAnnotations, sample_random_indices()

### Community 29 - "apply_net Visualizers"
Cohesion: 0.14
Nodes (5): create_argument_parser(), main(), verbosity_to_level(), dump_esample.sh script, example.sh script

### Community 30 - "DensePose DB Helper"
Cohesion: 0.12
Nodes (5): AllEntrySelector, EntrySelector, FieldEntrySelector, _FieldEntryRangePredicate, _FieldEntryValuePredicate

### Community 31 - "DensePose Results Visualizer"
Cohesion: 0.22
Nodes (3): DensePoseResultsCustomContourVisualizer, _extract_u_from_iuvarr(), _extract_v_from_iuvarr()

### Community 32 - "ASAPNet Model"
Cohesion: 0.12
Nodes (5): ASAPNetModel, loss_filter(), InferenceModel, ASAPNet_RGBA, InferenceModel

### Community 33 - "apply_net Inference Action"
Cohesion: 0.16
Nodes (3): InferenceAction, ShowAction, CompoundVisualizer

### Community 34 - "DensePose Modeling Build"
Cohesion: 0.09
Nodes (9): get_class_to_mesh_name_mapping(), build_densepose_data_filter(), build_densepose_embedder(), build_densepose_head(), build_densepose_losses(), build_densepose_predictor(), DensePoseDataFilter, densepose_inference() (+1 more)

### Community 35 - "Human Pose Estimation"
Cohesion: 0.14
Nodes (5): HeatmapGenerator, parse_args(), HumanJointHeatmap, HumanPose, parse_args()

### Community 36 - "pix2pixHD Networks"
Cohesion: 0.22
Nodes (7): define_D(), define_G(), Encoder, GANLoss, get_norm_layer(), VGGLoss, weights_init()

### Community 37 - "Segm-to-Mask Converter"
Cohesion: 0.12
Nodes (10): make_int_box(), densepose_chart_predictor_output_to_result(), densepose_chart_predictor_output_to_result_with_confidences(), HFlipConverter, predictor_output_with_coarse_segm_to_mask(), predictor_output_with_fine_and_coarse_segm_to_mask(), resample_coarse_segm_tensor_to_bbox(), resample_fine_and_coarse_segm_tensors_to_bbox() (+2 more)

### Community 38 - "chart_output_to_chart_result.py"
Cohesion: 0.31
Nodes (3): resample_confidences_to_bbox(), resample_uv_tensors_to_bbox(), resample_uv_to_bbox()

### Community 39 - "Inference-Based Loader"
Cohesion: 0.11
Nodes (6): build_data_filter(), build_inference_based_loader(), build_inference_based_loaders(), _grouper(), InferenceBasedLoader, ScoreBasedFilter

### Community 40 - ".forward"
Cohesion: 0.18
Nodes (4): _create_pixel_dist_matrix(), PixToShapeCycleLoss, _sample_fg_pixels_multinomial(), _sample_fg_pixels_randperm()

### Community 41 - "Deeplab"
Cohesion: 0.11
Nodes (6): ASPP, ASPPConv, ASPPPooling, DensePoseDeepLabHead, NONLocalBlock2D, _NonLocalBlockND

### Community 42 - "coco.py"
Cohesion: 0.21
Nodes (14): _add_categories_metadata(), CocoDatasetInfo, _combine_images_with_annotations(), create_video_frame_mapping(), get_metadata(), load_coco_json(), _maybe_add_bbox(), _maybe_add_densepose() (+6 more)

### Community 44 - "Mesh"
Cohesion: 0.19
Nodes (5): load_mesh_auxiliary_data(), load_mesh_data(), load_mesh_symmetry(), _maybe_copy_to_device(), Mesh

### Community 45 - "Random Affine Batch"
Cohesion: 0.17
Nodes (5): compute_inv_trans(), compute_trans(), RandomAffineBatch, RandomAffineBatchNumpy, RandomAffineMatrix

### Community 46 - "densepose/__init__.py"
Cohesion: 0.05
Nodes (17): add_bootstrap_config(), add_dataset_category_config(), add_densepose_config(), add_densepose_head_config(), add_densepose_head_cse_config(), add_evaluation_config(), add_hrnet_config(), get_bootstrap_dataset_config() (+9 more)

### Community 48 - "samplers/__init__.py"
Cohesion: 0.08
Nodes (7): build_data_sampler(), DensePoseConfidenceBasedSampler, DensePoseCSEUniformSampler, DensePoseUniformSampler, MaskFromDensePoseSampler, PredictionToGroundTruthSampler, _Sampler

### Community 49 - "rtl_demo.py"
Cohesion: 0.13
Nodes (7): CameraApp, VitonThread, crop2_169(), crop2_43(), crop_resize2size(), resize_img(), FullBodySeqFrameProcessor

### Community 50 - "BaseConverter"
Cohesion: 0.18
Nodes (4): BaseConverter, wrapper(), ToChartResultConverter, ToChartResultConverterWithConfidences

### Community 52 - "Densepose Data Points"
Cohesion: 0.26
Nodes (8): PointsVisualizer, _densepose_data_i_for_cmap(), _densepose_data_u_for_cmap(), _densepose_data_v_for_cmap(), DensePoseDataPointsIVisualizer, DensePoseDataPointsUVisualizer, DensePoseDataPointsVisualizer, DensePoseDataPointsVVisualizer

### Community 53 - "Networks"
Cohesion: 0.14
Nodes (4): GlobalGenerator, GlobalGenerator_Mask, LocalEnhancer, ResnetBlock

### Community 55 - "Evaluator"
Cohesion: 0.17
Nodes (4): maybe_filter_and_map_categories_cocoapi(), DensePoseCOCOEvaluator, MeshAlignmentEvaluator, f()

### Community 57 - "MaskOrSegmentationLoss"
Cohesion: 0.09
Nodes (5): DataForMaskLoss, extract_data_for_mask_loss_from_matches(), MaskLoss, MaskOrSegmentationLoss, SegmentationLoss

### Community 59 - "Base 2"
Cohesion: 0.22
Nodes (4): RectangleVisualizer, TextVisualizer, BoundingBoxVisualizer, ScoredBoundingBoxVisualizer

### Community 61 - "Off Screen Render"
Cohesion: 0.37
Nodes (4): generate_vao(), read_texture(), render(), set_shader_params()

### Community 62 - "Tensor Storage"
Cohesion: 0.17
Nodes (3): MultiProcessTensorStorage, SingleProcessFileTensorStorage, SingleProcessTensorStorage

### Community 63 - "Pose Filter"
Cohesion: 0.18
Nodes (3): Average, PoseFilter, slerp()

### Community 64 - "Convlstm"
Cohesion: 0.20
Nodes (3): ConvLSTM, ConvLSTMCell, RNNGenerator

### Community 68 - "Tensor Storage 2"
Cohesion: 0.29
Nodes (4): _file_storage_gather(), MultiProcessFileTensorStorage, _ram_storage_gather(), storage_gather()

### Community 69 - "extract_packed_annotations_from_matches"
Cohesion: 0.19
Nodes (3): AnnotationsAccumulator, extract_packed_annotations_from_matches(), _linear_interpolation_utilities()

### Community 70 - "Evaluator 2"
Cohesion: 0.19
Nodes (6): densepose_chart_predictions_to_storage_dict(), densepose_cse_predictions_to_dict(), _derive_results_from_coco_eval(), _evaluate_predictions_on_coco(), _get_densepose_metrics(), prediction_to_dict()

### Community 71 - "Confidence"
Cohesion: 0.23
Nodes (4): DensePoseConfidenceModelConfig, DensePoseSegmConfidenceConfig, DensePoseUVConfidenceConfig, DensePoseUVConfidenceType

### Community 75 - "Densepose Outputs Iuv"
Cohesion: 0.23
Nodes (4): DensePoseOutputsFineSegmentationVisualizer, DensePoseOutputsUVisualizer, DensePoseOutputsVisualizer, DensePoseOutputsVVisualizer

### Community 77 - "Upperbody Garment"
Cohesion: 0.22
Nodes (3): get_random_affine_params(), RandomAffineMatrix, UpperBodyGarment

### Community 83 - "Chart With Confidence"
Cohesion: 0.15
Nodes (4): DensePoseChartWithConfidencePredictor, DensePoseEmbeddingConfidencePredictorMixin, DensePoseEmbeddingPredictor, DensePoseEmbeddingWithConfidencePredictor

### Community 84 - "Cse Confidence"
Cohesion: 0.18
Nodes (4): decorate_cse_predictor_output_class_with_confidences(), PredictorOutput_getitem(), PredictorOutput_to(), slice_if_not_none()

### Community 85 - "combine_detection_dataset_dicts"
Cohesion: 0.23
Nodes (7): _add_category_id_to_contiguous_id_maps_to_metadata(), combine_detection_dataset_dicts(), _DatasetCategory, _map_category_id_to_contiguous_id(), _maybe_filter_and_map_categories(), _merge_categories(), _warn_if_merged_different_categories()

### Community 86 - "Jin 00 White Bg"
Cohesion: 0.35
Nodes (11): Jin Garment Series (jin_00..jin_26, 27 images), Upper-Body Garment Category, White-Background Garment Input Convention (headless front-view cutout, 1024x1024), jin_07 Light-Blue Pussy-Bow Blouse, jin_13 White Polka-Dot Short-Sleeve Shirt, Knit Sweater, Collared Button-Up Shirt / Blouse, lab_00 White Graphic Tee ('Full-time' print) (+3 more)

### Community 92 - "create_mesh"
Cohesion: 0.16
Nodes (6): get_closest_vertices_mask_from_ES(), squared_euclidean_distance_matrix(), create_mesh(), DensePoseOutputsTextureVisualizer, DensePoseOutputsVertexVisualizer, get_xyz_vertex_embedding()

### Community 95 - "Densepose Results"
Cohesion: 0.27
Nodes (5): DensePoseMaskedColormapResultsVisualizer, DensePoseResultsFineSegmentationVisualizer, DensePoseResultsUVisualizer, DensePoseResultsVVisualizer, _extract_i_from_iuvarr()

### Community 96 - "Gl Utils 2"
Cohesion: 0.24
Nodes (3): _create_glfw_context(), create_opengl_context(), init_frame_buffer()

### Community 97 - "Detectron2COCOEvaluatorAdapter"
Cohesion: 0.24
Nodes (4): get_contiguous_id_to_category_id_map(), maybe_filter_categories_cocoapi(), Detectron2COCOEvaluatorAdapter, _maybe_add_iscrowd_annotations()

### Community 98 - "Error Measure"
Cohesion: 0.23
Nodes (3): get_texture_atlases(), ErrorMeasure, json_wrtiter()

### Community 102 - "Tensor Storage 3"
Cohesion: 0.20
Nodes (7): build_densepose_evaluator_storage(), _calculate_record_field_size_b(), _calculate_record_field_sizes_b(), _calculate_record_size_b(), MultiProcessRamTensorStorage, SingleProcessRamTensorStorage, SizeData

### Community 105 - "object"
Cohesion: 0.18
Nodes (4): BoundingBoxExtractor, NmsFilteredExtractor, ScoredBoundingBoxExtractor, ScoreThresholdedExtractor

### Community 106 - "Densepose Coco Evaluation 5"
Cohesion: 0.39
Nodes (5): _summarize(), _summarizeDets(), _summarizeKps(), _summarizeUvs(), _summarizeUvsOld()

### Community 107 - "DensePoseChartResult"
Cohesion: 0.09
Nodes (8): densepose_chart_predictions_to_dict(), compress_quantized_densepose_chart_result(), decompress_compressed_densepose_chart_result(), DensePoseChartResult, DensePoseChartResultCompressed, DensePoseChartResultQuantized, DensePoseChartResultWithConfidences, quantize_densepose_chart_result()

### Community 109 - "Pix2PixHD_RNN_RGBA"
Cohesion: 0.15
Nodes (3): InferenceModel, Pix2PixHD_RNN_RGBA, OpticalFlow

### Community 110 - "Screenshot"
Cohesion: 0.29
Nodes (7): board_300x300.png: Random Color Noise Pattern Board, Random Color Texture Pattern for Dense Correspondence / UV Calibration, Garment Mask Buffers (4 buffers, Save/Clear, history, prompt store, track), SAM 2.1 Hiera Large checkpoint (cuda/bfloat16), Screenshot.png: SAM2 Interactive Garment Masking/Tracking UI, SMPL Torso + Upper-arm UV Atlas (front, back, two sleeves), amputated_smpl_m_zero_uv.png: Amputated SMPL Upper-body UV Layout

### Community 118 - "skeleton_annotator.py"
Cohesion: 0.29
Nodes (3): GetJointCoord(), SkeletonAnnotation(), SkeletonTensor()

### Community 121 - "Chart Confidence 2"
Cohesion: 0.40
Nodes (4): decorate_predictor_output_class_with_confidences(), PredictorOutput_getitem(), PredictorOutput_to(), slice_if_not_none()

### Community 124 - "Han"
Cohesion: 0.40
Nodes (5): Full-body Garment Replacement (traditional Hanfu over casual outfit), Han.gif: Hanfu Virtual Try-on Before/After Demo, none.png: 'None' / No-garment Prohibition Icon, Garment Selection Sidebar (icons incl. none option), output.gif: Real-time Virtual Try-on Mirror Display Demo

### Community 130 - "DumpAction"
Cohesion: 0.29
Nodes (3): DumpAction, register_action(), get_palette()

### Community 132 - "DensePoseChartPredictorOutput"
Cohesion: 0.16
Nodes (4): densepose_chart_predictor_output_hflip(), _flip_iuv_semantics_tensor(), _flip_segm_semantics_tensor(), DensePoseChartPredictorOutput

### Community 166 - "build_bootstrap_dataset"
Cohesion: 0.33
Nodes (3): _add_category_info_to_bootstrapping_metadata(), build_bootstrap_dataset(), get_category_to_class_mapping()

### Community 169 - "advect_img.py"
Cohesion: 0.53
Nodes (4): advect_numpy_image(), advect_numpy_image2(), advect_tensor_image(), mask_smooth()

## Ambiguous Edges - Review These
- `Garment Mask Buffers (4 buffers, Save/Clear, history, prompt store, track)` → `SMPL Torso + Upper-arm UV Atlas (front, back, two sleeves)`  [AMBIGUOUS]
  assets/demo/Screenshot.png · relation: conceptually_related_to

## Knowledge Gaps
- **17 isolated node(s):** `Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting Garments (Wu, Shen, Igarashi, CGF 2025, arXiv 2506.12348)`, `SAM 2.1 Hiera Large checkpoint (cuda/bfloat16)`, `board_300x300.png: Random Color Noise Pattern Board`, `amputated_smpl_m_zero_uv.png: Amputated SMPL Upper-body UV Layout`, `Full-body Garment Replacement (traditional Hanfu over casual outfit)` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 843 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **93 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DensePoseChartLoss` connect `DensePose Chart Loss` to `Chart With Confidences`, `DensePose Module Imports`, `Loss Annotation Accumulator`, `MaskOrSegmentationLoss`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **What connects `Real-Time Per-Garment Virtual Try-On with Temporal Consistency for Loose-Fitting Garments (Wu, Shen, Igarashi, CGF 2025, arXiv 2506.12348)`, `SAM 2.1 Hiera Large checkpoint (cuda/bfloat16)`, `board_300x300.png: Random Color Noise Pattern Board` to the rest of the system?**
  _17 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `embedder.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14492753623188406 - nodes in this community are weakly interconnected._
- **Why does `initialize_module_params()` connect `predictors/chart_confidence.py` to `DensePose Modeling Build`, `Chart Confidence`, `Chart With Confidence`, `Chart`, `V1Convx`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Should `DensePose Data Build` be split into smaller, more focused modules?**
  _Cohesion score 0.11088709677419355 - nodes in this community are weakly interconnected._
- **Should `SMPL NumPy Model` be split into smaller, more focused modules?**
  _Cohesion score 0.06485671191553545 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Garment Mask Buffers (4 buffers, Save/Clear, history, prompt store, track)` and `SMPL Torso + Upper-arm UV Atlas (front, back, two sleeves)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._