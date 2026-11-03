#pragma once

/// BCDL — BPU Common Deep Learning. Umbrella header.
///
/// Pulls in EVERY installed public header. Keep it exhaustive: docs/CPP_API.md
/// tells users this one include is enough, so anything reachable from the
/// install tree but missing here is a compile error for someone following the
/// documentation.

// Core: memory, tasks, errors
#include "bcdl/core/mem_pool.h"
#include "bcdl/core/status.h"
#include "bcdl/core/sys_mem.h"
#include "bcdl/core/task.h"

// Backend: model execution and tensor I/O
#include "bcdl/backend/engine.h"
#include "bcdl/backend/input_writer.h"
#include "bcdl/backend/output_reader.h"

// Preprocessing: images, geometry, letterbox, GDC, voxelization, speech features
#include "bcdl/preproc/drive_inputs.h"
#include "bcdl/preproc/fbank.h"
#include "bcdl/preproc/istft.h"
#include "bcdl/preproc/gdc_letterbox.h"
#include "bcdl/preproc/gdc_remap.h"
#include "bcdl/preproc/geometry.h"
#include "bcdl/preproc/letterbox.h"
#include "bcdl/preproc/letterbox_cpu.h"
#include "bcdl/preproc/lidar_voxelize.h"
#include "bcdl/preproc/vp_image.h"

// Media: hardware codecs
#include "bcdl/media/jpeg_codec.h"
#include "bcdl/media/video_codec.h"

// Tasks: model-specific pre/post-processing
#include "bcdl/tasks/anomaly.h"
#include "bcdl/tasks/asr.h"
#include "bcdl/tasks/command.h"
#include "bcdl/tasks/vocoder.h"
#include "bcdl/tasks/tts.h"
#include "bcdl/tasks/vad.h"
#include "bcdl/tasks/kws.h"
#include "bcdl/tasks/classification.h"
#include "bcdl/tasks/depth.h"
#include "bcdl/tasks/depth_refine.h"
#include "bcdl/tasks/detection.h"
#include "bcdl/tasks/e2e_drive.h"
#include "bcdl/tasks/embedding.h"
#include "bcdl/tasks/face.h"
#include "bcdl/tasks/features.h"
#include "bcdl/tasks/instance_seg.h"
#include "bcdl/tasks/lidar_det3d.h"
#include "bcdl/tasks/mono3d.h"
#include "bcdl/tasks/obb.h"
#include "bcdl/tasks/ocr.h"
#include "bcdl/tasks/open_vocab.h"
#include "bcdl/tasks/optical_flow.h"
#include "bcdl/tasks/panoptic_drive.h"
#include "bcdl/tasks/pose.h"
#include "bcdl/tasks/promptable_seg.h"
#include "bcdl/tasks/segmentation.h"
#include "bcdl/tasks/superres.h"
#include "bcdl/tasks/wholebody.h"

// Tracking: association and re-identification
#include "bcdl/tracks/byte_tracker.h"
#include "bcdl/tracks/reid.h"

// Pipelines: end-to-end compositions
#include "bcdl/pipeline/async_detection_pipeline.h"
#include "bcdl/pipeline/async_video_detection_pipeline.h"
#include "bcdl/pipeline/detection_pipeline.h"
#include "bcdl/pipeline/stereo_pipeline.h"
#include "bcdl/pipeline/tracking_pipeline.h"
