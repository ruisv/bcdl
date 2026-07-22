cmake_minimum_required(VERSION 3.18)
project(bcdl LANGUAGES CXX VERSION 0.4.0)

# --- Guard: this project only builds on the aarch64 RDK board ----------------
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  message(FATAL_ERROR
    "BCDL builds on the RDK S100 board (aarch64) only. Edit on Mac, then "
    "scripts/sync.sh + scripts/board_build.sh. Current arch: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)

option(BCDL_BUILD_PYTHON "Build the nanobind Python module" ON)
option(BCDL_BUILD_EXAMPLES "Build example executables" ON)
option(BCDL_INSTALL "Generate install + find_package(bcdl) export rules" ON)

find_package(Hobot REQUIRED)
find_package(Threads REQUIRED)  # std::thread in the async pipeline
# OpenMP parallelizes the CPU preproc (letterbox / BGR->NV12 already have
# #pragma omp parallel for). Optional: without it those loops run serially.
find_package(OpenMP)

# OpenCV (libopencv installed via mamba into the board's bcdl conda env). Lets
# the task layer use cv::resize / findContours / minAreaRect /
# rotatedRectangleIntersection instead of hand-written approximations. Optional:
# code guards on BCDL_HAVE_OPENCV. Hint the conda prefix so find_package locates
# the OpenCV 5 CMake config (lib/cmake/opencv5).
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  set(BCDL_CONDA_LIB "$ENV{CONDA_PREFIX}/lib")
endif()
find_package(OpenCV CONFIG QUIET)
if(OpenCV_FOUND)
  message(STATUS "OpenCV ${OpenCV_VERSION} found at ${OpenCV_DIR}")
else()
  message(STATUS "OpenCV not found — task layer uses hand-written image ops")
endif()

# Runtime rpath for every target: the hobot lib dir we linked from PLUS
# /usr/hobot/lib (the device platform libs the SDK transitively needs) — both
# come from HOBOT_RUNTIME_LIB_DIRS. When OpenCV is used, also the conda lib dir
# that holds opencv's .so AND the newer libstdc++ it needs.
set(BCDL_RPATH ${HOBOT_RUNTIME_LIB_DIRS})
if(OpenCV_FOUND)
  list(APPEND BCDL_RPATH "${BCDL_CONDA_LIB}")
  # conda's OpenCV is built with a newer gcc and needs conda's libstdc++
  # (CXXABI_1.3.15, absent from the board's system gcc 11.4). Put conda lib on
  # the link search path globally, and emit DT_RPATH (not RUNPATH) so the rpath
  # also applies to transitive deps — otherwise opencv's libstdc++ falls back to
  # the system one and fails to load.
  link_directories("${BCDL_CONDA_LIB}")
  add_link_options("-Wl,--disable-new-dtags")
endif()

# --- core library ------------------------------------------------------------
# Shared so consumers (the Python module, examples, downstream C++) link one
# libbcdl.so rather than baking a static archive into each — smaller artifacts
# and a single copy of the code in the conda package.
add_library(bcdl SHARED
  src/core/status.cc
  src/core/sys_mem.cc
  src/core/task.cc
  src/core/mem_pool.cc
  src/backend/engine.cc
  src/backend/output_reader.cc
  src/preproc/vp_image.cc
  src/preproc/letterbox.cc
  src/preproc/letterbox_cpu.cc
  src/media/jpeg_codec.cc
  src/media/video_codec.cc
  src/tasks/detection.cc
  src/tasks/classification.cc
  src/tasks/pose.cc
  src/tasks/instance_seg.cc
  src/tasks/promptable_seg.cc
  src/tasks/obb.cc
  src/tasks/mono3d.cc
  src/tasks/ocr.cc
  src/tasks/depth.cc
  src/tasks/segmentation.cc
  src/tasks/embedding.cc
  src/tasks/panoptic_drive.cc
  src/tasks/face.cc
  src/tasks/wholebody.cc
  src/tasks/features.cc
  src/tasks/superres.cc
  src/pipeline/detection_pipeline.cc
  src/pipeline/tracking_pipeline.cc
  src/pipeline/async_detection_pipeline.cc
  src/pipeline/async_video_detection_pipeline.cc
  src/pipeline/stereo_pipeline.cc
  src/tracks/byte_tracker.cc
  src/tracks/reid.cc)
add_library(bcdl::bcdl ALIAS bcdl)

# Version header generated from the project() version (single source of truth).
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/bcdl/version.h" @ONLY)

target_include_directories(bcdl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# hobot::dnn = BPU runtime; hobot::media = hb_vp preproc (resize/warp/cvtcolor).
target_link_libraries(bcdl PUBLIC hobot::dnn Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(bcdl PUBLIC OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found — CPU preproc (letterbox/BGR->NV12) runs "
                  "single-threaded; streaming throughput will be preproc-bound.")
endif()
if(TARGET hobot::media)
  target_link_libraries(bcdl PUBLIC hobot::media)
else()
  message(WARNING "hobot::media (libffmedia/libgdcbin) not found — VP preproc "
                  "ops (letterbox/cvtColor/resize) will fail to link. Detection "
                  "decode/NMS still work. Set HOBOT_LIB_DIR if the libs exist.")
endif()
# --- VPS GDC hardware letterbox (bcdl::GdcLetterbox) --------------------------
# The GDC / hbn-VPF headers live in /usr/hobot/include (NOT /usr/include/hobot),
# and the engine is driven by libvpf/libvio/libcam/libhbmem on /usr/hobot/lib.
# GDC runs on /dev/gdc (VPS), bypassing the root-locked/offline vDSP that
# hb_vp_warp_affine (bcdl::letterbox) uses. Optional: only wired when present.
find_path(HOBOT_GDC_INCLUDE_DIR NAMES gdc_cfg.h PATHS /usr/hobot/include)
# libmultimedia = the media_codec (hb_mm_mc_*) streaming codec API used by
# VideoDecoder for reorder-correct H.264/H.265 decode.
find_library(HB_MULTIMEDIA_LIB NAMES multimedia PATHS /usr/hobot/lib)
if(HB_MULTIMEDIA_LIB)
  target_link_libraries(bcdl PUBLIC ${HB_MULTIMEDIA_LIB})
else()
  message(WARNING "libmultimedia not found — VideoDecoder (media_codec) will not link. "
                  "Set HOBOT_LIB_DIR if it exists.")
endif()

find_library(HB_VPF_LIB  NAMES vpf   PATHS /usr/hobot/lib)
find_library(HB_VIO_LIB  NAMES vio   PATHS /usr/hobot/lib)
find_library(HB_CAM_LIB  NAMES cam   PATHS /usr/hobot/lib)
find_library(HB_MEM_LIB  NAMES hbmem PATHS /usr/hobot/lib)
find_library(HB_GDCBIN_LIB NAMES gdcbin PATHS /usr/hobot/lib)
if(HOBOT_GDC_INCLUDE_DIR AND HB_VPF_LIB AND HB_VIO_LIB AND HB_CAM_LIB AND HB_MEM_LIB)
  set(BCDL_HAVE_GDC ON)
  target_sources(bcdl PRIVATE src/preproc/gdc_bin.cc src/preproc/gdc_letterbox.cc
                          src/preproc/gdc_remap.cc)
  target_include_directories(bcdl PRIVATE ${HOBOT_GDC_INCLUDE_DIR})
  target_link_libraries(bcdl PUBLIC ${HB_VPF_LIB} ${HB_VIO_LIB} ${HB_CAM_LIB} ${HB_MEM_LIB})
  if(HB_GDCBIN_LIB)   # GdcRemap's runtime warp-LUT generation (hbn_gen_gdc_cfg)
    target_link_libraries(bcdl PUBLIC ${HB_GDCBIN_LIB})
  endif()
  target_compile_definitions(bcdl PUBLIC BCDL_HAVE_GDC=1)
  message(STATUS "GDC hardware letterbox: enabled (${HOBOT_GDC_INCLUDE_DIR})")
else()
  message(STATUS "GDC hardware letterbox: disabled (headers/libs not found)")
endif()

# OpenCV: the task layer uses cv::resize / findContours / minAreaRect /
# rotatedRectangleIntersection. PUBLIC so consumers (examples, bcdl_py) inherit
# the include dirs and link; BCDL_HAVE_OPENCV guards the OpenCV code paths.
if(OpenCV_FOUND)
  target_link_libraries(bcdl PUBLIC ${OpenCV_LIBS})
  target_compile_definitions(bcdl PUBLIC BCDL_HAVE_OPENCV=1)
endif()
target_compile_options(bcdl PRIVATE -Wall -Wextra)
# PIC so the static lib can be linked into the Python shared module;
# keep the hobot lib dir + /usr/hobot/lib on the runtime search path.
set_target_properties(bcdl PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH "${HOBOT_RUNTIME_LIB_DIRS}" INSTALL_RPATH "${HOBOT_RUNTIME_LIB_DIRS}")

# --- examples ----------------------------------------------------------------
if(BCDL_BUILD_EXAMPLES)
  add_executable(det_infer examples/det_infer.cc)
  target_link_libraries(det_infer PRIVATE bcdl::bcdl)
  set_target_properties(det_infer PROPERTIES
    BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

  add_executable(det_demo examples/det_demo.cc)
  target_link_libraries(det_demo PRIVATE bcdl::bcdl)
  set_target_properties(det_demo PROPERTIES
    BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

  add_executable(vp_probe examples/vp_probe.cc)
  target_link_libraries(vp_probe PRIVATE bcdl::bcdl)
  set_target_properties(vp_probe PROPERTIES
    BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

  if(BCDL_HAVE_GDC)
    add_executable(gdc_letterbox_bench examples/gdc_letterbox_bench.cc)
    target_link_libraries(gdc_letterbox_bench PRIVATE bcdl::bcdl)
    set_target_properties(gdc_letterbox_bench PROPERTIES
      BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")
  endif()

  add_executable(jpeg_roundtrip examples/jpeg_roundtrip.cc)
  target_link_libraries(jpeg_roundtrip PRIVATE bcdl::bcdl)
  set_target_properties(jpeg_roundtrip PROPERTIES
    BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

  add_executable(video_roundtrip examples/video_roundtrip.cc)
  target_link_libraries(video_roundtrip PRIVATE bcdl::bcdl)
  set_target_properties(video_roundtrip PROPERTIES
    BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

  add_executable(video_decode examples/video_decode.cc)
  target_link_libraries(video_decode PRIVATE bcdl::bcdl)
  set_target_properties(video_decode PROPERTIES
    BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

  # Same-process VPU-decode || BPU-infer coexistence probe (no shared data, no
  # OpenCV, no pipeline) — isolates a vendor-stack fault from our own pipeline
  # structure. A local diagnostic harness that is not part of the distribution,
  # so the target only exists when the source is present.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/coexist_probe.cc")
    add_executable(coexist_probe examples/coexist_probe.cc)
    target_link_libraries(coexist_probe PRIVATE bcdl::bcdl)
    set_target_properties(coexist_probe PROPERTIES
      BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")
  endif()

  add_executable(depth_demo examples/depth_demo.cc)
  target_link_libraries(depth_demo PRIVATE bcdl::bcdl)
  set_target_properties(depth_demo PROPERTIES
    BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

  add_executable(mempool_demo examples/mempool_demo.cc)
  target_link_libraries(mempool_demo PRIVATE bcdl::bcdl)
  set_target_properties(mempool_demo PROPERTIES
    BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

  add_executable(pipeline_bench examples/pipeline_bench.cc)
  target_link_libraries(pipeline_bench PRIVATE bcdl::bcdl)
  set_target_properties(pipeline_bench PROPERTIES
    BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

  add_executable(async_bench examples/async_bench.cc)
  target_link_libraries(async_bench PRIVATE bcdl::bcdl)
  set_target_properties(async_bench PROPERTIES
    BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

  add_executable(async_check examples/async_check.cc)
  target_link_libraries(async_check PRIVATE bcdl::bcdl)
  set_target_properties(async_check PROPERTIES
    BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

  add_executable(tracks_ocr_check examples/tracks_ocr_check.cc)
  target_link_libraries(tracks_ocr_check PRIVATE bcdl::bcdl)
  set_target_properties(tracks_ocr_check PROPERTIES
    BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

  if(OpenCV_FOUND)
    # -L conda lib / DT_RPATH / --disable-new-dtags are applied globally above.
    add_executable(opencv_check examples/opencv_check.cc)
    target_link_libraries(opencv_check PRIVATE ${OpenCV_LIBS})
    set_target_properties(opencv_check PROPERTIES
      BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

    # End-to-end PP-OCRv3 demo: reads/crops/resizes with OpenCV, runs det+rec
    # through bcdl on the board's pre-installed .hbm models.
    add_executable(ocr_demo examples/ocr_demo.cc)
    target_link_libraries(ocr_demo PRIVATE bcdl::bcdl ${OpenCV_LIBS})
    set_target_properties(ocr_demo PROPERTIES
      BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

    # Image-embedding retrieval demo: embeds a gallery folder into an
    # EmbeddingBank and ranks it against a query image.
    add_executable(embed_demo examples/embed_demo.cc)
    target_link_libraries(embed_demo PRIVATE bcdl::bcdl ${OpenCV_LIBS})
    set_target_properties(embed_demo PROPERTIES
      BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

    # Detect-and-track demo: TrackingPipeline (DetectionPipeline + ByteTracker)
    # on a real YOLO NV12 .hbm, fed a panned still image as a moving sequence.
    add_executable(track_demo examples/track_demo.cc)
    target_link_libraries(track_demo PRIVATE bcdl::bcdl ${OpenCV_LIBS})
    set_target_properties(track_demo PROPERTIES
      BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

    # End-to-end: video file -> VPU hardware decode -> DetectionPipeline (BPU).
    # The decode half of the future live-camera path.
    add_executable(video_det_demo examples/video_det_demo.cc)
    target_link_libraries(video_det_demo PRIVATE bcdl::bcdl ${OpenCV_LIBS})
    set_target_properties(video_det_demo PROPERTIES
      BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

    # 3-stage overlapped video path: VPU decode || CPU preproc || BPU infer+NMS.
    add_executable(video_det_demo_async examples/video_det_demo_async.cc)
    target_link_libraries(video_det_demo_async PRIVATE bcdl::bcdl ${OpenCV_LIBS})
    set_target_properties(video_det_demo_async PROPERTIES
      BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")

    # Stereo depth: a side-by-side stereo pair -> StereoPipeline (LAS2 .hbm) ->
    # disparity / depth, written as a side-by-side visualization.
    add_executable(stereo_demo examples/stereo_demo.cc)
    target_link_libraries(stereo_demo PRIVATE bcdl::bcdl ${OpenCV_LIBS})
    set_target_properties(stereo_demo PROPERTIES
      BUILD_RPATH "${BCDL_RPATH}" INSTALL_RPATH "${BCDL_RPATH}")
  endif()
endif()

# --- python module -----------------------------------------------------------
if(BCDL_BUILD_PYTHON)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE _nb_rc)
  if(NOT _nb_rc EQUAL 0)
    message(FATAL_ERROR "nanobind not found in this env: pip/conda install nanobind "
                        "(or configure with -DBCDL_BUILD_PYTHON=OFF)")
  endif()
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(bcdl_py src/python/bcdl_py.cc)
  target_link_libraries(bcdl_py PRIVATE bcdl::bcdl)
  # bcdl::bcdl is imported from the build tree; its INTERFACE compile defs don't
  # always reach the module compile, so set the GDC guard explicitly.
  if(BCDL_HAVE_GDC)
    target_compile_definitions(bcdl_py PRIVATE BCDL_HAVE_GDC=1)
  endif()
  # bcdl_py.so dynamically links libbcdl.so. The install/export rules put the
  # shared lib one level down (lib/), while bcdl_py.so sits at the package root
  # (wheel) — so add $ORIGIN/lib alongside the hobot runtime dirs.
  set_target_properties(bcdl_py PROPERTIES
    BUILD_RPATH "${BCDL_RPATH}"
    INSTALL_RPATH "$ORIGIN/lib;${BCDL_RPATH}")

  # When built as a pip wheel (scikit-build-core sets SKBUILD), install the
  # compiled module at the wheel root so `import bcdl_py` resolves; the pure
  # `bcdl` package is added via [tool.scikit-build] wheel.packages in pyproject.
  if(DEFINED SKBUILD)
    install(TARGETS bcdl_py LIBRARY DESTINATION ".")
    # Ship the committed type stub at the wheel root, next to bcdl_py.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/bcdl_py.pyi")
      install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/python/bcdl_py.pyi" DESTINATION ".")
    endif()
  else()
    # Plain cmake --install (e.g. the conda libbcdl/bcdl split): the compiled
    # module, its stub, and the pure-Python wrapper package go to this
    # interpreter's site-packages under CMAKE_INSTALL_PREFIX, while libbcdl.so /
    # headers / cmake config land in lib/ + include/ via the BCDL_INSTALL rules.
    set(BCDL_SITE_DIR
      "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
    install(TARGETS bcdl_py LIBRARY DESTINATION "${BCDL_SITE_DIR}")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/bcdl_py.pyi")
      install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/python/bcdl_py.pyi" DESTINATION "${BCDL_SITE_DIR}")
    endif()
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/bcdl" DESTINATION "${BCDL_SITE_DIR}")
  endif()

  # Optional: (re)generate the bcdl_py.pyi type stub. Needs to import the module,
  # so the hobot runtime libs must be present — board-only. Off by default; the
  # generated stub is committed (python/bcdl_py.pyi) and shipped in the wheel.
  option(BCDL_BUILD_STUBS "Regenerate the bcdl_py.pyi type stub (imports the module)" OFF)
  if(BCDL_BUILD_STUBS)
    nanobind_add_stub(bcdl_py_stub
      MODULE bcdl_py
      OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/python/bcdl_py.pyi"
      PYTHON_PATH $<TARGET_FILE_DIR:bcdl_py>
      DEPENDS bcdl_py)
  endif()
endif()

# --- install / export: find_package(bcdl) for downstream consumers -----------
if(BCDL_INSTALL)
  include(CMakePackageConfigHelpers)

  install(TARGETS bcdl EXPORT bcdlTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

  # Public headers (hand-written tree + generated version.h).
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/bcdl"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/bcdl/version.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/bcdl")

  # The exported targets reference hobot::dnn/media, so ship FindHobot.cmake and
  # re-find it from the package config.
  set(BCDL_PACKAGE_OPTIONAL_DEPS "")
  if(OpenMP_CXX_FOUND)
    string(APPEND BCDL_PACKAGE_OPTIONAL_DEPS "find_dependency(OpenMP)\n")
  endif()
  if(OpenCV_FOUND)
    string(APPEND BCDL_PACKAGE_OPTIONAL_DEPS "find_dependency(OpenCV)\n")
  endif()

  set(BCDL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/bcdl")
  install(EXPORT bcdlTargets
    NAMESPACE bcdl::
    FILE bcdlTargets.cmake
    DESTINATION "${BCDL_CMAKE_DIR}")
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindHobot.cmake"
    DESTINATION "${BCDL_CMAKE_DIR}")

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/bcdlConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/bcdlConfig.cmake"
    INSTALL_DESTINATION "${BCDL_CMAKE_DIR}")
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/bcdlConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}" COMPATIBILITY SameMajorVersion)
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/bcdlConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/bcdlConfigVersion.cmake"
    DESTINATION "${BCDL_CMAKE_DIR}")
endif()
