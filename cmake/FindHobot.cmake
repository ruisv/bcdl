# Locates the D-Robotics hobot SDK for an RDK S100/S100P board build.
#
# Two possible sources, searched in this order:
#   1. The active conda env ($CONDA_PREFIX) — when the hobot-dnn / hobot-media
#      conda packages are installed (headers in include/hobot, libs in lib/).
#   2. The board system image — headers in /usr/include/hobot, libs in
#      /usr/hobot/lib.
#
# Whichever source provides the *linked* libs, /usr/hobot/lib is ALWAYS kept on
# the runtime rpath: the SDK libs transitively need device platform libs
# (libbpu / libhbmem / libalog / libvdsp, plus ffmpeg / libcjson for the codec
# line) that only the board image ships — never the conda packages.
#
# Provides imported targets:
#   hobot::dnn   (BPU runtime: hbDNN*)        -> libdnn + libhbucp
#   hobot::media (hb_vp preproc + codecs)     -> libhbvp (+ libffmedia/libgdcbin
#                                                for the JPEG/Video codec phases)
# and variables HOBOT_INCLUDE_DIR / HOBOT_LIB_DIR / HOBOT_RUNTIME_LIB_DIRS.

set(HOBOT_INCLUDE_ROOT "/usr/include" CACHE PATH "board hobot headers root (contains hobot/)")
set(HOBOT_SYSTEM_LIB_DIR "/usr/hobot/lib" CACHE PATH "board hobot shared libraries directory")

# Prefixes searched (as HINTS, i.e. before the /usr fallback PATHS) for the
# hobot conda packages:
#  - $PREFIX / $BUILD_PREFIX : conda-build / rattler-build host environment
#  - $CONDA_PREFIX           : an activated conda env (board-side dev build)
# Plain HINTS (not NO_DEFAULT_PATH) so CMAKE_PREFIX_PATH — which rattler-build
# points at the host $PREFIX — is still honoured. /usr is the board fallback.
set(_hobot_inc_hints "")
set(_hobot_lib_hints "")
foreach(_env PREFIX BUILD_PREFIX CONDA_PREFIX)
  if(DEFINED ENV{${_env}})
    list(APPEND _hobot_inc_hints "$ENV{${_env}}/include")
    list(APPEND _hobot_lib_hints "$ENV{${_env}}/lib")
  endif()
endforeach()

find_path(HOBOT_INCLUDE_DIR
  NAMES hobot/dnn/hb_dnn.h
  HINTS ${_hobot_inc_hints}
  PATHS ${HOBOT_INCLUDE_ROOT})

find_library(HOBOT_DNN_LIB   NAMES dnn     HINTS ${_hobot_lib_hints} PATHS ${HOBOT_SYSTEM_LIB_DIR})
find_library(HOBOT_UCP_LIB   NAMES hbucp   HINTS ${_hobot_lib_hints} PATHS ${HOBOT_SYSTEM_LIB_DIR})
# hbVP* (resize / warp-affine / cvt-color / codecs) live in libhbvp.
find_library(HOBOT_VP_LIB    NAMES hbvp    HINTS ${_hobot_lib_hints} PATHS ${HOBOT_SYSTEM_LIB_DIR})
find_library(HOBOT_MEDIA_LIB NAMES ffmedia HINTS ${_hobot_lib_hints} PATHS ${HOBOT_SYSTEM_LIB_DIR})
find_library(HOBOT_GDC_LIB   NAMES gdcbin  HINTS ${_hobot_lib_hints} PATHS ${HOBOT_SYSTEM_LIB_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Hobot
  REQUIRED_VARS HOBOT_INCLUDE_DIR HOBOT_DNN_LIB HOBOT_UCP_LIB)

# Directory that actually holds libdnn — the conda lib dir or the board image.
get_filename_component(HOBOT_LIB_DIR "${HOBOT_DNN_LIB}" DIRECTORY)

# Runtime rpath set: the dir we linked from, plus the board image dir for the
# device platform libs the SDK transitively needs. Deduplicated so a pure
# /usr/hobot/lib build keeps a single entry.
set(HOBOT_RUNTIME_LIB_DIRS "${HOBOT_LIB_DIR}" "${HOBOT_SYSTEM_LIB_DIR}")
list(REMOVE_DUPLICATES HOBOT_RUNTIME_LIB_DIRS)

if(Hobot_FOUND AND NOT TARGET hobot::dnn)
  add_library(hobot::dnn UNKNOWN IMPORTED)
  set_target_properties(hobot::dnn PROPERTIES
    IMPORTED_LOCATION "${HOBOT_DNN_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${HOBOT_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${HOBOT_UCP_LIB}")
endif()

if(Hobot_FOUND AND HOBOT_VP_LIB AND NOT TARGET hobot::media)
  # libhbvp provides the hbVP* preproc + codec entry points; the ffmedia/gdcbin
  # libs are pulled in (when present) for the JPEG/Video codec phases.
  set(_media_deps "")
  if(HOBOT_MEDIA_LIB)
    list(APPEND _media_deps "${HOBOT_MEDIA_LIB}")
  endif()
  if(HOBOT_GDC_LIB)
    list(APPEND _media_deps "${HOBOT_GDC_LIB}")
  endif()
  add_library(hobot::media UNKNOWN IMPORTED)
  set_target_properties(hobot::media PROPERTIES
    IMPORTED_LOCATION "${HOBOT_VP_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${HOBOT_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${_media_deps}")
endif()
