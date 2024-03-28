################################################
# MEDIA LIBRARY DEPS
################################################
media_library_common_dep = dependency('hailo_media_library_common', method : 'pkg-config')
media_library_frontend_dep = dependency('hailo_media_library_frontend', method : 'pkg-config')
media_library_encoder_dep = dependency('hailo_media_library_encoder', method : 'pkg-config')
media_library_api_dep = dependency('hailo_media_library_api', method : 'pkg-config')
encoder_dep = dependency('hailo_encoder', method : 'pkg-config')
gstmedialibrary_utils_dep = dependency('gstmedialibutils', method : 'pkg-config')

dependencies_apps = plugin_deps + media_library_encoder_dep + media_library_common_dep + media_library_frontend_dep + encoder_dep + media_library_api_dep + gstmedialibrary_utils_dep

################################################
# LPR OCR SINK SOURCE
################################################
lpr_ocrsink_source = '../x86/lpr/lpr_ocrsink.cpp'

lpr_ocrsink_lib = shared_library('lpr_ocrsink',
  lpr_ocrsink_source,
  cpp_args : hailo_lib_args,
  include_directories: hailo_general_inc,
  dependencies : plugin_deps + [opencv_dep, meta_dep, cv_singleton_dep, image_dep, tracker_dep],
  gnu_symbol_visibility : 'default',
  install: install_lpr,
  install_dir: apps_install_dir + '/license_plate_recognition/resources',
)

subdir('encoder_pipelines')
subdir('encoder_pipelines_new_api')
subdir('vision_preproc_applications')
