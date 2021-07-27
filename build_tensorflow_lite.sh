cd ~/tensorflow

rm -r ./tensorflow/lite/tools/make/gen/linux_x86_64/bin/minimal
mkdir ./tensorflow/lite/tools/make/gen/linux_x86_64/bin/minimal

sudo bazel build -s -c opt --copt="-DMESA_EGL_NO_X11_HEADERS" --define DEBUG=1 \
tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so

rm -r ./tensorflow/lite/tools/make/gen/linux_x86_64/bin/minimal
mkdir ./tensorflow/lite/tools/make/gen/linux_x86_64/bin/minimal

./tensorflow/lite/tools/make/build_lib.sh $@
