sudo docker run -d --runtime=nvidia --gpus all -it --net=host --ipc=host \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY=unix$DISPLAY \
-e GDK_SCALE \
-e GDK_DPI_SCALE \
-v /path/to/workspace:/workspace \
-v /path/to/datasets:/data \
-v /path/to/output:/output \
ovsam3d:v1
