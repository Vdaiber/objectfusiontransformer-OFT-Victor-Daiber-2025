#!/bin/bash

# Perfect Docker Run Command for GPU Server
# Based on current container configuration

echo "Starting GPU development container..."

docker run \
  --name oft_dev_gpu \
  -d \
  --gpus device=1 \
  -v /home/ge36vox/training:/app:rw \
  -v /data/daiber_fent:/data/daiber_fent:rw \
  -v /data/v1.0-test:/data/v1.0-test:ro \
  -v /data/v1.0-mini:/data/v1.0-mini:ro \
  -v /data/v1.0-trainval:/data/v1.0-trainval:ro \
  -v /data/samples:/data/samples:ro \
  -v /data/sweeps:/data/sweeps:ro \
  -w /app \
  oft_gpu \
  tail -f /dev/null

echo "Container started. Connect with: docker exec -it oft_dev_gpu /bin/bash"
