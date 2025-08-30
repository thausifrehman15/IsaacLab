
#!/usr/bin/env bash
set -euo pipefail

apptainer exec --nv \
  --writable \
  --bind ~/isaac_sim_cache:/isaac-sim/kit/cache \
  --bind ~/isaac_sim_logs:/isaac-sim/kit/logs \
  --bind ~/isaaclab_outputs:/opt/isaaclab/outputs \
  --bind ~/isaaclab_container_logs:/opt/isaaclab/logs \
  --bind /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
  --bind /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:ro \
  --env ACCEPT_EULA=Y \
  --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  isaaclab_sandbox \
  /bin/bash -lc 'cd /opt/isaaclab && \
  export PYTHONPATH=/opt/isaaclab/source:/opt/isaaclab/source/isaaclab:$PYTHONPATH && \
  exec /opt/isaaclab/_isaac_sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Jump-Table-Unitree-Go2-v0 \
  --video \
  --video_length=400 \
  --video_interval=500 \
  --max_iterations=5'


# apptainer exec --nv \
#   --bind ~/isaac_sim_cache:/isaac-sim/kit/cache \
#   --bind ~/isaac_sim_logs:/isaac-sim/kit/logs \
#   --bind ~/isaaclab_outputs:/opt/isaaclab/outputs \
#   --bind ~/isaaclab_container_logs:/opt/isaaclab/logs \
#   --bind /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
#   --bind /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:ro \
#   --env ACCEPT_EULA=Y \
#   --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
#   isaaclab_sandbox \
#   /bin/bash -lc 'cd /opt/isaaclab && \
#   export PYTHONPATH=/opt/isaaclab/source:/opt/isaaclab/source/isaaclab:$PYTHONPATH && \
#   exec /opt/isaaclab/_isaac_sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
#   --task=Isaac-Jump-Table-Unitree-Go2-v0 \
#   --checkpoint "/opt/isaaclab/logs/rsl_rl/unitree_go2_flat/2025-08-29_23-59-17/model_999.pt" \
#   --max_iterations=2000'
