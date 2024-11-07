from huggingface_hub import snapshot_download

dataset_name = "ycbv"
local_dir = "/home/mendax/project/SplitPose/datasets"

snapshot_download(repo_id="bop-benchmark/datasets", 
                  allow_patterns=f"{dataset_name}/*zip",
                  repo_type="dataset", 
                  local_dir=local_dir)