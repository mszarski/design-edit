"""Copy each 'Training in progress, step N' commit's adapter (repo root) into a checkpoint-N folder."""
import re, sys, tempfile
from huggingface_hub import HfApi, hf_hub_download
api, repo = HfApi(), sys.argv[1]
have = {f.split("/")[0] for f in api.list_repo_files(repo)}
for c in api.list_repo_commits(repo):
    m = re.match(r"Training in progress, step (\d+)", c.title)
    if not m or f"checkpoint-{m.group(1)}" in have:
        continue
    with tempfile.TemporaryDirectory() as d:
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            hf_hub_download(repo, f, revision=c.commit_id, local_dir=d)
        api.upload_folder(repo_id=repo, folder_path=d, path_in_repo=f"checkpoint-{m.group(1)}",
                          allow_patterns=["adapter_*"], commit_message=f"checkpoint-{m.group(1)} from {c.commit_id[:10]}")
    print("materialized checkpoint-" + m.group(1))
