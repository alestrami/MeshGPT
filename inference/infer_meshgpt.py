import random
import sys
import os
from model.pointnet import get_pointnet_classifier
import omegaconf
import torch
from pathlib import Path
import trimesh

from dataset.quantized_soup import QuantizedSoupTripletsCreator
from dataset.triangles import TriangleNodesWithFacesAndSequenceIndices
from trainer import get_rvqvae_v0_decoder
from trainer.train_transformer import get_qsoup_model_config
from util.meshlab import meshlab_proc
from util.misc import get_parameters_from_state_dict
from util.visualization import plot_vertices_and_faces
from tqdm import tqdm
from model.transformer import QuantSoupTransformer
from pytorch_lightning import seed_everything

import wandb


@torch.no_grad()
def main(config, mode):
    # Initialize W&B
    wandb_flag = False
    if wandb_flag: wandb.init(project="meshAle", mode="offline") 
    # Enable system metrics logging
    if wandb_flag: wandb.config.update({"log_system_metrics": True})

    seed_everything(42)
    device = torch.device('cuda:0')
    vq_cfg = omegaconf.OmegaConf.load(Path(config.vq_resume).parents[1] / "config.yaml")
    # prendo il dataset
    dataset = TriangleNodesWithFacesAndSequenceIndices(config, 'train', True, True , config.ft_category)
    prompt_num_faces  = 4
    output_dir_image = Path(f'runs/{config.experiment}/inf_image_{mode}')
    output_dir_image.mkdir(exist_ok=True, parents=True)
    output_dir_mesh = Path(f'runs/{config.experiment}/inf_mesh_{mode}')
    output_dir_mesh.mkdir(exist_ok=True, parents=True)
    model_cfg = get_qsoup_model_config(config, vq_cfg.embed_levels)
    model = QuantSoupTransformer(model_cfg, vq_cfg)
    state_dict = torch.load(config.resume, map_location="cpu")["state_dict"]
    sequencer = QuantizedSoupTripletsCreator(config, vq_cfg)
    model.load_state_dict(get_parameters_from_state_dict(state_dict, "model"))
    model = model.to(device)
    model = model.eval()
    sequencer = sequencer.to(device)
    sequencer = sequencer.eval()
    decoder = get_rvqvae_v0_decoder(vq_cfg, config.vq_resume, device)
    pnet = get_pointnet_classifier().to(device)

    k = 0
    while k < config.num_val_samples:

        data = dataset.get(random.randint(0, len(dataset) - 1))
        print(data.x.shape)
        plot_vertices_and_faces(data.x[:,:3].numpy(), data.faces.numpy(),  output_dir_image / f"{k:06d}_start_a.jpg") #Ale added
        soup_sequence, face_in_idx, face_out_idx, target = sequencer.get_completion_sequence(
            data.x.to(device),
            data.edge_index.to(device),
            data.faces.to(device),
            data.num_vertices,
            1 + 6 * prompt_num_faces
        )

        y = None
        print(f"[inference/infer_meshgpt.py/#main] mode {mode}")
        if mode == "topp":
            # more diversity but more change of bad sequences
            y = model.generate(
                soup_sequence, face_in_idx, face_out_idx, sequencer, config.max_val_tokens,
                temperature=config.temperature, top_k=config.top_k_tokens, top_p=config.top_p, use_kv_cache=True
            )
        elif mode == "beam":
            # less diversity but more robust
            y = model.generate_with_beamsearch(
                soup_sequence, face_in_idx, face_out_idx, sequencer, config.max_val_tokens, use_kv_cache=True, beam_width=6
            )

        if y is None:
            continue

        gen_vertices, gen_faces = sequencer.decode(y[0], decoder)
        print(f"[inference/infer_meshgpt.py#main] start try mesh gen_vertices {len(gen_vertices)}, gen_faces {len(gen_faces)}")
        try:
            # dopo che ho generato i vertici e decodificati provo a farci la mesh
            mesh = trimesh.Trimesh(vertices=gen_vertices, faces=gen_faces, process=False)
            if pnet.classifier_guided_filter(mesh, config.ft_category):
                mesh.export(output_dir_mesh / f"{k:06d}.obj")
                meshlab_proc(output_dir_mesh / f"{k:06d}.obj")
                plot_vertices_and_faces(gen_vertices, gen_faces, output_dir_image / f"{k:06d}.jpg")
                print('Generated mesh', k + 1)
                print(f"[inference/infer_meshgpt.py/#main] Length of faces of generated mesh {len(mesh.faces)}")
                # Log to W&B with mode
                if wandb_flag: wandb.log({"Generated Mesh Face Length": len(mesh.faces)})
                k = k + 1
        except Exception as e:
            print('Exception occured: ', e)
            pass  # sometimes the mesh is invalid (ngon) and we don't want to crash


if __name__ == "__main__":
    cfg = omegaconf.OmegaConf.load(Path(sys.argv[1]).parents[1] / "config.yaml")
    cfg.resume = sys.argv[1]
    cfg.padding = 0.0
    cfg.num_val_samples = int(sys.argv[3])
    cfg.sequence_stride = cfg.block_size
    cfg.top_p = 0.95
    cfg.temperature = 1.0
    cfg.low_augment = True
    main(cfg, sys.argv[2])
