<p align="center" >
    <img src="docs/logo.png"  width="60%" >
</p>

<h2 align="center">AnyRecon: Arbitrary-View 3D Reconstruction<br>with Video Diffusion Model</h2>

<br>
<p align="center">
    <b>Your star means a lot for us to develop this project! ✨</b>
</p>
<p align="center" width="100%">
    <img src="docs/gif.gif"  width="90%" >
</p>


## TODO List

- [ ] Upload sparse attention weight.

## 🛠️ Environment Setup

###  1. Clone Repository and Setup Environment

The point-cloud rendering pipeline depends on [π³](https://github.com/yyfz/Pi3/), which is included as a git submodule. Make sure to clone **recursively** so that `Pi3/` is fetched at the same time:

```bash
git clone --recursive https://github.com/OpenImagingLab/AnyRecon.git
# If you already cloned without --recursive, run:
#   git submodule update --init --recursive
cd AnyRecon
conda create -n anyrecon python=3.10 -y
conda activate anyrecon
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -r Pi3/requirements.txt
```

###  2. Download Models
AnyRecon relies on specific pre-trained weights. By default, Wan2.1 weights are read from `./checkpoints`; use `--wan_model_dir` to load them from another directory.

- Base Video Diffusion Model (Wan2.1 I2V 14B 720P) [[download](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main)]
- AnyRecon LoRA weights [[download](https://huggingface.co/Yutian10/AnyRecon/tree/main)]
- π³ checkpoint (for point-cloud rendering) [[download](https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors)] → place at `Pi3/model.safetensors`

## 🚀 Quick Start
For inference, processing an 869x512 video at 40 frames requires approximately 45GB of VRAM, but you can lower the resolution if your VRAM is insufficient. To reproduce the provided example, run:

```bash
bash test.sh
```

Or directly:

```bash
BASE_DIR="/home/dataset/data/z84450662/hf_models"

python run_AnyRecon.py \
    --root_dir example/valley \
    --output_dir example/valley \
    --wan_model_dir "${BASE_DIR}/Wan-AI/Wan2.1-I2V-14B-720P" \
    --lora_path "${BASE_DIR}/Yutian10/AnyRecon/AnyRecon_full_attention.ckpt"
```

## 🌟 Run on Your Own Data

`run_AnyRecon.py` expects point-cloud rendered **condition videos** as input. To prepare them from a raw video, we provide a helper script built on top of [π³](https://github.com/yyfz/Pi3/):

```bash
bash run_pi3.sh
```


**Input video format.** Your input video must be organized so that:

- the **first `--num_cond_frames` frames** are the **capture views** — these provide the 3D point cloud,
- the **remaining frames** are the **test views** — they are *only* used to estimate the camera poses at which the point cloud is rendered, and **do not contribute any points** to the reconstruction.


**Custom test-view trajectory.** You can replace the test-view portion of the input video with placeholder frames and provide a NumPy trajectory directly:

```bash
BASE_DIR="/home/dataset/data/z84450662/hf_models"

python run_pi3.py \
    --base_scene_dir example/my_scene.mp4 \
    --num_cond_frames 6 \
    --output_dir example/my_scene_custom \
    --ckpt "${BASE_DIR}/yyfz233/Pi3/model.safetensors" \
    --trajectory_path forward_trajectory.npy \
    --trajectory_convention c2w
```

The trajectory must be a `.npy` array with shape `(num_render_frames, 4, 4)`, where `num_render_frames` is the number of input frames after the first `num_cond_frames` capture views. Both camera-to-world (`c2w`) and world-to-camera (`w2c`) matrices are supported. By default, the first custom pose is rigidly aligned to the last capture-view pose estimated by π³, so a relative trajectory may start at the identity matrix. Pass `--no-align_trajectory_first_pose` only when the trajectory is already expressed in π³'s reconstructed world frame.

The capture views still provide all points used for reconstruction. The placeholder frames determine only the required number of rendered target views; their π³-estimated poses are replaced by the supplied trajectory.

Once `run_pi3.py` has produced the condition videos in `--output_dir`, point `run_AnyRecon.py --root_dir` to that directory and run inference as shown above.

## 💗 Acknowledgments
Thanks to these great repositories: [Wan2.1](https://github.com/Wan-Video/Wan2.1), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), and [π³](https://github.com/yyfz/Pi3/).

## 🔗 Citation
If you find our work helpful, please cite it:
```
@article{chen2026anyrecon,
  title={AnyRecon: Arbitrary-View 3D Reconstruction with Video Diffusion Model},
  author={Chen, Yutian and Guo, Shi and Jin, Renbiao and Yang, Tianshuo and Cai, Xin and Luo, Yawen and Yang, Mingxin and Yu, Mulin and Xu, Linning and Xue, Tianfan},
  journal={arXiv preprint arXiv:2604.19747},
  year={2026}
}
```
