# LOKI

# https://github.com/rbbrdckybk/ai-art-generator/

# requirments start

# Make Director to store Loki at and navagate there
mkdir Loki && cd Loki

# Download Tesla GPU Driver
curl -OL https://us.download.nvidia.com/tesla/515.65.01/NVIDIA-Linux-x86_64-515.65.01.run
chmod +x NVIDIA-Linux-x86_64-515.65.01.run
./NVIDIA-Linux-x86_64-515.65.01.run

# Download and install anaconda
curl -OL https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
sh Anaconda3-2022.05-Linux-x86_64.sh


# requirments end

# Set up our conda env
conda create --name ai-art python=3.9
conda activate ai-art

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Install other required Python packages:
conda install -c anaconda git urllib3
pip install transformers keyboard pillow ftfy regex tqdm omegaconf pytorch-lightning IPython kornia imageio imageio-ffmpeg einops torch_optimizer

# Clone this repository and switch to its directory
git clone https://github.com/rbbrdckybk/ai-art-generator
cd ai-art-generator

# Clone additional required repositories
git clone https://github.com/openai/CLIP
git clone https://github.com/CompVis/taming-transformers

git clone https://github.com/crowsonkb/guided-diffusion
git clone https://github.com/assafshocher/ResizeRight
git clone https://github.com/CompVis/latent-diffusion

git clone https://github.com/rbbrdckybk/stable-diffusion

# Make Required Folder Structer
mkdir checkpoints
mkdir content
mkdir content/models
mkdir content/models/superres
mkdir stable-diffusion/models/ldm/stable-diffusion-v1

# Download the default VQGAN pre-trained model checkpoint files
curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1"
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1"

# FFHQ model (trained on faces) support
curl -L -o checkpoints/ffhq.yaml -C - "https://app.koofr.net/content/links/0fc005bf-3dca-4079-9d40-cdf38d42cd7a/files/get/2021-04-23T18-19-01-project.yaml?path=%2F2021-04-23T18-19-01_ffhq_transformer%2Fconfigs%2F2021-04-23T18-19-01-project.yaml&force"
curl -L -o checkpoints/ffhq.ckpt -C - "https://app.koofr.net/content/links/0fc005bf-3dca-4079-9d40-cdf38d42cd7a/files/get/last.ckpt?path=%2F2021-04-23T18-19-01_ffhq_transformer%2Fcheckpoints%2Flast.ckpt"

# Install packages for CLIP-guided diffusion (if you're only interested in VQGAN+CLIP, you can skip everything from here to the end):
pip install ipywidgets omegaconf torch-fidelity einops wandb opencv-python matplotlib lpips datetime timm
conda install pandas

# Download models needed for CLIP-guided diffusion
curl -L -o content/models/256x256_diffusion_uncond.pt -C - "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
curl -L -o content/models/512x512_diffusion_uncond_finetune_008100.pt -C - "http://batbot.tv/ai/models/guided-diffusion/512x512_diffusion_uncond_finetune_008100.pt"
curl -L -o content/models/secondary_model_imagenet_2.pth -C - "https://ipfs.pollinations.ai/ipfs/bafybeibaawhhk7fhyhvmm7x24zwwkeuocuizbqbcg5nqx64jq42j75rdiy/secondary_model_imagenet_2.pth"
curl -L -o content/models/superres/project.yaml -C - "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1"
curl -L -o content/models/superres/last.ckpt -C - "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1"

# Install additional dependancies required by Stable Diffusion
pip install diffusers

# Download the Stable Diffusion pre-trained checkpoint file
# Might have to access it manually see link below, sign in required:
# https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
curl -L -o stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt -C - "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt"


# Now Run The Code
# python make_art.py example-prompts.txt
