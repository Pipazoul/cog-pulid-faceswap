# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import torch
import time
import mimetypes
import subprocess
import numpy as np
from PIL import Image
from typing import Any, List, Optional, Tuple
import insightface
import onnxruntime
from insightface.app import FaceAnalysis

import gfpgan
import cv2
import tempfile
import time

mimetypes.add_type("image/webp", ".webp")

MODEL_CACHE = "models"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

torch.set_grad_enabled(False)


# other params
DEFAULT_NEGATIVE_PROMPT = (
    'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
    'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
    'low resolution, partially rendered objects,  deformed or partially rendered eyes, '
    'deformed, deformed eyeballs, cross-eyed,blurry'
)


# Define the function to check if an image is black
def is_black_image(image_np):
    return np.mean(image_np) < 10


def run(
    id_image: Optional[np.ndarray],
    supp_image1: Optional[np.ndarray],
    supp_image2: Optional[np.ndarray],
    supp_image3: Optional[np.ndarray],
    prompt: str,
    neg_prompt: str,
    scale: float,
    n_samples: int,
    seed: int,
    steps: int,
    H: int,
    W: int,
    id_scale: float,
    mode: str,
    id_mix: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    supp_images = (supp_image1, supp_image2, supp_image3)

    pipeline.debug_img_list = []
    if mode == 'fidelity':
        attention.NUM_ZERO = 8
        attention.ORTHO = False
        attention.ORTHO_v2 = True
    elif mode == 'extremely style':
        attention.NUM_ZERO = 16
        attention.ORTHO = True
        attention.ORTHO_v2 = False
    else:
        raise ValueError

    if id_image is not None:
        id_image = resize_numpy_image_long(id_image, 1024)
        id_embeddings = pipeline.get_id_embedding(id_image)
        for supp_id_image in supp_images:
            if supp_id_image is not None:
                supp_id_image = resize_numpy_image_long(supp_id_image, 1024)
                supp_id_embeddings = pipeline.get_id_embedding(supp_id_image)
                id_embeddings = torch.cat(
                    (id_embeddings, supp_id_embeddings if id_mix else supp_id_embeddings[:, :5]), dim=1
                )
    else:
        id_embeddings = None

    seed_everything(seed)
    ims = []
    for _ in range(n_samples):
        img = pipeline.inference(prompt, (1, H, W), neg_prompt, id_embeddings, id_scale, scale, steps)[0]
        ims.append(np.array(img))

    return ims, pipeline.debug_img_list


def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        global attention, resize_numpy_image_long, seed_everything, pipeline

        model_files = [
            "antelopev2.tar",
            "models--ByteDance--SDXL-Lightning.tar",
            "models--DIAMONIK7777--antelopev2.tar",
            "models--QuanSun--EVA-CLIP.tar",
            "models--guozinan--PuLID.tar",
            "models--stabilityai--stable-diffusion-xl-base-1.0.tar",
            "pulid_v1.bin",
            "sdxl_lightning_4step_unet.safetensors",
            "version.txt",
            "version_diffusers_cache.txt",
        ]

        base_url = f"https://weights.replicate.delivery/default/PuLID/{MODEL_CACHE}/"

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        for model_file in model_files:
            url = base_url + model_file

            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # We need to download the weights before we can import these (bad ik)
        from pulid.pipeline import PuLIDPipeline
        from pulid import attention_processor as attention
        from pulid.utils import resize_numpy_image_long, seed_everything

        pipeline = PuLIDPipeline()

        # Check and convert the VAE's datatype to float16 if it's float32
        if pipeline.pipe.vae.dtype == torch.float32:
            pipeline.pipe.vae.to(dtype=torch.float16)

        self.black_image_count = 0
        self.threshold = 2

        self.face_swapper = insightface.model_zoo.get_model('cache/inswapper_128.onnx', providers=onnxruntime.get_available_providers())
        self.face_enhancer = gfpgan.GFPGANer(model_path='cache/GFPGANv1.4.pth', upscale=1)
        self.face_analyser = FaceAnalysis(name='buffalo_l')
        self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))


    def get_face(self,img_data):
        analysed = self.face_analyser.get(img_data)
        try:
            largest = max(analysed, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            return largest
        except:
            print("No face found") 
            return None
        

    def predict_face_swap(self,target_image_path: Path, swap_image_path: Path) -> Path:
        """Run a single prediction to swap faces between two images."""
        try:
            frame = cv2.imread(str(target_image_path))
            face = self.get_face(frame)
            source_face = self.get_face(cv2.imread(str(swap_image_path)))
            try:
                print(frame.shape, face.shape, source_face.shape)
            except:
                print("Printing shapes failed.")
            
            result = self.face_swapper.get(frame, face, source_face, paste_back=True)
            _, _, result = self.face_enhancer.enhance(result, paste_back=True)
            out_path = Path(tempfile.mkdtemp()) / f"{int(time.time())}.jpg"
            cv2.imwrite(str(out_path), result)
            return out_path
        except Exception as e:
            print(f"Error: {e}")
            return None


    def predict(
        self,
        main_face_image: Path = Input(description="ID image (main)"),
        auxiliary_face_image1: Path = Input(description="Additional ID image (auxiliary)", default=None),
        auxiliary_face_image2: Path = Input(description="Additional ID image (auxiliary)", default=None),
        auxiliary_face_image3: Path = Input(description="Additional ID image (auxiliary)", default=None),
        prompt: str = Input(
            description="Prompt", default="portrait,color,cinematic,in garden,soft light,detailed face"
        ),
        negative_prompt: str = Input(description="Negative Prompt", default=DEFAULT_NEGATIVE_PROMPT),
        cfg_scale: float = Input(
            description="CFG, recommend value range [1, 1.5], 1 will be faster", ge=1.0, le=1.5, default=1.2
        ),
        num_steps: int = Input(description="Steps", ge=1, le=100, default=4),
        image_height: int = Input(description="Height", ge=512, le=2024, default=1024),
        image_width: int = Input(description="Width", ge=512, le=2024, default=768),
        identity_scale: float = Input(description="ID scale", ge=0.0, le=5.0, default=0.8),
        generation_mode: str = Input(description="mode", choices=["fidelity", "extremely style"], default="fidelity"),
        mix_identities: bool = Input(
            description="ID Mix (if you want to mix two ID image, please turn this on, otherwise, turn this off)",
            default=False,
        ),
        seed: int = Input(description="Random seed. Leave blank to randomize the seed", default=None),
        num_samples: int = Input(description="Num samples", ge=1, le=8, default=4),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality of the output images, from 0 to 100. 100 is best quality, 0 is lowest quality.",
            default=80,
            ge=0,
            le=100,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        print(f"[!] ({type(main_face_image)}) main_face_image={main_face_image}")
        print(f"[!] ({type(auxiliary_face_image1)}) auxiliary_face_image1={auxiliary_face_image1}")
        print(f"[!] ({type(auxiliary_face_image2)}) auxiliary_face_image2={auxiliary_face_image2}")
        print(f"[!] ({type(auxiliary_face_image3)}) auxiliary_face_image3={auxiliary_face_image3}")
        print(f"[!] ({type(prompt)}) prompt={prompt}")
        print(f"[!] ({type(negative_prompt)}) negative_prompt={negative_prompt}")
        print(f"[!] ({type(cfg_scale)}) cfg_scale={cfg_scale}")
        print(f"[!] ({type(num_samples)}) num_samples={num_samples}")
        print(f"[!] ({type(seed)}) seed={seed}")
        print(f"[!] ({type(num_steps)}) num_steps={num_steps}")
        print(f"[!] ({type(image_height)}) image_height={image_height}")
        print(f"[!] ({type(image_width)}) image_width={image_width}")
        print(f"[!] ({type(identity_scale)}) identity_scale={identity_scale}")
        print(f"[!] ({type(generation_mode)}) generation_mode={generation_mode}")
        print(f"[!] ({type(mix_identities)}) mix_identities={mix_identities}")

        # Convert PIL Images to NumPy arrays right after opening
        main_face_image_np = np.array(Image.open(str(main_face_image))) if main_face_image else None
        auxiliary_face_image1_np = np.array(Image.open(str(auxiliary_face_image1))) if auxiliary_face_image1 else None
        auxiliary_face_image2_np = np.array(Image.open(str(auxiliary_face_image2))) if auxiliary_face_image2 else None
        auxiliary_face_image3_np = np.array(Image.open(str(auxiliary_face_image3))) if auxiliary_face_image3 else None

        inps = [
            main_face_image_np,
            auxiliary_face_image1_np,
            auxiliary_face_image2_np,
            auxiliary_face_image3_np,
            prompt,
            negative_prompt,
            cfg_scale,
            num_samples,
            seed,
            num_steps,
            image_height,
            image_width,
            identity_scale,
            generation_mode,
            mix_identities,
        ]

        output, _ = run(*inps)
        all_black = all(is_black_image(img_array) for img_array in output)

        if all_black:
            print("[!] All images were black")
            self.black_image_count += 1
            print(f"All generated images are black. Black image count: {self.black_image_count}")
            if self.black_image_count >= self.threshold:
                print("[~] Threshold reached. Re-running setup...")
                self.setup()
                output, _ = run(*inps)
                self.black_image_count = 0  # Reset the counter after re-setup

       # Save images and collect their paths
        saved_paths = []
        for idx, img_array in enumerate(output):
            img = Image.fromarray(img_array)
            extension = output_format.lower()
            extension = "jpeg" if extension == "jpg" else extension
            output_path = f"output_image_{idx}.{extension}"

            print(f"[~] Saving to {output_path}...")
            print(f"[~] Output format: {extension.upper()}")
            if output_format != "png":
                print(f"[~] Output quality: {output_quality}")

            save_params = {"format": extension.upper()}
            if output_format != "png":
                save_params["quality"] = output_quality
                save_params["optimize"] = True

            img.save(output_path, **save_params)
            saved_paths.append(Path(output_path))

        face_swap_paths = []
        try:
            for saved_path in saved_paths:
                face_swap_path = self.predict_face_swap(saved_path, main_face_image)
                # open the image
                img = Image.open(face_swap_path)
                # save the image as a webp
                img.save(face_swap_path.with_suffix(".webp"), "WEBP", quality=80)
                face_swap_paths.append(face_swap_path.with_suffix(".webp"))
            return face_swap_paths
        except Exception as e:
            return saved_paths
            