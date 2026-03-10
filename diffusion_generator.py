"""
diffusion_generator.py

Wraps a Stable Diffusion pipeline (runwayml/stable-diffusion-v1-5) to generate
forensic-style pencil face sketches from plain English descriptions.
"""

import torch
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

# ── Prompt templates ─────────────────────────────────────────────────────────

POSITIVE_PREFIX = (
    "police forensic composite sketch of a single person's face, detailed graphite pencil drawing, "
    "highly detailed face, realistic anatomic proportions, front view, sharp lines, masterfully drawn, "
    "professional composite artist, black and white sketch, "
)

NEGATIVE_PROMPT = (
    "color, photorealistic, photo, painting, cartoon, anime, 3d render, "
    "blurry, low quality, distorted, deformed faces, asymmetrical features, multiple faces, "
    "text, watermark, signature, full body"
)

REAL_PREFIX = (
    "photorealistic face portrait, high resolution photograph, 8k, detailed skin texture, "
    "raw photo, DSLR 85mm lens, highly detailed, realistic proportions, authentic, "
)

REAL_NEGATIVE = (
    "sketch, drawing, illustration, painting, cartoon, anime, 3d render, "
    "blurry, low quality, deformed, asymmetrical, multiple faces, "
    "text, watermark, full body"
)


def _to_sketch(pil_img: Image.Image) -> Image.Image:
    """Apply pencil-sketch post-processing to a PIL image."""
    gray = pil_img.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=2)
    gray = gray.filter(ImageFilter.SHARPEN)
    gray = gray.filter(ImageFilter.EDGE_ENHANCE_MORE)
    gray = ImageEnhance.Contrast(gray).enhance(1.4)
    return gray.convert("RGB")


class DiffusionSketchGenerator:
    """
    Generates forensic face sketches from text descriptions using
    Stable Diffusion v1-5.

    First call downloads ~4 GB model weights to ~/.cache/huggingface.
    Subsequent calls use the cached weights.
    """

    MODEL_ID = "runwayml/stable-diffusion-v1-5"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.pipe = None

    def load(self):
        """Load the pipeline (call once at startup)."""
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

        print(f"[DiffusionGen] Loading {self.MODEL_ID} on {self.device} ...")
        print("[DiffusionGen] First run will download ~4 GB — please wait...")

        # float16 is unstable on MPS; use float32
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,       # skip NSFW filter for speed
            requires_safety_checker=False,
        )
        
        # Optimize with DPMSolverMultistepScheduler for faster inference
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        self.pipe = self.pipe.to(self.device)
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(**self.pipe.components)

        # Memory optimisations
        if self.device in ("mps", "cuda"):
            self.pipe.enable_attention_slicing()
            self.img2img_pipe.enable_attention_slicing()

        print("[DiffusionGen] Model loaded and ready.")

    def generate(
        self,
        description: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int | None = None,
        style: str = "sketch",
        view_modifier: str = ""
    ) -> Image.Image:
        """
        Generate a portrait from a plain-English description.

        Args:
            description: e.g. "young woman with long brown hair and glasses"
            num_inference_steps: more steps = better quality but slower
            guidance_scale: how strictly the image follows the prompt (7-9 recommended)
            seed: optional fixed seed for reproducibility; None = random each call
            style: "sketch" or "realistic"

        Returns:
            RGB PIL Image (512x512 by default)
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        if style == "realistic":
            prompt = REAL_PREFIX + view_modifier + description
            negative_prompt = REAL_NEGATIVE
        else:
            prompt = POSITIVE_PREFIX + view_modifier + description
            negative_prompt = NEGATIVE_PROMPT

        # Build generator for reproducibility (or random if seed is None)
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device).manual_seed(
                torch.randint(0, 2**31, (1,)).item()
            )

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=512,
            width=512,
        )

        pil_img = result.images[0]
        if style == "realistic":
            return pil_img
        return _to_sketch(pil_img)

    def generate_img2img(
        self,
        init_image: Image.Image,
        description: str,
        strength: float = 0.65,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> Image.Image:
        """
        Generate a refined sketch from an initial composite image and description.
        """
        if not hasattr(self, 'img2img_pipe') or self.img2img_pipe is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        prompt = POSITIVE_PREFIX + description

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device).manual_seed(
                torch.randint(0, 2**31, (1,)).item()
            )

        init_image = init_image.resize((512, 512)).convert("RGB")

        result = self.img2img_pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        pil_img = result.images[0]
        return _to_sketch(pil_img)
