import torch
from diffusers import StableDiffusion3Pipeline
from src.stable_diffusion_3_custom import skip_guidance_call

MODEL_NAME = "stabilityai/stable-diffusion-3.5-medium"
SEED = 42
HEIGHT = 512
WIDTH = 1024

TEST = True
ORTH = False
NORM = True
GUIDANCE_SCALE = 7.0

PROMPT_DICT = {
    "avocado" : (
        "An avocado reading a 'Phylosophy' book riding on a train. "
        "The Dark Volcano, with brightly lighting red lava flowing from it, could be seen in a window."
    ),
    "cat-in-sofa" : (
        "A picture of a cat laying in a greed cozy sofa with a cozy aura and some purring sound."
    ),
    "girl-rose-aura" : (
        "A photo of a beautiful girl in a big night city street with a rose aura around her"
    ),
    "city-street" : (
        "A photo of a big city street with highly detailed cars, buildings, "
        "some handsome men and beautiful women in the sidewalks, "
        "some taxi and cars on the road."
    )
}

def clear_mem(obj):
    try:
        if hasattr(obj, '__dict__'):
            for key in list(obj.__dict__.keys()):
                clear_mem(getattr(obj, key))
                delattr(obj, key)
    except Exception as e:
        print(e)

class Prompts():
    def __init__(self):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            transformer=None,
            scheduler  =None,
            vae        =None
            # device="cuda"
        )
        self.pipe.to("cuda")

    @torch.no_grad()
    def get_prompt(self, prompt_dict):
        output = {}
        for k in prompt_dict.keys():
            prompt = prompt_dict[k]
            (
                prompt_embeds, 
                negative_prompt_embeds, 
                pooled_prompt_embeds, 
                negative_pooled_prompt_embeds
            ) = self.pipe.encode_prompt(
                prompt  =prompt,
                prompt_2=prompt,
                prompt_3=prompt,
            )
            output[k] = dict(
                prompt_embeds                =prompt_embeds, 
                negative_prompt_embeds       =negative_prompt_embeds, 
                pooled_prompt_embeds         =pooled_prompt_embeds, 
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds
            )

        return output

    def clear_mem(self):
        self.pipe.to('cpu')
        del self.pipe


if __name__ == "__main__":
    import os
    os.environ["HF_HOME"] = "/home/oleg/Coding/repos/HF_HOME"

    # CREATE PROMPTS AND CLEAR THE MEMORY:
    get_P = Prompts()
    prompts = get_P.get_prompt(PROMPT_DICT)
    get_P.clear_mem()
    torch.cuda.empty_cache()
    del get_P

    # CREATE RENEWED MODEL INSTANCE:
    StableDiffusion3Pipeline.__call__ = skip_guidance_call
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype   =torch.float16,
        text_encoder  =None,
        text_encoder_2=None,
        text_encoder_3=None,
        tokenizer     =None,
        tokenizer_2   =None,
        tokenizer_3   =None,
    )
    pipe.enable_model_cpu_offload()
    
    # SETTINGS ADDITION:
    additional_folder = "result_" + f"{'_'.join(
        (['orth'] if ORTH else []) + (['norm'] if NORM else [])
    )}"
    additional_folder = additional_folder if (ORTH or NORM) else additional_folder[:-1]


    # =============== Prompt loop ===============
    with torch.no_grad():
        for prompt_name in prompts.keys():
            BASE_FOLDER = f"generation_output_CFG_UNCOND/{additional_folder}/{prompt_name}/output"
            os.makedirs(
                BASE_FOLDER,
                exist_ok=True
            )

            input_prompt = prompts[prompt_name]
        
            if not TEST:
                # NO GUIDANCE, ORIG STEP COUNT:
                print("Experiment original")
                generator = torch.Generator(device="cpu").manual_seed(SEED)
                image_orig = pipe(
                    # prompt                          =prompt,
                    height                          =HEIGHT,
                    width                           =WIDTH,
                    num_inference_steps             =28,
                    generator                       =generator,
                    guidance_scale                  =GUIDANCE_SCALE,
                    skip_guidance                   =0, 
                    skip_steps_count                =0,
                    **input_prompt,
                ).images[0]
                image_orig.save(f"{BASE_FOLDER}/orig_28.png")

            # TARGET STEP COUNT:
            for num_steps in [14, 28]:
                os.makedirs(
                    f"{BASE_FOLDER}/{num_steps}", 
                    exist_ok=True
                )
                print("Experiment low number of steps")
                
                if num_steps != 28 and not TEST:
                    generator = torch.Generator(device="cpu").manual_seed(SEED)
                    image_small = pipe(
                        # prompt                          =prompt,
                        height                          =HEIGHT,
                        width                           =WIDTH,
                        num_inference_steps             =num_steps,
                        generator                       =generator,
                        guidance_scale                  =GUIDANCE_SCALE,
                        skip_guidance                   =0, 
                        skip_steps_count                =0,
                        **input_prompt,
                    ).images[0]
                    image_small.save(f"{BASE_FOLDER}/orig_{num_steps}.png")

                # Experiment modified
                for skip_steps_count in [2, 3, 4]:
                    os.makedirs(
                        f"{BASE_FOLDER}/{num_steps}/skip_steps_{skip_steps_count}", 
                        exist_ok=True
                    )

                    for skip_guidance in [0.6, 0.8, 1, 1.4, 1.8]:
                        print("Experiment modified")
                        generator = torch.Generator(device="cpu").manual_seed(SEED)
                        image_skip = pipe(
                            # prompt                          =prompt,
                            height                          =HEIGHT,
                            width                           =WIDTH,
                            num_inference_steps             =num_steps,
                            generator                       =generator,
                            guidance_scale                  =GUIDANCE_SCALE,
                            skip_guidance                   =skip_guidance, 
                            skip_steps_count                =skip_steps_count,
                            **input_prompt,
                            orthogonalize                   =ORTH,
                            normilize                       =NORM,
                        ).images[0]

                        image_skip.save(f"{BASE_FOLDER}/{num_steps}/skip_steps_{skip_steps_count}/skip_{skip_guidance:.2f}.png")
