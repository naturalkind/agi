import torch
import time
from transformers import AutoTokenizer
import intel_extension_for_pytorch as ipex
from accelerate.utils import is_xpu_available, is_ipex_available

cuda_available = torch.cuda.is_available()
xpu_available = is_xpu_available() and is_ipex_available()

############# code changes ###############
if xpu_available:
    from transformers import AutoModelForCausalLM
    # Выбираем конкретное XPU устройство
    torch.xpu.set_device(0)  # Устанавливаем устройство с ID 0
    [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())]
    print(f"ipex_{ipex.__version__} torch.xpu {torch.xpu.is_available()}")
##########################################

PHI3_PROMPT_FORMAT = "<|user|>\n{prompt}<|end|>\n<|assistant|>"
model_id = "microsoft/Phi-3-mini-4k-instruct"
_prompt = "Написать сложный парсер текста Python"
_max_tokens = 512*2

if xpu_available:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_cache=True,
        attn_implementation='eager'
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        trust_remote_code=True,
        use_cache=True,
    )

############# code changes ###############
device_type = "cpu"
if xpu_available:
    device_type = "xpu:0"  # Явно указываем ID устройства
elif cuda_available:
    device_type = "cuda"
print(">>>>>>>>>>>>>>>>>>", device_type)

if xpu_available:
    model = model.to(device_type)
##########################################

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

with torch.inference_mode():
    prompt = PHI3_PROMPT_FORMAT.format(prompt=_prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device_type)
    
    # Разогрев
    output = model.generate(input_ids, max_new_tokens=_max_tokens)
    
    # Запуск инференса
    st = time.time()
    output = model.generate(input_ids, do_sample=False, max_new_tokens=_max_tokens)
    
    if xpu_available:
        torch.xpu.synchronize()
    elif cuda_available:
        torch.cuda.synchronize()
    else:
        torch.cpu.synchronize()
        
    end = time.time()
    output_str = tokenizer.decode(output[0], skip_special_tokens=False)
    output_str = (
        output_str.replace("<|user|>", "")
        .replace("<|assistant|>", "")
        .replace("<|end|>", "")[len(_prompt) + 4 :]
    )
    
    print(f"Inference time: {end-st} s")
    max_memory = (
        torch.cuda.max_memory_allocated()
        if cuda_available
        else torch.xpu.max_memory_allocated()
    )
    print(f"Max memory allocated: {max_memory / (1024 ** 3):02} GB")
    print("-" * 20, "Prompt", "-" * 20)
    print(
        prompt.replace("<|user|>", "")
        .replace("<|assistant|>", "")
        .replace("<|end|>", "")
    )
    print("-" * 20, "Output", "-" * 20)
    print(output_str)
