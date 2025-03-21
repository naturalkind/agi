import os
import torch
import time
from transformers import AutoTokenizer
import intel_extension_for_pytorch as ipex
from accelerate.utils import is_xpu_available, is_ipex_available

cuda_available = torch.cuda.is_available()
xpu_available = is_xpu_available() and is_ipex_available()

if xpu_available:
    from transformers import AutoModelForCausalLM
    # Выбираем конкретное XPU устройство
    torch.xpu.set_device(0)
    [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())]
    print(f"ipex_{ipex.__version__} torch.xpu {torch.xpu.is_available()}")

PHI3_PROMPT_FORMAT = "<|user|>\n{prompt}<|end|>\n<|assistant|>"
model_id = "microsoft/Phi-3-mini-4k-instruct"
_prompt = "Написать сложный парсер текста Python"
_max_tokens = 512

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

device_type = "cpu"
if xpu_available:
    device_type = "xpu:0"
elif cuda_available:
    device_type = "cuda"
print(">>>>>>>>>>>>>>>>>>", device_type)

if xpu_available:
    # Явно освобождаем память перед загрузкой модели
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    model = model.to(device_type)
    
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

times = []
test_count = 10

with torch.inference_mode():
    prompt = PHI3_PROMPT_FORMAT.format(prompt=_prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device_type)
    
    # Разогрев с синхронизацией
    output = model.generate(input_ids, max_new_tokens=_max_tokens)
    for i in range(test_count):
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
        times.append(end-st)
        print(f"Inference time: {end-st} s")
        torch.xpu.empty_cache()
print (f"Среднее значение {test_count}: {sum(times)/test_count}")

