import intel_extension_for_pytorch as ipex
import time, os
import torch
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoTokenizer
from openvino.runtime import Core
from typing import List, Dict

# Проверка доступности Intel GPU
print(f"XPU доступна: {torch.xpu.is_available()}")
print(f"Device count: {torch.xpu.device_count()}")

for device in Core().available_devices:
    print(device, Core().get_property(device, "FULL_DEVICE_NAME"))

## Функция для форматирования истории чата
def get_model_response(outputs, tokenizer):
    # Получаем полную декодированную последовательность
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Определяем маркеры начала/конца ответа
    assistant_start = "<|assistant|>"
    assistant_end = "<|end|>"
    
    # Извлекаем ответ между маркерами
    start_idx = full_text.rfind(assistant_start)
    if start_idx != -1:
        start_idx += len(assistant_start)
        end_idx = full_text.find(assistant_end, start_idx)
        response = full_text[start_idx:end_idx].strip() if end_idx != -1 else full_text[start_idx:].strip()
    else:
        response = full_text.strip()
    
    # Чистим артефакты токенизации
    response = tokenizer.clean_up_tokenization(response)
    return response


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.xpu.get_device_name(0), torch.xpu.is_available())  

class GenerationRequest(BaseModel):
    messages: list
    max_new_tokens: int = 250
    temperature: float = 0.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    ## Загрузка модели и токенизатора
    model_id = "/home/npu/sd/Phi-3.5-mini-instruct-openvino-4bit"
    model = OVModelForCausalLM.from_pretrained(model_id, device="GPU.0")

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    yield
    
    logger.info("Cleaning up...")
    torch.xpu.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(request: GenerationRequest):
    print ("GENERATE ---------------->", request.messages)
    try:
        s = time.time()
        # Генерация текста с использованием XPU
        formatted_prompt = tokenizer.apply_chat_template(
                                                    request.messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True
                                                )
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        s = time.time()
        # V1
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        
        # Декодирование результата
#        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        outputs = get_model_response(outputs, tokenizer)
        print ("END GENERATE ---------------->", outputs , f"\n End time {time.time()-s}")
        torch.xpu.empty_cache()
        return {"response": outputs}
    
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Generation failed")

#async def generate(request: GenerationRequest):
#    print ("GENERATE ---------------->", request.messages)
#    return {"response": "ОТВЕТ"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, ssl_keyfile="/home/npu/agi/YOURPRIVATE.key", ssl_certfile="/home/npu/agi/YOURPUBLIC.pem")
