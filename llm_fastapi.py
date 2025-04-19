import time, os, sys
import logging
import uuid

import intel_extension_for_pytorch as ipex
import torch
import openvino
from optimum.intel import OVModelForCausalLM
from optimum.intel.openvino import OVDiffusionPipeline
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from openvino.runtime import Core

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Dict
import base64

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

#class GenerationRequest(BaseModel):
#    messages: list
#    max_new_tokens: int = 250
#    temperature: float = 0.0

class TextGenerationRequest(BaseModel):
    messages: list
    max_new_tokens: int = 250
    temperature: float = 0.0

class ImageGenerationRequest(BaseModel):
    prompt: str
    message_id: str | None = None
    chat_id: str | None = None


#@asynccontextmanager
#async def lifespan(app: FastAPI):
#    global model, tokenizer
#    ## Загрузка модели и токенизатора
#    model_id = "/home/npu/sd/Phi-3.5-mini-instruct-openvino-4bit"
#    model = OVModelForCausalLM.from_pretrained(model_id, device="GPU.1")

#    # Загрузка токенизатора
#    tokenizer = AutoTokenizer.from_pretrained(model_id)
#    
#    yield
#    
#    logger.info("Cleaning up...")
#    torch.xpu.empty_cache()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, pipeline, model_translater, tokenizer_translate
    ## Загрузка модели и токенизатора
    model_id = "/home/npu/sd/Phi-3.5-mini-instruct-openvino-4bit"
    model = OVModelForCausalLM.from_pretrained(model_id, device="GPU.1")

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model_id = "/home/npu/sd/LCM_Dreamshaper_v7-int8-ov"
    pipeline = OVDiffusionPipeline.from_pretrained(model_id, device="GPU.1")

    #### переводчик v1

    tokenizer_translate = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    model_translater = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")    
    
    
    yield
    
    logger.info("Cleaning up...")
    torch.xpu.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(request: TextGenerationRequest):
    print ("GENERATE -->", request.messages)
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

#@app.post("/generate_image")
#async def generate_image(request: GenerationRequest):
#    # Tokenize text
#    tokenized_text = tokenizer([str(request.messages).lower()], return_tensors='pt')

#    # Perform translation and decode the output
#    translation = model_translater.generate(**tokenized_text)
#    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
#    images = pipeline(translated_text, num_inference_steps=10).images
#    images[0].show()
#    print ("GENERATE ---------------->", request.messages)
#    return {"response": "ОТВЕТ"}

@app.post("/generate_image")
async def generate_image(request: ImageGenerationRequest):
    # Tokenize text
    tokenized_text = tokenizer_translate([str(request.prompt).lower()], return_tensors='pt')
    
    # Perform translation and decode the output
    translation = model_translater.generate(**tokenized_text)
    translated_text = tokenizer_translate.batch_decode(translation, skip_special_tokens=True)[0]
    
    # Generate the image
    result = pipeline(translated_text, num_inference_steps=10)
    image = result.images[0]
    
    # Save image to a temporary file
    temp_file_path = f"temp_{uuid.uuid4()}.jpg"
    image.save(temp_file_path)
    
    # Send the image to another server
    try:
        with open(temp_file_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")
            data = {
                "message_id": request.message_id if hasattr(request, "message_id") else None,
                "chat_id": request.chat_id if hasattr(request, "chat_id") else None,
                "text": translated_text,
                "image": image_data
            }
            
            # Send the image to your Telegram bot server
            torch.xpu.empty_cache()
            return JSONResponse(content={"response": data})
    except Exception as e:
        result = {"status": "error", "message": f"Error sending image: {str(e)}"}
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, ssl_keyfile="/home/npu/agi/YOURPRIVATE.key", ssl_certfile="/home/npu/agi/YOURPUBLIC.pem")
