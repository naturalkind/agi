import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

torch.random.manual_seed(0) 

# Путь к модели
model_path = "/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/Phi-3.5-vision-instruct/"

model = AutoModelForCausalLM.from_pretrained( 
    model_path,  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

tokenizer = AutoTokenizer.from_pretrained(model_path) 

messages = [ 
    {"role": "system", "content": "You are a helpful AI assistant."}, 
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}, 
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."}, 
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"}, 
] 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 1000, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

output = pipe(messages, **generation_args) 
print(output[0]['generated_text'])

#import torch
#from transformers import AutoModelForCausalLM, AutoTokenizer



## Загрузка модели с указанием типа данных
#model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)

## Загрузка токенизатора
#tokenizer = AutoTokenizer.from_pretrained(model_path)

## Перевод модели в режим вычислений
#model.eval()

## Если у вас есть GPU, переместите модель на него
#if torch.cuda.is_available():
#    model = model.cuda()

## Пример использования
#input_text = "Привет, как дела?"
#input_ids = tokenizer.encode(input_text, return_tensors="pt")

## Если модель на GPU, переместите входные данные тоже на GPU
#if torch.cuda.is_available():
#    input_ids = input_ids.cuda()

## Генерация ответа
#with torch.no_grad():
#    output = model.generate(input_ids, max_length=200)

#response = tokenizer.decode(output[0], skip_special_tokens=True)

#print(f"Ввод: {input_text}")
#print(f"Ответ модели: {response}")
