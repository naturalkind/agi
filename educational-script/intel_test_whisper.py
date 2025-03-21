import tornado.httpserver
import tornado.ioloop
import tornado.web
import ssl
import json
import requests
import logging
import zlib
import pickle
import zmq
import re
from zmq.asyncio import Context
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import redis
import sqlite3
from datetime import datetime
from io import StringIO
import aiohttp
import asyncio
import time
import os
from multiprocessing import Process, Queue
import multiprocessing as mp
from pydub import AudioSegment
from io import BytesIO
import soundfile as sf
import numpy as np
import librosa
import cairosvg
import intel_extension_for_pytorch as ipex
from contextlib import contextmanager
import gc


xpu_available = torch.xpu.is_available()
if xpu_available:
    [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())]

WHISPER_MODEL_ID = "whisper-large-v3"
device = "xpu:0"#"xpu:1/0"

# Инициализация Whisper
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_MODEL_ID,
    torch_dtype=torch.bfloat16,
    #low_cpu_mem_usage=True,
    use_safetensors=True
)
whisper_model.to(device)
whisper_model = ipex.optimize(whisper_model, dtype=torch.bfloat16)

whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    torch_dtype=torch.bfloat16,
    device=device,
    chunk_length_s=30,
    batch_size=16, 
)

times = []
test_count = 10
for i in range(test_count):
    st = time.time()    
    result = whisper_pipe("/home/npu/agi/data_users/clon_out.wav")
    end = time.time()
    times.append(end-st)
    #print (result, f"\nInference time: {end-st} s")   

print (f"Среднее значение {test_count}: {sum(times)/test_count}")


