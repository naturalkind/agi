from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json

def mistral(messages):
    device = "cuda" # the device to load the model onto
    
    model = AutoModelForCausalLM.from_pretrained("Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained("Mistral-7B-Instruct-v0.2")
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=10000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    #print(decoded[0])
    with open("generated_texts.txt", "w") as f:
        f.write(decoded[0])    

def codellama(messages):
    print ("CodeLlama")
    device = "cuda" # the device to load the model onto
    quantization_config = BitsAndBytesConfig(
                                                load_in_4bit=True,
                                                bnb_4bit_use_double_quant=True,
                                                bnb_4bit_quant_type="nf4",
                                                bnb_4bit_compute_dtype=torch.bfloat16
                                            )


    model = AutoModelForCausalLM.from_pretrained("CodeLlama-13b-Python-hf", quantization_config=quantization_config)#,
    tokenizer = AutoTokenizer.from_pretrained("CodeLlama-13b-Python-hf")
    
    ### простой запрос к моделе 
    pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=10000,
            do_sample=True,
            temperature=0.7,
        )

    print (pipe("search for duplicate images")) 


    ### сложный запрос
#    prompt = 'def remove_non_ascii(s: str) -> str:\n    """ '
#    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

#    output = model.generate(
#        inputs["input_ids"],
#        max_new_tokens=10000,
#        do_sample=True,
#        top_p=0.9,
#        temperature=0.1,
#    )
#    output = output[0].to("cpu")
#    with open("generated_texts.txt", "w") as f:
#        f.write(tokenizer.decode(output))    

    
inf = """
import cv2
import numpy as np
import time
import yams
from yamspy import YAML
from pid import PID

class Drone:
    # Drone class definition

class TrackerCSRT:
    # TrackerCSRT class definition

    def __init__(self):
        self.tracker = cv2.TrackerCSRT_create()

    def init(self, frame, target_bbox):
        self.tracker.init(frame, target_bbox)

    def update(self, frame):
        success, bbox = self.tracker.update(frame)
        if success:
            return bbox
        else:
            return None

class PIDController:
    # PID controller class definition

class DroneController:
    def __init__(self, drone, tracker, pid_controller):
        self.drone = drone
        self.tracker = tracker
        self.pid_controller = pid_controller

    def fly_behind_target(self):
        drone_state = self.drone.get_state()

        captured_frame = self.drone.capture_image()
        target_bbox = self.tracker.update(captured_frame)

        if target_bbox is not None:
            target_position = (target_bbox[0] + target_bbox[1] // 2, target_bbox[2] + target_bbox[3] // 2)

            desired_movements = self.pid_controller.calculate_desired_movements(
                drone_state, target_position
            )

            self.drone.send_commands(desired_movements)

def main():
    drone = Drone()  # Initialize drone class instance
    tracker = TrackerCSRT()  # Initialize object tracker class instance
    pid_controller = PIDController()  # Initialize PID controller class instance
    drone_controller = DroneController(drone, tracker, pid_controller)

    drone.init()  # Initialize drone hardware and start camera stream

    while True:
        drone_controller.fly_behind_target()

if __name__ == "__main__":
    main()
"""

if __name__ == '__main__':
    messages = [
                {"role": "user", "content": f"add full code for control a drone flying behind a target using the TrackerCSRT object tracker {inf}"},
#                {"role": "user", "content": f"write a program to control a drone flying behind a target using the TrackerCSRT object tracker. In python using the yamspy library to control a drone using the YAW ROLL PITCH throttle PID control"}

                ]

    mistral(messages)
    
    #codellama(messages)

    
    

