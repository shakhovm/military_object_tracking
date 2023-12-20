import os

if __name__ == '__main__':
    model_name = "atom"
    lst = os.listdir("data/bounding_boxes")
    for filename in lst:
        print(filename, model_name)
        os.system(f"python model_evaluator.py {filename} 0 {model_name}")