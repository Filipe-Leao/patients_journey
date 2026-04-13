import json
from module.report_generation import generator
from module.report_evaluation import evaluator
import os
import time
import argparse

start_time = time.time()
# Specify the path to your JSON config file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(BASE_DIR,"config.json")

# Open the file and load the JSON data
with open(json_file_path, 'r') as json_file:
    config = json.load(json_file)
    
parser = argparse.ArgumentParser()
parser.add_argument("--part", type=int, default=None)
args = parser.parse_args()

# Altera o path do CSV consoante a parte
if args.part:
    config["CASE_REPORT_CSV_PATH"] = config["CASE_REPORT_CSV_PATH"][:-4] + f"_part{args.part}.csv"

print(f"Time: {start_time}")
print("\nReport Gereration started...\n")
generator(config)
print(f"\nGeneration Finished!\n Time: {time.time()}\n")


print("\nEvaluation Started...\n")
evaluator(config)
print(f"\nEvaluation Finished...\n Time:{time.time()}\n")
end_time = time.time() 
duration = end_time - start_time
print(f"Full process done... \nStart Time: {start_time} \nEnd Time: {end_time} \nDuration: {duration}")





