
#Import Library
import pandas as pd
import os
import torch
import re
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from vllm import LLM, SamplingParams

#from unidecode import unidecode

# Check if GPU is available
def device(config):
    if config["GPU"]=="YES":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
    else:
        device= "cpu"
        print(f"Using device: {device}")
    return device

#CASE REPORT LOADED
def case_report_load(config):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, config["CASE_REPORT_CSV_PATH"])
    return pd.read_csv(file_path)

def load_pipeline(config):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_ID = config["MODEL_ID"]
    LOCAL_DIR = os.path.join(BASE_DIR, "../../models/" + MODEL_ID)
    num_gpus = config["GPU_NUMBER"] if config["GPU"] == "YES" else 0
    print(f"Loading model {MODEL_ID} from {LOCAL_DIR} with {num_gpus} GPUs...")
    
    if not os.path.exists(LOCAL_DIR):
        print("Modelo não encontrado localmente. A fazer download...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        
    
    return LLM(
        model=LOCAL_DIR,
        tensor_parallel_size=num_gpus,
        dtype="auto",
        tokenizer_mode="auto",
        max_model_len=32000,           # cover 24 146 + some headroom for output
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,   # essential for long contexts — avoids OOM
        max_num_batched_tokens=4098,   # reduce chunk size for better batching
        #max_num_seqs=16,               # further reduce for diversity
    )
    
 
#pip install pandas torch transformers numpy bert-score sacrebleu scikit-learn huggingface-hub vllm py-heideltime
    
#ml OpenMPI/5.0.3-GCC-13.3.0
#ml NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0

#ml CUDA/11.8.0
#ml cuDNN/8.7.0.84-CUDA-11.8.0
#ml NCCL/2.18.5-GCCcore-11.3.0-CUDA-11.8.0
#ml PyTorch/1.13.1-foss-2022a-CUDA-11.8.0

def generate_text_with_local_model_batch(
    model: LLM,
    prompts: list[str],
    config=None,
) -> list[str]:

    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        min_p=0.01,
        repetition_penalty=1.1,
    )

    tokenizer = model.get_tokenizer()
    formatted = [
        tokenizer.apply_chat_template(
            [
                #{"role": "system", "content": "/no_think"},
                {"role": "user",   "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    # vLLM processa o batch inteiro de uma vez, com continuous batching
    outputs = model.generate(formatted, sampling_params)

    return [out.outputs[0].text.strip() for out in outputs]

def clean_output(text):
    # Remove <think> and </think> from model output
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

def admission_report_generation(model, config):
    def generate_report(index, clinical_narrative, config):
        """Gera um único relatório via API."""
        prompt = f"""
            "{clinical_narrative}"
            
            Based on the information above, write a realistic medical admission report in {config["GEN_LANGUAGE"]}
            for a patient upon arrival at the hospital. Use the information provided in {config["GEN_LANGUAGE"]} 
            and follow the writing style and terminology consistent with provided {config["GEN_LANGUAGE"]} case report. 
            While writing, adopt the perspective of a doctor and remember this is not discharge report. 
            
            Follow these guidelines:
                1. Write the report as a single, unstructured paragraph in clinical language.
                2. Include only symptoms, signs, and relevant history of previous diseases,
                   using appropriate medical abbreviations (e.g., HTA, DM),
                3. Do not include treatment details, exam results, specific diagnoses, or follow-up treatments,
                4. Conclude the report with an indication of the initial treatment provided, specifying the administered dose, but avoid explicitly labelling this section as 'initial treatment.'
                5. Include time-related information, such as the duration of the hospital stay, and the dates of key events (e.g., admission, medicine administration).

            
            Ensure the report is in {config["GEN_LANGUAGE"]}
            and feels authentic, mimicking how a doctor might write the admission scenario. 
            Also remember, doctors can make simple mistakes while writing (e.g., typographical mistakes).
        """
        return index, prompt
    
    case_report=case_report_load(config)
    if isinstance(config["N_TESTING_ROW"], int):# Check if it's an integer
        print(config["N_TESTING_ROW"])
        case_report=case_report[0:config["N_TESTING_ROW"]]# Use only the specified number of rows
    elif config["N_TESTING_ROW"]=="all":# If it's the string "all"
        case_report # Use all rows
    else:
        case_report # Default case: no filtering

    print("\n number of row:",len(case_report))
    
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(BASE_DIR, "../output/" + config["MODEL_ID"])
    
    updated_file_path = os.path.join(BASE_DIR, config["CASE_REPORT_CSV_PATH"][:-4] + "_new.csv")
    file_name = os.path.splitext(os.path.basename(config["CASE_REPORT_CSV_PATH"]))[0]

    updated_file_path_output = os.path.join(output_path, file_name + "_synthetic_admission_report.csv")
    file_name = os.path.splitext(os.path.basename(config["CASE_REPORT_CSV_PATH"]))[0]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Recolhe todos os prompts primeiro
    prompts = []
    indices = []
    
    valid_rows = {
        index: row[config["CASE_REPORT_COLUMN_NAME"]]
        for index, row in case_report.iterrows()
        if row[config["CASE_REPORT_COLUMN_NAME"]]
    }

    invalid_indices = [
        index for index, row in case_report.iterrows()
        if not row[config["CASE_REPORT_COLUMN_NAME"]]
    ]

    # Marca as inválidas logo
    for index in invalid_indices:
        case_report.loc[index, 'syn_admission_report'] = "Report generation failed"

    # Processa as válidas em paralelo
    lock = threading.Lock()
    max_workers = 32
    results = {}  # dict mantém o alinhamento index→prompt

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(generate_report, index, narrative, config): index
            for index, narrative in valid_rows.items()
        }

        for future in as_completed(futures):
            try:
                index, prompt = future.result()
                with lock:
                    results[index] = prompt  # ← guarda junto, sem desalinhar
            except Exception as e:
                index = futures[future]
                with lock:
                    case_report.loc[index, 'syn_admission_report'] = "Report generation failed"
                print(f"[ERROR] Index {index}: {e}")

    # Reconstrói em ordem original após threading
    valid_indices = list(valid_rows.keys())
    prompts = [results[index] for index in valid_indices if index in results]
    valid_indices = [index for index in valid_indices if index in results]
    
    print("Prompts created")

    # Gera todos os reports de uma vez (batch)
    print(f"A gerar {len(prompts)} reports...")
    
    valid_indices = list(valid_rows.keys())

    reports = generate_text_with_local_model_batch(model, prompts, config)

    for i, (index, report) in enumerate(zip(valid_indices, reports)):
        print(f"GEN: {i + 1}/{len(prompts)}")
        report = clean_output(report)
        case_report.loc[index, 'syn_admission_report'] = report or "Report generation failed"

    case_report.to_csv(updated_file_path, index=False, encoding='utf-8-sig')
    case_report.to_csv(updated_file_path_output, index=False, encoding='utf-8-sig')

#DISCHARGE REPORT GEN
def discharge_report_generation(model, config):
    def generate_report(index, clinical_narrative, admission_report, config):
        prompt = f"""
            Clinical Narrative: "{clinical_narrative}" 
            Admission Report: "{admission_report}"

            Based on the information above, write a realistic medical discharge report in {config["GEN_LANGUAGE"]} 
            for a patient upon leaving the hospital. Use the information provided in {config["GEN_LANGUAGE"]} and follow 
            the writing style and terminology consistent with {config["GEN_LANGUAGE"]} case report. 
            While writing, adopt the perspective of a doctor and remember this is not an admission report. 

            Follow these guidelines:
                1. Write the report as a single, unstructured paragraph in clinical language.
                2. Include a summary of the patient's stay in the hospital.
                3. Include treatment summary, details of exams and their results, discharge medications, and follow-up instructions.
                4. Include time-related information, such as the duration of the hospital stay, and the dates of key events (e.g., surgery, discharge, medicine administration).
                5. Do not repeat information already mentioned in the admission report.

            Ensure the report is in {config["GEN_LANGUAGE"]}.
            And feels authentic, mimicking how a doctor might write the discharge scenario. 
            Also, remember that doctors can make simple mistakes while writing (e.g., typographical mistakes).
            """
            
        return index, prompt

    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    file_path: str = os.path.join(BASE_DIR, config["CASE_REPORT_CSV_PATH"][:-4] + "_new.csv")
    case_report = pd.read_csv(file_path)
    file_name = os.path.splitext(os.path.basename(config["CASE_REPORT_CSV_PATH"]))[0]

    output_path: str = os.path.join(BASE_DIR, "../output/" + config["MODEL_ID"])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("\n number of row:", len(case_report))

    valid_rows = {
        index: (row[config["CASE_REPORT_COLUMN_NAME"]], row["syn_admission_report"])
        for index, row in case_report.iterrows()
        if row[config["CASE_REPORT_COLUMN_NAME"]] and row["syn_admission_report"] != "Report generation failed"
    }

    invalid_indices = [
        index for index, row in case_report.iterrows()
        if not row[config["CASE_REPORT_COLUMN_NAME"]] or row["syn_admission_report"] == "Report generation failed"
    ]

    for index in invalid_indices:
        case_report.loc[index, 'syn_discharge_report'] = "Report generation failed"

    lock = threading.Lock()
    results = {}

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(generate_report, index, clinical_narrative, admission_report, config): index
            for index, (clinical_narrative, admission_report) in valid_rows.items()
        }

        for future in as_completed(futures):
            try:
                index, prompt = future.result()
                with lock:
                    results[index] = prompt
            except Exception as e:
                index = futures[future]
                with lock:
                    case_report.loc[index, 'syn_discharge_report'] = "Report generation failed"
                print(f"[ERROR] Index {index}: {e}")

    valid_indices = [index for index in valid_rows.keys() if index in results]
    prompts = [results[index] for index in valid_indices]

    print("Prompts created")
    print(f"A gerar {len(prompts)} discharge reports...")

    updated_file_path = os.path.join(BASE_DIR, config["CASE_REPORT_CSV_PATH"][:-4] + "_new.csv")
    updated_file_path_output = os.path.join(output_path, file_name + "_synthetic_discharge_report.csv")
        
    valid_indices = list(valid_rows.keys())

    reports = generate_text_with_local_model_batch(model, prompts, config)

    for i, (index, report) in enumerate(zip(valid_indices, reports)):
        print(f"GEN: {i + 1}/{len(prompts)}")
        report = clean_output(report)
        case_report.loc[index, 'syn_discharge_report'] = report or "Report generation failed"

    case_report.to_csv(updated_file_path, index=False, encoding='utf-8-sig')
    case_report.to_csv(updated_file_path_output, index=False, encoding='utf-8-sig')

# FULL JOURNEY REPORT GEN
def patients_full_journey(model, config):
    def generate_report(index, admission_report, discharge_report, config):
        prompt = f"""
        Admission Report: "{admission_report}"
        Discharge Report: "{discharge_report}"

        Based on the admission and discharge reports provided, generate a detailed 
        report of the patient's full journey during their hospital stay. 
        Divide the information into multiple reports, such as 'divided days in different report based on patients situations' 
        'Surgery Report,' and so on, as appropriate to the events mentioned in the discharge report. 
        If the patient underwent surgery or any operation during their stay, 
        create a separate report detailing that specific event. 

        Write from the perspective of a doctor, ensuring the language feels authentic and mimics how 
        a doctor might document such scenarios. The report should be written in {config["GEN_LANGUAGE"]} 
        and formatted as a single, unstructured paragraph in clinical language. 
        Introduce small, natural errors like typographical mistakes to reflect a realistic documentation style.

        The generation should be in this order,
        1. Admission report (do not include date in the heading)
        2. Several reports based on patients situations during stay in the hospital. The report should be in day wise.
        3. Discharge Report (do not include date in the heading and also must mention the whole day of staying in the hospital)
        """
        return index, prompt

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, config["CASE_REPORT_CSV_PATH"][:-4] + "_new.csv")
    case_report = pd.read_csv(file_path)
    file_name = os.path.splitext(os.path.basename(config["CASE_REPORT_CSV_PATH"]))[0]
    output_path = os.path.join(BASE_DIR, "../output/" + config["MODEL_ID"])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("\n number of row:", len(case_report))

    valid_rows = {
        index: (row["syn_admission_report"], row["syn_discharge_report"])
        for index, row in case_report.iterrows()
        if row["syn_admission_report"] != "Report generation failed"
        and row["syn_discharge_report"] != "Report generation failed"
    }

    invalid_indices = [
        index for index, row in case_report.iterrows()
        if row["syn_admission_report"] == "Report generation failed"
        or row["syn_discharge_report"] == "Report generation failed"
    ]

    for index in invalid_indices:
        case_report.loc[index, 'syn_full_journey'] = "Full journey generation failed"

    lock = threading.Lock()
    results = {}

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(generate_report, index, admission_report, discharge_report, config): index
            for index, (admission_report, discharge_report) in valid_rows.items()
        }

        for future in as_completed(futures):
            try:
                index, prompt = future.result()
                with lock:
                    results[index] = prompt
            except Exception as e:
                index = futures[future]
                with lock:
                    case_report.loc[index, 'syn_full_journey'] = "Full journey generation failed"
                print(f"[ERROR] Index {index}: {e}")

    valid_indices = [index for index in valid_rows.keys() if index in results]
    prompts = [results[index] for index in valid_indices]

    print("Prompts created")
    print(f"A gerar {len(prompts)} full journey reports...")

    updated_file_path = os.path.join(BASE_DIR, config["CASE_REPORT_CSV_PATH"][:-4] + "_new.csv")
    updated_file_path_output = os.path.join(output_path, file_name + "_synthetic_full_journey_report.csv")

    valid_indices = list(valid_rows.keys())

    reports = generate_text_with_local_model_batch(model, prompts, config)

    for i, (index, report) in enumerate(zip(valid_indices, reports)):
        print(f"GEN: {i + 1}/{len(prompts)}")
        report = clean_output(report)
        case_report.loc[index, 'syn_full_journey'] = report or "Report generation failed"

    case_report.to_csv(updated_file_path, index=False, encoding='utf-8-sig')
    case_report.to_csv(updated_file_path_output, index=False, encoding='utf-8-sig')

    print(f"Finished: DataFrame guardado em: {updated_file_path_output}")