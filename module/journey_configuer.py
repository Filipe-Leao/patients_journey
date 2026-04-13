
#Import Library
import pandas as pd
import google.generativeai as genai
import os
import torch
import re
#from transformers import pipeline, GenerationConfig
from huggingface_hub import snapshot_download
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bert_score import score
from sacrebleu.metrics import BLEU
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
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

#text generation by gemini api
def generate_text_with_gemini(prompt, config):
    genai.configure(api_key=config["API_KEY"])
    #print("Available models: ", [model.name for model in genai.list_models()])
    model = genai.GenerativeModel("models/gemini-3.1-flash-lite-preview")  # Or use gemini-1.5-flash if required
    response = model.generate_content(prompt)
    return response.text

def load_pipeline(config):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_ID = "Qwen/Qwen3-14B"
    LOCAL_DIR = os.path.join(BASE_DIR, "../../models/Qwen/Qwen3-14B")
    
    try:                    
        return LLM(
            model=LOCAL_DIR,
            #quantization="awq",           # remove se usares BF16
            dtype="auto",
            max_model_len=8192,
            gpu_memory_utilization=0.92,
            tensor_parallel_size=1,
            enable_prefix_caching=True,   # cache do "/no_think" system prompt → speedup grátis
            max_num_batched_tokens=16384,
            max_num_seqs=512,
            #disable_log_requests=True,
        )
    except:
        print("Modelo não encontrado localmente. A fazer download...")
        
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        
        return LLM(
            model=LOCAL_DIR,
            #quantization="awq",           # remove se usares BF16
            dtype="auto",
            max_model_len=8192,
            gpu_memory_utilization=0.92,
            tensor_parallel_size=1,
            enable_prefix_caching=True,   # cache do "/no_think" system prompt → speedup grátis
            max_num_batched_tokens=16384,
            max_num_seqs=512,
            #disable_log_requests=True,
        )
        


def generate_text_with_local_model_batch(
    model: LLM,
    prompts: list[str],
    config=None,
) -> list[str]:

    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
    )

    # Constrói os prompts com o chat template — igual ao teu código
    tokenizer = model.get_tokenizer()
    formatted = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "/no_think"},
                {"role": "user",   "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,  # ← estava False (bug): o modelo não gerava corretamente
        )
        for prompt in prompts
    ]

    # vLLM processa o batch inteiro de uma vez, com continuous batching
    outputs = model.generate(formatted, sampling_params)

    return [out.outputs[0].text.strip() for out in outputs]


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
            using appropriate medical abbreviations (e.g., HTA, DM).
            3. Do not include treatment details, exam results, specific diagnoses, or follow-up treatments.
            4. Conclude the report with an indication of the initial treatment provided, 
            specifying the administered dose, but avoid explicitly labelling this section as 'initial treatment.'
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
    output_path = os.path.join(BASE_DIR, config["OUTPUT_PATH"])
    updated_file_path = os.path.join(BASE_DIR, config["CASE_REPORT_CSV_PATH"][:-4] + "_new.csv")
    updated_file_path_output = os.path.join(output_path, "synthetic_admission_report.csv")
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
    size = 1000
    for i in range(0, len(prompts), size):
        batch_prompts  = prompts[i:i+size]
        batch_indices  = valid_indices[i:i+size]

        reports = generate_text_with_local_model_batch(model, batch_prompts, config)

        for i_local, (index, report) in enumerate(zip(batch_indices, reports)):
            print(f"GEN: {i + i_local + 1}/{len(prompts)}\n{report}")
            case_report.loc[index, 'syn_admission_report'] = report or "Report generation failed"

        case_report.to_csv(updated_file_path, index=False, encoding='utf-8-sig')
        case_report.to_csv(updated_file_path_output, index=False, encoding='utf-8-sig')
        print(f"Checkpoint guardado: {i + len(batch_prompts)}/{len(prompts)}")

#DISCHARGE REPORT GEN
def discharge_report_generation(model, config):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, config["CASE_REPORT_CSV_PATH"][:-4] + "_new.csv")
    case_report = pd.read_csv(file_path)
    file_name = os.path.splitext(os.path.basename(config["CASE_REPORT_CSV_PATH"]))[0]

    output_path = os.path.join(BASE_DIR, config["OUTPUT_PATH"])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Recolhe todos os prompts primeiro
    prompts = []
    indices = []

    for index, row in case_report.iterrows():
        clinical_narrative = row[config["CASE_REPORT_COLUMN_NAME"]]
        admission_report = row['syn_admission_report']
        if clinical_narrative:
            prompt = f"""
            "{clinical_narrative}" and "{admission_report}"

            Based on the information above, write a realistic medical discharge report in {config["GEN_LANGUAGE"]} 
            for a patient upon leaving the hospital. Use the information provided in {config["GEN_LANGUAGE"]} and follow 
            the writing style and terminology consistent with {config["GEN_LANGUAGE"]} case report. 
            While writing, adopt the perspective of a doctor and remember this is not an admission report. 

            Follow these guidelines:
            1. Write the report as a single, unstructured paragraph in clinical language.
            2. Include a summary of the patient's stay in the hospital.
            3. Include treatment summary, details of exams and their results, discharge medications, and follow-up instructions.

            Ensure the report is in {config["GEN_LANGUAGE"]}.
            And feels authentic, mimicking how a doctor might write the discharge scenario. 
            Also, remember that doctors can make simple mistakes while writing (e.g., typographical mistakes).
            """
            prompts.append(prompt)
            indices.append(index)
        else:
            case_report.loc[index, 'syn_discharge_report'] = "Report generation failed"

    # Gera todos os reports de uma vez (batch)
    print(f"A gerar {len(prompts)} discharge reports...")
    reports = generate_text_with_local_model_batch(model, prompts, config)

    # Guarda os resultados
    for index, report in zip(indices, reports):
        print(f"GEN: {indices.index(index)+1}/{len(prompts)}\n{report}")
        case_report.loc[index, 'syn_discharge_report'] = report if report else "Report generation failed"

    # Guarda o CSV apenas uma vez no final
    updated_file_path = os.path.join(BASE_DIR, config["CASE_REPORT_CSV_PATH"][:-4] + "_new.csv")
    updated_file_path_output = os.path.join(output_path, file_name + "_synthetic_discharge_report.csv")
    case_report.to_csv(updated_file_path, index=False, encoding='utf-8-sig')
    case_report.to_csv(updated_file_path_output, index=False, encoding='utf-8-sig')
    print(f"DataFrame guardado em: {updated_file_path}")

# FULL JOURNEY REPORT GEN
def patients_full_journey(model, config):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, config["CASE_REPORT_CSV_PATH"][:-4] + "_new.csv")
    case_report = pd.read_csv(file_path)
    file_name = os.path.splitext(os.path.basename(config["CASE_REPORT_CSV_PATH"]))[0]

    output_path = os.path.join(BASE_DIR, config["OUTPUT_PATH"])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Recolhe todos os prompts primeiro
    prompts = []
    indices = []

    for index, row in case_report.iterrows():
        admission_report = row['syn_admission_report']
        discharge_report = row['syn_discharge_report']

        prompt = f"""
        "{admission_report}" and "{discharge_report}"

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
        prompts.append(prompt)
        indices.append(index)

    # Gera todos os reports de uma vez (batch)
    print(f"A gerar {len(prompts)} full journey reports...")
    reports = generate_text_with_local_model_batch(model, prompts, config)

    # Guarda os resultados
    for index, report in zip(indices, reports):
        print(f"GEN: {indices.index(index)+1}/{len(prompts)}\n{report}")
        case_report.loc[index, 'syn_full_journey'] = report if report else "Full journey generation failed"

    # Guarda o CSV apenas uma vez no final
    updated_file_path = os.path.join(BASE_DIR, config["CASE_REPORT_CSV_PATH"][:-4] + "_new.csv")
    updated_file_path_output = os.path.join(output_path, file_name + "_synthetic_full_journey_report.csv")
    case_report.to_csv(updated_file_path, index=False, encoding='utf-8-sig')
    case_report.to_csv(updated_file_path_output, index=False, encoding='utf-8-sig')
    print(f"Finished: DataFrame guardado em: {updated_file_path_output}")

