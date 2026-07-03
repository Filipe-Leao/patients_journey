# Importing necessary libraries
import pandas as pd  # For handling and manipulating structured data
import os  # For operating system-related functionalities (e.g., file path handling)
import torch  # For deep learning models and computations
from transformers import pipeline  # For using pre-trained models from Hugging Face
import numpy as np  # For numerical computations
from bert_score import score, BERTScorer  # For evaluating text similarity using BERT embeddings
from sacrebleu.metrics import BLEU  # For BLEU score calculation (text similarity evaluation)
from sklearn.feature_extraction.text import TfidfVectorizer  # For text vectorization using TF-IDF
from sklearn.cluster import KMeans  # K-means clustering
from huggingface_hub import snapshot_download
import re
import json
from collections import defaultdict
from py_heideltime import heideltime
import concurrent.futures
from threading import Lock


_TEMPORAL_CACHE = {}
_TEMPORAL_CACHE_LOCK = Lock()

# Custom function from another module to define the device (CPU/GPU)
from module.journey_configuer import device

def load_ner(config):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_ID = config["NER_MODEL"]
    LOCAL_DIR = os.path.join(BASE_DIR, f"../../models/{MODEL_ID}")
    try:
        print("Loadin ner model")
        ner = pipeline('ner', model=LOCAL_DIR, aggregation_strategy='average', device=device(config))
        print("Modelo ner carregado localmente.")
    except Exception:
        print("Modelo ner não encontrado localmente. A fazer download...")
        
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        
        ner = pipeline('ner', model=LOCAL_DIR, aggregation_strategy='average', device=device(config))
        print("Modelo ner carregado localmente.")
    
    return ner

def load_bert(config):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_ID = "google-bert/bert-base-multilingual-cased"
    LOCAL_DIR = os.path.join(BASE_DIR, "../../models/google-bert/bert-base-multilingual-cased")
    try:
        print("Loadin model")
        scorer = BERTScorer(
            model_type=LOCAL_DIR, 
            device=device(config),
            lang="PT",
            num_layers=9,               # camada ótima para BERT-base
            #idf=True,
            #idf_sents=corpus,
            batch_size=32,              # reduz se tiveres pouca RAM/VRAM
            rescale_with_baseline=False, # True só funciona para modelos conhecidos do HF   
        )
        print("Modelo carregado localmente.")
    except Exception:
        print("Modelo não encontrado localmente. A fazer download...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # ignora pesos não PyTorch
        )
        print("Loadin model")
        scorer = BERTScorer(
            model_type=LOCAL_DIR, 
            device=device(config),
            lang="PT",
            num_layers=9,               # camada ótima para BERT-base
            #idf=True,
            #idf_sents=corpus,
            batch_size=32,              # reduz se tiveres pouca RAM/VRAM
            rescale_with_baseline=False # True só funciona para modelos conhecidos do HF
        )
        print("Modelo carregado localmente.")
    
    return scorer


# Function to extract named entities from text using the NER pipeline
def extract_ner(ner_pipeline, texts):
    """
    Extracts named entities from the provided text using the NER model pipeline.
    """
    def clean_entity(e):
        e = e.strip()
        e = re.sub(r'\s+', ' ', e)
        return e
    try:
        texts = [text.replace("\n", " ") for text in texts]
        texts = [re.sub(r"\s+", " ", text) for text in texts]
        ners = ner_pipeline(texts)  # Extract entities using the NER pipeline
        res = {
            (e['entity_group'], clean_entity(texts[i][e['start']:e['end']]))
            for ners in ners for i, e in enumerate([ners])  # Create a set of tuples (entity_group, entity_text)
        }  # Use a set to store unique entities
        return res
        
    except Exception as e:
        print(f"Error processing text: {e}")
        return []  # If there's an error, return an empty list
    

def extract_ner_batch(ner_pipeline, texts, batch_size=4):
    def clean_text(text):
        text = text.replace("\n", " ")
        return re.sub(r"\s+", " ", text)

    def clean_entity(value):
        return re.sub(r"\s+", " ", value.strip())

    cleaned_texts = [
        clean_text(text) if isinstance(text, str) else ""
        for text in texts
    ]

    batch_outputs = ner_pipeline(
        cleaned_texts,
        batch_size=batch_size,
        #truncation=True,
    )

    results = []
    for text, ner_list in zip(cleaned_texts, batch_outputs):
        entities = {
            (entity["entity_group"], clean_entity(text[entity["start"]:entity["end"]]))
            for entity in ner_list
        }
        results.append(entities)

    print(f"Extracted NER for {len(results)} texts.")
    print(f"Sample NER output: {results[0] if results else 'No results'}")
    return results

def _normalize_entity_text(value):
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _token_levenshtein_distance(tokens1, tokens2, max_distance=1):
    if abs(len(tokens1) - len(tokens2)) > max_distance:
        return max_distance + 1

    previous = list(range(len(tokens2) + 1))
    for i, t1 in enumerate(tokens1, start=1):
        current = [i]
        row_min = current[0]
        for j, t2 in enumerate(tokens2, start=1):
            cost = 0 if t1 == t2 else 1
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            ))
            if current[-1] < row_min:
                row_min = current[-1]

        if row_min > max_distance:
            return max_distance + 1

        previous = current

    return previous[-1]


def _is_relaxed_entity_match(entity_a, entity_b, max_token_distance=1):
    norm_a = _normalize_entity_text(entity_a)
    norm_b = _normalize_entity_text(entity_b)

    if not norm_a or not norm_b:
        return False

    if norm_a == norm_b:
        return True

    tokens_a = norm_a.split()
    tokens_b = norm_b.split()
    distance = _token_levenshtein_distance(tokens_a, tokens_b, max_distance=max_token_distance)
    return distance <= max_token_distance


def _entities_by_class(entities):
    grouped = defaultdict(set)
    for entity_class, entity_text in entities:
        grouped[entity_class].add(_normalize_entity_text(entity_text))
    return grouped


def ner_similarity(ner1, ner2):
    """
    Checks if all named entities in ner1 exist in ner2 and ner2 exist in ner1.
    If true, returns 1,1. Otherwise, calculates the percentage of ner1 entities present in ner2 and the percentage of ner2 entities present in ner1.

    Parameters:
    - ner1: List of named entities, where each entity is a dictionary with an 'entity' key.
    - ner2: List of named entities, where each entity is a dictionary with an 'entity' key.

    Returns:
    - list of 2 values:
        1st value:
            - 1 if all entities in ner2 exist in ner1.
            - A float value representing the percentage of ner2 entities found in ner1 if not all match.
        2st value:
            - 1 if all entities in ner1 exist in ner2.
            - A float value representing the percentage of ner1 entities found in ner2 if not all match.
    """
    entities1 = set(ner1)
    entities2 = set(ner2)

    by_class_1 = _entities_by_class(entities1)
    by_class_2 = _entities_by_class(entities2)
    all_classes = sorted(set(by_class_1.keys()).union(set(by_class_2.keys())))

    strict_lost_by_class = {}
    strict_lost_rate_by_class = {}
    relaxed_lost_by_class = {}
    relaxed_lost_rate_by_class = {}
    only_strict_lost_by_class = {}  # Entities lost in strict matching (before relaxed recovery)
    only_relaxed_lost_by_class = {}  # Entities lost in both strict and relaxed matching
    recovery_by_relaxed_by_class = {}  # Entities recovered by relaxed matching from strict

    strict_matched_total = 0
    relaxed_extra_total = 0
    total_entities_1 = 0
    not_found_by_class = {}

    for entity_class in all_classes:
        class_entities_1 = set(by_class_1.get(entity_class, set()))
        class_entities_2 = set(by_class_2.get(entity_class, set()))

        total_ref = len(class_entities_1)
        total_entities_1 += total_ref

        strict_matched = class_entities_1.intersection(class_entities_2)
        strict_matched_total += len(strict_matched)

        strict_lost = total_ref - len(strict_matched)
        strict_lost_by_class[entity_class] = strict_lost
        strict_lost_rate_by_class[entity_class] = (strict_lost / total_ref) if total_ref else 0.0

        unmatched_1 = list(class_entities_1 - strict_matched)
        unmatched_2 = list(class_entities_2 - strict_matched)
        used_2 = set()
        recovered_by_relaxed = []
        
        not_found = []

        for candidate_1 in unmatched_1:
            match_pos = -1
            for pos, candidate_2 in enumerate(unmatched_2):
                if pos in used_2:
                    continue
                if _is_relaxed_entity_match(candidate_1, candidate_2, max_token_distance=1):
                    match_pos = pos
                    break

            if match_pos >= 0:
                used_2.add(match_pos)
                recovered_by_relaxed.append(candidate_1)
            else:
                not_found.append(candidate_1)

        relaxed_extra_total += len(recovered_by_relaxed)
        relaxed_lost = strict_lost - len(recovered_by_relaxed)
        relaxed_lost_by_class[entity_class] = relaxed_lost
        relaxed_lost_rate_by_class[entity_class] = (relaxed_lost / total_ref) if total_ref else 0.0
        
        strict_lost_entities = sorted(unmatched_1)
        recovered_by_relaxed = sorted(recovered_by_relaxed)
        not_found = sorted(not_found)

        # Track the type of losses as entity lists instead of counts.
        # only_strict_lost: all entities lost in strict matching
        # only_relaxed_lost: entities not found even in relaxed matching
        recovery_by_relaxed_by_class[entity_class] = recovered_by_relaxed
        only_relaxed_lost_by_class[entity_class] = not_found
        only_strict_lost_by_class[entity_class] = strict_lost_entities

    total_entities_2 = len(entities2)
    strict_ner1_in_ner2 = (strict_matched_total / total_entities_1) if total_entities_1 else 0.0
    relaxed_ner1_in_ner2 = ((strict_matched_total + relaxed_extra_total) / total_entities_1) if total_entities_1 else 0.0

    reverse_exact = len(entities1.intersection(entities2))
    strict_ner2_in_ner1 = (reverse_exact / total_entities_2) if total_entities_2 else 0.0

    return {
        "strict": {
            "ner1_in_ner2": strict_ner1_in_ner2,
            "ner2_in_ner1": strict_ner2_in_ner1,
            "lost_by_class": strict_lost_by_class,
            "lost_rate_by_class": strict_lost_rate_by_class,
        },
        "relaxed": {
            "ner1_in_ner2": relaxed_ner1_in_ner2,
            "lost_by_class": relaxed_lost_by_class,
            "lost_rate_by_class": relaxed_lost_rate_by_class,
        },
        "not_found": not_found_by_class,
        "lost_types": {
            "Lost_in_strict": only_strict_lost_by_class,  # Lost in strict matching (before relaxed)
            "Lost_in_both": only_relaxed_lost_by_class,  # Lost in both strict and relaxed
            "Recovered_by_relaxed": recovery_by_relaxed_by_class,  # Entities recovered by relaxed
        }
    }

# Function to calculate the average BERT score between references and candidates
def calculate_bert_score(bert, references, candidates):
    """
    Calculates BERT score (precision, recall, F1) between the reference and candidate text.
    """
    P, R, F1 = bert.score(candidates, references, verbose=True)  # Calculate BERT score
    return F1.mean().item()  # Return the average F1 score

# Function to calculate BLEU score for text similarity
def calculate_bleu_score(bleu, references, candidates):
    """
    Calculates BLEU score for the similarity between references and candidates.
    """
    score = bleu.sentence_score(candidates, [references])  # Calculate BLEU score
    return score.score / 100.0  # Normalize BLEU score between 0 and 1

def get_temporal_expressions(text):
    """
    Extracts temporal expressions from the given text using Heideltime and returns a list of normalized temporal expressions.
    """
    if not isinstance(text, str):
        return []
    
    config_path = (os.path.dirname(os.path.abspath(__file__)))

    normalized_text = re.sub(r"\s+", " ", text).strip()
    if not normalized_text:
        return []

    with _TEMPORAL_CACHE_LOCK:
        cached = _TEMPORAL_CACHE.get(normalized_text)

    if cached is not None:
        return list(cached)

    try: 
        expressions = heideltime(
            normalized_text,
            language='Portuguese',
            document_type='Narrative',
        )
        seen = set()
        res = []
        for exp in expressions:
            key = (exp['text'], exp['type'], exp['value'])
            if key not in seen:
                seen.add(key)
                res.append({
                    'text': exp['text'],
                    'type': exp['type'],
                    'value': exp['value'],
                })

        with _TEMPORAL_CACHE_LOCK:
            _TEMPORAL_CACHE[normalized_text] = tuple(res)
    except Exception as e:
        print(f"Error occurred while extracting temporal expressions: {e}")
        res = []
    return res

def lost_temporal_expressions(temporal_expressions1, temporal_expressions2):
    """
    Compares two sets of temporal expressions and identifies which expressions from the first set are not present in the second set.
    """
    lost_expressions = []
    for exp1 in temporal_expressions1:
        found = False
        for exp2 in temporal_expressions2:
            if _is_relaxed_entity_match(exp1['text'], exp2['text'], max_token_distance=1):
                found = True
                break
        if not found:
            lost_expressions.append(exp1)
    return lost_expressions

# Main evaluation function that performs various evaluations on clinical narratives
def evaluator(config):
    """
    Evaluates clinical narratives using different metrics: NER, BERT score, BLEU score, clustering, and classification.
    """
    # Load the input data from a specified CSV file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, config["CASE_REPORT_CSV_PATH"][:-4]+"_new.csv")
    file_name = os.path.splitext(os.path.basename(config["CASE_REPORT_CSV_PATH"]))[0]
    gen_data=pd.read_csv(file_path)
    torch.cuda.empty_cache()  # Clear GPU cache to free up memory

    # Handling a subset of data based on the configuration
    if isinstance(config["N_TESTING_ROW"], int):  # Check if it's an integer
        gen_data = gen_data[0:config["N_TESTING_ROW"]]  # Use only the specified number of rows
    elif config["N_TESTING_ROW"] == "all":  # If it's the string "all"
        gen_data  # Use all rows
    else:
        gen_data  # Default case: no filtering

    
    # Initialize the NER pipeline using a pre-trained model
    print("NER model:", config["NER_MODEL"])
    ner_pipeline = load_ner(config)
    bleu = BLEU(effective_order=True)  # Initialize BLEU metric
    bert = load_bert(config)

    if config["SCORING"].lower() == "yes":
        print("I AM SCORE")
        temporal_max_workers = int(config.get("TEMPORAL_MAX_WORKERS", 4))
        temporal_max_workers = max(1, temporal_max_workers)

        # List to store evaluation results
        results = []

        # Reuse one executor across all rows to avoid pool creation overhead
        temporal_executor = None
        if temporal_max_workers > 1:
            print(f"Using ThreadPoolExecutor with {temporal_max_workers} workers for temporal expression extraction.")
            temporal_executor = concurrent.futures.ThreadPoolExecutor(max_workers=temporal_max_workers)

        # Iterate over the rows of the data to perform evaluations
        try:
            for index, row in gen_data.iterrows():  # Limit processing to 5 rows for demonstration
                try:
                    print(f"\nProcessing Patient {index}/{len(gen_data)}")
                    print("Processing NER calculation")
                    # Extract named entities from various text columns
                    admission_ner, discharge_ner, clinical_ner, journey_ner = extract_ner_batch(
                        ner_pipeline,
                        [
                            row["syn_admission_report"],
                            row["syn_discharge_report"],
                            row[config["CASE_REPORT_COLUMN_NAME"]],
                            row["syn_full_journey"],
                        ],
                        batch_size=4,
                    )
                    # Calculate NER-based similarity scores in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        future_admission = executor.submit(ner_similarity, clinical_ner, admission_ner)
                        future_discharge = executor.submit(ner_similarity, clinical_ner, discharge_ner)
                        future_journey = executor.submit(ner_similarity, clinical_ner, journey_ner)

                        ner_similarity_admission = future_admission.result()
                        ner_similarity_discharge = future_discharge.result()
                        ner_similarity_journey = future_journey.result()

                    print("Processing BERT score calculation")
                    # Calculate BERT scores for text similarity
                    bert_score_admission = calculate_bert_score(bert, [row[config["CASE_REPORT_COLUMN_NAME"]]], [row['syn_admission_report']])
                    bert_score_discharge = calculate_bert_score(bert, [row[config["CASE_REPORT_COLUMN_NAME"]]], [row['syn_discharge_report']])
                    bert_score_journey = calculate_bert_score(bert, [row[config["CASE_REPORT_COLUMN_NAME"]]], [row['syn_full_journey']])

                    print("Processing BLEU score calculation")
                    # Calculate BLEU scores for text similarity
                    bleu_score_admission = calculate_bleu_score(bleu, row[config["CASE_REPORT_COLUMN_NAME"]], row['syn_admission_report'])
                    bleu_score_discharge = calculate_bleu_score(bleu, row[config["CASE_REPORT_COLUMN_NAME"]], row['syn_discharge_report'])
                    bleu_score_journey = calculate_bleu_score(bleu, row[config["CASE_REPORT_COLUMN_NAME"]], row['syn_full_journey'])
                    """
                    # Get temporal expressions from the clinical report (configurable parallelism)
                    if temporal_executor is not None:
                        fut_clinical = temporal_executor.submit(get_temporal_expressions, row[config["CASE_REPORT_COLUMN_NAME"]])
                        fut_admission = temporal_executor.submit(get_temporal_expressions, row['syn_admission_report'])
                        fut_discharge = temporal_executor.submit(get_temporal_expressions, row['syn_discharge_report'])
                        fut_journey = temporal_executor.submit(get_temporal_expressions, row['syn_full_journey'])

                        temporal_expressions_clinical = fut_clinical.result()
                        temporal_expressions_admission = fut_admission.result()
                        temporal_expressions_discharge = fut_discharge.result()
                        temporal_expressions_journey = fut_journey.result()

                        # Compute lost temporal expressions in parallel
                        fut_lost_adm = temporal_executor.submit(lost_temporal_expressions, temporal_expressions_clinical, temporal_expressions_admission)
                        fut_lost_dis = temporal_executor.submit(lost_temporal_expressions, temporal_expressions_clinical, temporal_expressions_discharge)
                        fut_lost_jour = temporal_executor.submit(lost_temporal_expressions, temporal_expressions_clinical, temporal_expressions_journey)

                        lost_temporal_expressions_admission = fut_lost_adm.result()
                        lost_temporal_expressions_discharge = fut_lost_dis.result()
                        lost_temporal_expressions_journey = fut_lost_jour.result()
                    else:
                        temporal_expressions_clinical = get_temporal_expressions(row[config["CASE_REPORT_COLUMN_NAME"]])
                        temporal_expressions_admission = get_temporal_expressions(row['syn_admission_report'])
                        temporal_expressions_discharge = get_temporal_expressions(row['syn_discharge_report'])
                        temporal_expressions_journey = get_temporal_expressions(row['syn_full_journey'])

                        lost_temporal_expressions_admission = lost_temporal_expressions(temporal_expressions_clinical, temporal_expressions_admission)
                        lost_temporal_expressions_discharge = lost_temporal_expressions(temporal_expressions_clinical, temporal_expressions_discharge)
                        lost_temporal_expressions_journey = lost_temporal_expressions(temporal_expressions_clinical, temporal_expressions_journey)
                    """
                    # Append the results for this row to the results list
                    results.append({
                        'extracted_clinical_ner': json.dumps(list(clinical_ner), ensure_ascii=False),
                        'extracted_admission_ner': json.dumps(list(admission_ner), ensure_ascii=False),
                        'extracted_discharge_ner': json.dumps(list(discharge_ner), ensure_ascii=False),
                        'extracted_journey_ner': json.dumps(list(journey_ner), ensure_ascii=False),

                        'admission_ner1_similarity': ner_similarity_admission['strict']['ner1_in_ner2'],
                        'admission_ner1_similarity_relaxed': ner_similarity_admission['relaxed']['ner1_in_ner2'],

                        'discharge_ner1_similarity': ner_similarity_discharge['strict']['ner1_in_ner2'],
                        'discharge_ner1_similarity_relaxed': ner_similarity_discharge['relaxed']['ner1_in_ner2'],

                        'full_journey_ner1_similarity': ner_similarity_journey['strict']['ner1_in_ner2'],
                        'full_journey_ner1_similarity_relaxed': ner_similarity_journey['relaxed']['ner1_in_ner2'],

                        'admission_ner2_similarity': ner_similarity_admission['strict']['ner2_in_ner1'],
                        'discharge_ner2_similarity': ner_similarity_discharge['strict']['ner2_in_ner1'],
                        'full_journey_ner2_similarity': ner_similarity_journey['strict']['ner2_in_ner1'],

                        'admission_strict_lost_by_class': ner_similarity_admission['strict']['lost_by_class'],
                        'admission_relaxed_lost_by_class': ner_similarity_admission['relaxed']['lost_by_class'],
                        'admission_Lost_in_strict': ner_similarity_admission['lost_types']['Lost_in_strict'],
                        'admission_recovery_by_relaxed': ner_similarity_admission['lost_types']['Recovered_by_relaxed'],
                        'admission_Lost_in_both': ner_similarity_admission['lost_types']['Lost_in_both'],

                        'discharge_strict_lost_by_class': ner_similarity_discharge['strict']['lost_by_class'],
                        'discharge_relaxed_lost_by_class': ner_similarity_discharge['relaxed']['lost_by_class'],
                        'discharge_Lost_in_strict': ner_similarity_discharge['lost_types']['Lost_in_strict'],
                        'discharge_recovery_by_relaxed': ner_similarity_discharge['lost_types']['Recovered_by_relaxed'],
                        'discharge_Lost_in_both': ner_similarity_discharge['lost_types']['Lost_in_both'],

                        'full_journey_strict_lost_by_class': ner_similarity_journey['strict']['lost_by_class'],
                        'full_journey_relaxed_lost_by_class': ner_similarity_journey['relaxed']['lost_by_class'],
                        'full_journey_Lost_in_strict': ner_similarity_journey['lost_types']['Lost_in_strict'],
                        'full_journey_recovery_by_relaxed': ner_similarity_journey['lost_types']['Recovered_by_relaxed'],
                        'full_journey_Lost_in_both': ner_similarity_journey['lost_types']['Lost_in_both'],

                        'admission_strict_lost_rate_by_class': ner_similarity_admission['strict']['lost_rate_by_class'],
                        'admission_relaxed_lost_rate_by_class': ner_similarity_admission['relaxed']['lost_rate_by_class'],

                        'discharge_strict_lost_rate_by_class': ner_similarity_discharge['strict']['lost_rate_by_class'],
                        'discharge_relaxed_lost_rate_by_class': ner_similarity_discharge['relaxed']['lost_rate_by_class'],

                        'full_journey_strict_lost_rate_by_class': ner_similarity_journey['strict']['lost_rate_by_class'],
                        'full_journey_relaxed_lost_rate_by_class': ner_similarity_journey['relaxed']['lost_rate_by_class'],

#                        'clinical_temporal_expressions': list(temporal_expressions_clinical),
#                        'admission_temporal_expressions': list(temporal_expressions_admission),
#                        'discharge_temporal_expressions': list(temporal_expressions_discharge),
#                        'full_journey_temporal_expressions': list(temporal_expressions_journey),
#
#                        'lost_temporal_expressions_admission': list(lost_temporal_expressions_admission),
#                        'lost_temporal_expressions_discharge': list(lost_temporal_expressions_discharge),
#                        'lost_temporal_expressions_journey': list(lost_temporal_expressions_journey),

                        'bert_score_admission': bert_score_admission,
                        'bert_score_discharge': bert_score_discharge,
                        'bert_score_full_journey': bert_score_journey,

                        'bleu_score_admission': bleu_score_admission,
                        'bleu_score_discharge': bleu_score_discharge,
                        'bleu_score_full_journey': bleu_score_journey
                    })

                except KeyError as e:
                    print(f"Warning: Column {e} not found in DataFrame for row {index}. Skipping.")
                    results.append({
                    'admission_ner1_similarity': np.nan,
                    'discharge_ner1_similarity': np.nan,
                    'full_journey_ner1_similarity': np.nan,

                    'admission_ner1_similarity_relaxed': np.nan,
                    'discharge_ner1_similarity_relaxed': np.nan,
                    'full_journey_ner1_similarity_relaxed': np.nan,

                    'admission_ner2_similarity': np.nan,
                    'discharge_ner2_similarity': np.nan,
                    'full_journey_ner2_similarity': np.nan,

                    'admission_strict_lost_by_class': None,
                    'discharge_strict_lost_by_class': None,
                    'full_journey_strict_lost_by_class': None,

                    'admission_strict_lost_rate_by_class': None,
                    'discharge_strict_lost_rate_by_class': None,
                    'full_journey_strict_lost_rate_by_class': None,

                    'admission_relaxed_lost_by_class': None,
                    'discharge_relaxed_lost_by_class': None,
                    'full_journey_relaxed_lost_by_class': None,

                    'admission_relaxed_lost_rate_by_class': None,
                    'discharge_relaxed_lost_rate_by_class': None,
                    'full_journey_relaxed_lost_rate_by_class': None,

                    'admission_only_strict_lost': None,
                    'admission_only_relaxed_lost': None,
                    'admission_recovery_by_relaxed': None,
                    
                    'discharge_only_strict_lost': None,
                    'discharge_only_relaxed_lost': None,
                    'discharge_recovery_by_relaxed': None,
                    
                    'full_journey_only_strict_lost': None,
                    'full_journey_only_relaxed_lost': None,
                    'full_journey_recovery_by_relaxed': None,
                    
                    'clinical_temporal_expressions': None,
                    'admission_temporal_expressions': None,
                    'discharge_temporal_expressions': None,
                    'full_journey_temporal_expressions': None,

                    'bert_score_admission': None,
                    'bert_score_discharge': None,
                    'bert_score_full_journey': None,

                    'bleu_score_admission': None,
                    'bleu_score_discharge': None,
                    'bleu_score_full_journey': None
                    })
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    results.append({
                    'admission_ner1_similarity': np.nan,
                    'discharge_ner1_similarity': np.nan,
                    'full_journey_ner1_similarity': np.nan,

                    'admission_ner1_similarity_relaxed': np.nan,
                    'discharge_ner1_similarity_relaxed': np.nan,
                    'full_journey_ner1_similarity_relaxed': np.nan,

                    'admission_ner2_similarity': np.nan,
                    'discharge_ner2_similarity': np.nan,
                    'full_journey_ner2_similarity': np.nan,

                    'admission_strict_lost_by_class': None,
                    'discharge_strict_lost_by_class': None,
                    'full_journey_strict_lost_by_class': None,

                    'admission_strict_lost_rate_by_class': None,
                    'discharge_strict_lost_rate_by_class': None,
                    'full_journey_strict_lost_rate_by_class': None,

                    'admission_relaxed_lost_by_class': None,
                    'discharge_relaxed_lost_by_class': None,
                    'full_journey_relaxed_lost_by_class': None,

                    'admission_relaxed_lost_rate_by_class': None,
                    'discharge_relaxed_lost_rate_by_class': None,
                    'full_journey_relaxed_lost_rate_by_class': None,

                    'admission_only_strict_lost': None,
                    'admission_only_relaxed_lost': None,
                    'admission_recovery_by_relaxed': None,
                    
                    'discharge_only_strict_lost': None,
                    'discharge_only_relaxed_lost': None,
                    'discharge_recovery_by_relaxed': None,
                    
                    'full_journey_only_strict_lost': None,
                    'full_journey_only_relaxed_lost': None,
                    'full_journey_recovery_by_relaxed': None,
                    
                    'clinical_temporal_expressions': None,
                    'admission_temporal_expressions': None,
                    'discharge_temporal_expressions': None,
                    'full_journey_temporal_expressions': None,

                    'bert_score_admission': None,
                    'bert_score_discharge': None,
                    'bert_score_full_journey': None,

                    'bleu_score_admission': None,
                    'bleu_score_discharge': None,
                    'bleu_score_full_journey': None
                    })
        finally:
            if temporal_executor is not None:
                temporal_executor.shutdown(wait=True)

        # Append evaluation results to the original data
        gen_data = pd.concat([gen_data, pd.DataFrame(results)], axis=1)
        # Save the evaluated data to a CSV file
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(BASE_DIR, "../output/" + config["MODEL_ID"])
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        gen_data.to_csv(output_path + "/" + file_name +  "_ner_bert_bleu_score_evaluation.csv", index=False)
        print(f"Evaluation results saved to {output_path}/{file_name}_ner_bert_bleu_score_evaluation.csv")
        print("Scoring Processing done..\n")

    # Perform clustering if enabled in the configuration
    if config["CLUSTERING"].lower() == "yes":
        print("\nClustering processing...\n")

        # Replace NaN values with empty strings to avoid issues during vectorization
        gen_data['syn_full_journey'] = gen_data['syn_full_journey'].fillna("")

        # Create a TfidfVectorizer object for text vectorization
        tfidf = TfidfVectorizer()

        # Fit the vectorizer to the 'syn_full_journey' column
        tfidf.fit(gen_data['syn_full_journey'])

        # Get the feature names (keywords) from the vectorizer
        keywords = tfidf.get_feature_names_out()

        # Add the top keywords to the DataFrame
        gen_data['processed_keywords'] = [
            ' '.join(keywords[index]) 
            for index in tfidf.transform(gen_data['syn_full_journey']).toarray().argsort()[:, -5:][:, ::-1]
        ]

        # Vectorize the processed keywords using TF-IDF
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
        tfidf_matrix = vectorizer.fit_transform(gen_data['processed_keywords'])

        # Create clusters using the K-means algorithm
        num_clusters = int(config["N_CLUSTER"])  # Define the number of clusters
        kmeans = KMeans(n_clusters=num_clusters)

        gen_data['cluster'] = kmeans.fit_predict(tfidf_matrix)

        # Save the results with clustering to a CSV file
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(BASE_DIR, "../output/" + config["MODEL_ID"])
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        gen_data.to_csv(output_path + "/" + file_name + "_cluster_ner_bert_bleu_score_evaluation.csv", index=False)
        print(f"Clustering results saved to {output_path}/{file_name}_cluster_ner_bert_bleu_score_evaluation.csv")
        print("Clustering Processing done..\n")


    if config["PT_CLASSIFYING"].lower() == "yes":
        # Initialize the pipeline for text classification (Portuguese language model)
        pipe = pipeline("text-classification", model="liaad/LVI_albertina-900m-portuguese-ptpt-encoder", device=device(config))

        # Define constants
        MAX_TOKENS = 722 # Maximum token length for the model

        # Function to chunk sentences exceeding max tokens
        def chunkSentence(text):
            """
            Splits text into smaller chunks if it exceeds the maximum token length.
            """
            words = text.split()
            return [' '.join(words[i:i + MAX_TOKENS]) for i in range(0, len(words), MAX_TOKENS)]

        # Function to clean processed lines (placeholder, implement as needed)
        def cleaningProcess(lines):
            """
            Clean the processed lines by stripping extra whitespace and removing empty lines.
            """
            return [line.strip() for line in lines if line.strip()]

        # Function to identify variant from a single text input
        def identifyVariantFromText(text):
            """
            Identifies the language variant from a single text input.
            """
            processed_lines = []

            # Split and chunk if necessary
            if len(text.split()) > MAX_TOKENS:
                sentences = chunkSentence(text)
                processed_lines.extend(sentences)
            else:
                processed_lines.append(text)

            # Clean the processed lines
            processed_lines = cleaningProcess(processed_lines)

            # Get predictions from the classifier
            variant_result = pipe(processed_lines)

            # Create DataFrame from results
            res_df = pd.DataFrame.from_dict(variant_result)

            # Adjust scores for PT-BR label
            res_df.loc[res_df["label"] == "PT-BR", "score"] *= -1

            # Calculate overall result
            ret_result = {
                "score": res_df["score"].sum(),
                "label": res_df["label"].value_counts().idxmax()
            }

            return ret_result

  
        # Apply the variant identification function to the relevant columns
        admission_results = gen_data["syn_admission_report"].apply(identifyVariantFromText)
        discharge_results = gen_data["syn_discharge_report"].apply(identifyVariantFromText)
        full_journey_results = gen_data["syn_full_journey"].apply(identifyVariantFromText)
        

        # Extract scores and labels into separate columns
        print("Finding variant for admission report")
        gen_data["admission_variant_score"] = admission_results.apply(lambda x: x["score"])
        gen_data["admission_variant_label"] = admission_results.apply(lambda x: x["label"])

        print("Finding variant for discharge report")
        gen_data["discharge_report_variant_score"] = discharge_results.apply(lambda x: x["score"])
        gen_data["discharge_report_variant_label"] = discharge_results.apply(lambda x: x["label"])

        print("Finding variant for full journey")
        gen_data["full_journey_variant_score"] = full_journey_results.apply(lambda x: x["score"])
        gen_data["full_journey_variant_label"] = full_journey_results.apply(lambda x: x["label"])

        # Save the results with PT classification to a new CSV file
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(BASE_DIR, "../output/" + config["MODEL_ID"])        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        gen_data.to_csv(output_path + "/" + file_name + "_PT_cluster_ner_bert_bleu_score_evaluation.csv", index=False)
        print(f"PT classification results saved to {output_path}/{file_name}_PT_cluster_ner_bert_bleu_score_evaluation.csv")
        print("\nPT Classifying Processing done..\n")
