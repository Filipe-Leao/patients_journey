# Importing necessary libraries
import pandas as pd  # For handling and manipulating structured data
import os  # For operating system-related functionalities (e.g., file path handling)
import torch  # For deep learning models and computations
import re
import json
from collections import defaultdict
from py_heideltime import heideltime
import concurrent.futures
#from threading import Lock

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
    output_path = f"{BASE_DIR}/../final_csv"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_path = os.path.join(BASE_DIR, config["CASE_REPORT_CSV_PATH"][:-4]+"_new.csv")
    file_name = os.path.splitext(os.path.basename(config["CASE_REPORT_CSV_PATH"]))[0]
    file_path = f"{output_path}/{file_name}_ner_bert_bleu_score_evaluation_time.csv"
    gen_data=pd.read_csv(
        file_path, 
        low_memory=False, 
        dtype={
            'clinical_temporal_expressions': str,
            'admission_temporal_expressions': str,
            'discharge_temporal_expressions': str,
            'full_journey_temporal_expressions': str,
            'lost_temporal_expressions_admission': str,
            'lost_temporal_expressions_discharge': str,
            'lost_temporal_expressions_journey': str,
        })
    
    torch.cuda.empty_cache()  # Clear GPU cache to free up memory
    
    print(f"Loaded data from {file_path}. Total rows: {len(gen_data)}")

    # Handling a subset of data based on the configuration
    if isinstance(config["N_TESTING_ROW"], int):  # Check if it's an integer
        gen_data = gen_data[0:config["N_TESTING_ROW"]]  # Use only the specified number of rows
    elif config["N_TESTING_ROW"] == "all":  # If it's the string "all"
        gen_data  # Use all rows
    else:
        gen_data  # Default case: no filtering

    
    # Initialize the NER pipeline using a pre-trained model
    if config["SCORING"].lower() == "yes":
        print("I AM SCORE")
        temporal_max_workers = int(config.get("TEMPORAL_MAX_WORKERS", 4))
        temporal_max_workers = max(1, temporal_max_workers)

        # Ensure output columns exist and initialize with None
        output_columns = [
            'clinical_temporal_expressions',
            'admission_temporal_expressions',
            'discharge_temporal_expressions',
            'full_journey_temporal_expressions',
            'lost_temporal_expressions_admission',
            'lost_temporal_expressions_discharge',
            'lost_temporal_expressions_journey',
        ]

        for col in output_columns:
            if col not in gen_data.columns:
                gen_data[col] = None
            else:
                try:
                    gen_data[col] = gen_data[col].astype(object)
                except Exception:
                    gen_data[col] = gen_data[col].astype(str)

        # Iterate rows and write results directly into gen_data; skip rows already processed
        try:
            for index, row in gen_data.iterrows():
                if index < 30000:
                    print(f"Skipping row {index}.")
                    continue
                
                if index == 36000:
                    print(f"Skipping rest of dataset.")
                    break
                
                print(f"\nProcessing Patient {index}/{len(gen_data)}")
                

                existing = row.get('clinical_temporal_expressions', None)
                if pd.notna(existing) and str(existing).strip() not in ('', '[]'):
                    print(f"Patient {index} already has temporal expressions. Skipping.")
                    continue

                try:
                    temporal_expressions_clinical = get_temporal_expressions(row[config["CASE_REPORT_COLUMN_NAME"]])
                    temporal_expressions_admission = get_temporal_expressions(row['syn_admission_report'])
                    temporal_expressions_discharge = get_temporal_expressions(row['syn_discharge_report'])
                    temporal_expressions_journey = get_temporal_expressions(row['syn_full_journey'])

                    lost_temporal_expressions_admission = lost_temporal_expressions(temporal_expressions_clinical, temporal_expressions_admission)
                    lost_temporal_expressions_discharge = lost_temporal_expressions(temporal_expressions_clinical, temporal_expressions_discharge)
                    lost_temporal_expressions_journey = lost_temporal_expressions(temporal_expressions_clinical, temporal_expressions_journey)

                    gen_data.at[index, 'clinical_temporal_expressions'] = json.dumps(list(temporal_expressions_clinical), ensure_ascii=False)
                    gen_data.at[index, 'admission_temporal_expressions'] = json.dumps(list(temporal_expressions_admission), ensure_ascii=False)
                    gen_data.at[index, 'discharge_temporal_expressions'] = json.dumps(list(temporal_expressions_discharge), ensure_ascii=False)
                    gen_data.at[index, 'full_journey_temporal_expressions'] = json.dumps(list(temporal_expressions_journey), ensure_ascii=False)

                    gen_data.at[index, 'lost_temporal_expressions_admission'] = json.dumps(list(lost_temporal_expressions_admission), ensure_ascii=False)
                    gen_data.at[index, 'lost_temporal_expressions_discharge'] = json.dumps(list(lost_temporal_expressions_discharge), ensure_ascii=False)
                    gen_data.at[index, 'lost_temporal_expressions_journey'] = json.dumps(list(lost_temporal_expressions_journey), ensure_ascii=False)

                except KeyError as e:
                    print(f"Warning: Column {e} not found in DataFrame for row {index}. Skipping.")
                    gen_data.at[index, 'clinical_temporal_expressions'] = None
                    gen_data.at[index, 'admission_temporal_expressions'] = None
                    gen_data.at[index, 'discharge_temporal_expressions'] = None
                    gen_data.at[index, 'full_journey_temporal_expressions'] = None
                    gen_data.at[index, 'lost_temporal_expressions_admission'] = None
                    gen_data.at[index, 'lost_temporal_expressions_discharge'] = None
                    gen_data.at[index, 'lost_temporal_expressions_journey'] = None
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    gen_data.at[index, 'clinical_temporal_expressions'] = None
                    gen_data.at[index, 'admission_temporal_expressions'] = None
                    gen_data.at[index, 'discharge_temporal_expressions'] = None
                    gen_data.at[index, 'full_journey_temporal_expressions'] = None
                    gen_data.at[index, 'lost_temporal_expressions_admission'] = None
                    gen_data.at[index, 'lost_temporal_expressions_discharge'] = None
                    gen_data.at[index, 'lost_temporal_expressions_journey'] = None
        except KeyboardInterrupt:
            print('\nKeyboardInterrupt detected. Saving partial results...')
            gen_data.to_csv(output_path + "/" + file_name +  "_ner_bert_bleu_score_evaluation_time.csv", index=False)
            print(f"Partial results saved to {output_path}/{file_name}_ner_bert_bleu_score_evaluation_time.csv")
            return
        # Save the evaluated data to a CSV file
        
        gen_data.to_csv(output_path + "/" + file_name +  "_ner_bert_bleu_score_evaluation_time.csv", index=False)
        print(f"Evaluation results saved to {output_path}/{file_name}_ner_bert_bleu_score_evaluation_time.csv")
        print("Scoring Processing done..\n")


if __name__ == "__main__":
    evaluator({
        "GEN_LANGUAGE":"European Portugues *(NOT BRAZILIAN PORTUGUESE)*",
        "CASE_REPORT_CSV_PATH":"../data/PMCPatients_part1.csv",
        "CASE_REPORT_COLUMN_NAME":"patient_translated",
        "NER_MODEL":"portugueseNLP/medialbertina_pt-pt_900m_NER",
        "GENERATED_REPORT_TYPE":"all",
        "SCORING":"yes",
        "PT_CLASSIFYING":"NO",
        "CLUSTERING":"yes",
        "N_CLUSTER":"10",
        "N_TESTING_ROW":"all", 
        "MODEL_ID":"Qwen/Qwen3-14B",
        "GPU":"YES",
        "GPU_NUMBER":1,
        "API_KEY":""
    })
