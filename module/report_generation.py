from module.journey_configuer import admission_report_generation
from module.journey_configuer import discharge_report_generation
from module.journey_configuer import patients_full_journey
from module.journey_configuer import load_pipeline

def generator(config):
    model = load_pipeline(config=config)

    if config["GENERATED_REPORT_TYPE"]=="admission":
        admission_report_generation(model, config)
    else:
        admission_report_generation(model, config)
        print("Admission Done")
        discharge_report_generation(model, config)
        print("Discharge Done")
        patients_full_journey(model, config)
        print("Full journey Done")