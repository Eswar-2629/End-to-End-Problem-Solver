"""Run the complete reproducible analysis."""
from src.data_generation import save_synthetic_data
from src.data_processing import process_and_save
from src.eda import run_eda
from src.modeling import train_and_evaluate


def main():
    raw = save_synthetic_data()
    cleaned, cleaning_log = process_and_save()
    eda_results = run_eda(cleaned)
    model_results = train_and_evaluate(cleaned)
    print("Synthetic raw rows:", len(raw))
    print("Cleaned rows:", len(cleaned))
    print("Cleaning decisions:", len(cleaning_log))
    print("Improved test ROC-AUC:", round(model_results["improved_test"]["roc_auc"], 4))
    print("Outputs written under outputs/")


if __name__ == "__main__":
    main()
