"""Assignment 1, Track A - run every task end to end.

Each task also runs on its own (e.g. `python task1_3_evaluation.py`), but this is
the entry point that produces the full set of figures and tables for the report.
"""
import warnings

import config
import task1_1_preprocessing as t11
import task1_2_supervised as t12
import task1_3_evaluation as t13
import task1_4_anomaly as t14
import task1_5_drift as t15
import task1_6_explainability as t16
import task1_7_recommendation as t17

warnings.filterwarnings("ignore")   # sklearn convergence / deprecation noise only


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    data = t11.prepare_data()
    supervised = t12.run(data, top_k=3)   # top 3 so the ROC/PR plot has three curves
    ranked = supervised["ranked"]
    best_name = ranked[0][0]

    evaluation = t13.run(data, ranked)
    anomaly = t14.run(data, best_name)
    drift = t15.run(data, best_name)
    explain = t16.run(data, ranked, anomaly["scores"])

    t17.run({"supervised": supervised, "evaluation": evaluation, "anomaly": anomaly,
             "drift": drift, "explain": explain})

    print(f"\nDone. Figures -> {config.FIG_DIR.name}/, tables -> {config.TABLE_DIR.name}/")


if __name__ == "__main__":
    main()
