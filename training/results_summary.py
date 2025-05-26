import os

import pandas as pd
from matplotlib import pyplot as plt

from training.utils.process_json import process_json_files


def process_json(input_dir="training/results", output_dir="results/summary", summary_table=True):
    """
    Process JSON files in the input directory and save summaries to the output directory.
    """
    process_json_files(input_dir, output_dir)
    print(f"Processed JSON files from {input_dir} and saved summaries to {output_dir}")

    if summary_table:
        for file in os.listdir(output_dir):
            if file.endswith(".csv"):
                file_path = os.path.join(output_dir, file)
                df = pd.read_csv(file_path)

                selected_columns = ["model", "dataset", "final_test_results-f1_score"]
                df = df[selected_columns]
                df.rename(columns={"final_test_results-f1_score": "f1-score"}, inplace=True)

                model_name = df["model"].iloc[0]
                png_file = os.path.join(output_dir, f"{model_name}_summary.png")

                fig, ax = plt.subplots(figsize=(12, len(df) * 0.6))
                ax.axis('tight')
                ax.axis('off')

                table = ax.table(
                    cellText=df.values,
                    colLabels=df.columns,
                    loc='center',
                    cellLoc='center',
                    colColours=["#4CAF50"] * len(df.columns),  # Color for header cells
                )
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.auto_set_column_width(col=list(range(len(df.columns))))

                for (row, col), cell in table.get_celld().items():
                    cell.PAD = 0.2  # Padding
                    if row == 0:  # Headers
                        cell.set_text_props(weight='bold', color='white')
                        cell.set_facecolor('#4CAF50')
                    else:  # Rows
                        cell.set_facecolor('#f9f9f9' if row % 2 == 0 else '#ffffff')

                plt.savefig(png_file, bbox_inches='tight')
                print(f"Tabela dla modelu {model_name} zapisana jako obraz w {png_file}")


def create_summary(input_dir, output_file, time_threshold=1.5, score_threshold=0.01):
    """
    Creates a summary from CSV files in the input directory.

    :param time_threshold: The proportion of time that considers the result to be significantly longer.
    :param score_threshold: The maximum difference in f1-score to add to the comparison.
    """
    # os.makedirs(output_file, exist_ok=True)
    summary_data = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(file_path)

            # Find the best f1-score
            best_row = df.loc[df['final_test_results-f1_score'].idxmax()]
            best_f1_score = best_row['final_test_results-f1_score']
            best_dataset = best_row['dataset']
            best_time = best_row['final_test_results-training_time']
            model = best_row['model']

            # Add the best result to the summary
            summary_data.append({
                "model": model,
                "dataset": best_dataset,
                "f1-score": best_f1_score,
                "training_time": best_time
            })

            # Check other results for comparison
            for _, row in df.iterrows():
                if row['dataset'] != best_dataset:
                    f1_score = row['final_test_results-f1_score']
                    training_time = row['final_test_results-training_time']

                    # Add the result if the time is significantly longer and the result is not much worse
                    if training_time > best_time * time_threshold and best_f1_score - f1_score <= score_threshold:
                        summary_data.append({
                            "model": model,
                            "dataset": row['dataset'],
                            "f1-score": f1_score,
                            "training_time": training_time
                        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"Podsumowanie zapisane do pliku: {output_file}")


def main():
    process_json("training/results", output_dir="results/summary/first_15_trials", summary_table=False)
    process_json("training/results/20250526", output_dir="results/summary", summary_table=False)

    create_summary("results/summary", "results/summary.csv", time_threshold=1.5, score_threshold=0.01)


if __name__ == "__main__":
    main()
