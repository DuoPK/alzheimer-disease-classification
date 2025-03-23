from analyze.analyzer import AlzheimerDatasetAnalyzer
from features import range_features, binary_features, categorical_features

selected_features = list(range_features.keys()) + binary_features + list(categorical_features.keys())

if __name__ == "__main__":
    file_path = "../modified_extreme_dataset.csv"
    analyzer = AlzheimerDatasetAnalyzer(file_path, selected_features, dependent_feature="Diagnosis")
    # analyzer.check_and_save_dataset()
    # analyzer.validate_data_ranges()

    analyzer.full_analysis(incorrect_data_to_nan=True, save_img=True, hist_bins=20, path_imgs="with_out_of_range",
                           remove_missing_dependent=True)

    # file_path = "../ready_dataset_to_preprocessing.csv"
    # analyzer = AlzheimerDatasetAnalyzer(file_path, selected_features, dependent_feature="Diagnosis")
    # analyzer.full_analysis(incorrect_data_to_nan=False, save_img=True, hist_bins=None, show_count_on_hist=True,
    #                        path_imgs="with_all_nulls")

    # df = analyzer.df
    # sns.pairplot(df)
    # plt.show()
