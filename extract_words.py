import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

from sampling import split_data
from smt203util import *
from text_preprocess import text_preprocess

nltk.download("stopwords")
stop = stopwords.words("english")


def clean_text_round2(text):
    return " ".join([word for word in text.split() if word not in (stop)])


def count_words_from_dataframe(df):
    result_dict = {}
    for index, row in df.iterrows():
        text = row["clean_text"]
        tokens = text.split()
        for i in range(0, len(tokens)):
            token = tokens[i]
            try:
                result_dict[token] += 1
            except KeyError:
                result_dict[token] = 1
    return result_dict


def split_dataframe_by_percentages(df, percentages):

    if not abs(sum(percentages) - 1.0) < 1e-6:
        raise ValueError("Percentages must sum to 1.")

    n = len(df)
    indices = [int(n * sum(percentages[: i + 1])) for i in range(len(percentages))]
    split_dfs = []

    start_idx = 0
    for end_idx in indices:
        split_dfs.append(df.iloc[start_idx:end_idx])
        start_idx = end_idx

    return split_dfs


def process_and_save_word_counts(df, city, single, percentages, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    df_sorted = df.sort_values(by="duration")
    df_sorted.reset_index(drop=True, inplace=True)
    split_dfs = split_dataframe_by_percentages(df_sorted, percentages)
    group_names = [f"group_{i}" for i in range(len(split_dfs))]

    for group_name, group_data in zip(group_names, split_dfs):
        print(
            f"{city}_{single}_{group_name}: {group_data['duration'].values[0]} to {group_data['duration'].values[-1]} Days"
        )
        result = count_words_from_dataframe(group_data)
        sorted_dic = (
            (k, result[k]) for k in sorted(result, key=result.get, reverse=True)
        )

        file_path = os.path.join(output_dir, f"{city}_{single}_{group_name}_counts.csv")
        with open(file_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp, delimiter=",")
            writer.writerows(sorted_dic)


def find_discriminative_words(
    top_words_df,
    threshold_i=5,
    threshold_j=5,
    num_i=100,
    num_j=100,
    mypath=".",
    percent=0.25,
):
    """write discriminative words to each file separately"""
    output_dir = os.path.join(mypath, "word_counts", str(percent))
    os.makedirs(output_dir, exist_ok=True)

    counts_i_name = top_words_df.columns[2]
    counts_j_name = top_words_df.columns[3]

    # tmp = top_words_df[top_words_df[counts_i_name] >= threshold_i].query('log_odds_z_score > 0').head(num_i)
    tmp = top_words_df[top_words_df[counts_i_name] >= threshold_i].head(num_i)
    with open(
        f"{mypath}/word_counts/{percent}/{counts_i_name}_zscore.csv", "w"
    ) as output:
        for _, row in tmp.iterrows():
            output.write(
                ",".join(["_".join(row["word"].split()), str(row["log_odds_z_score"])])
                + "\n"
            )

    # tmp = top_words_df[top_words_df[counts_j_name] >= threshold_j].query('log_odds_z_score < 0').iloc[::-1].head(num_j)
    tmp = (
        top_words_df[top_words_df[counts_j_name] >= threshold_j].iloc[::-1].head(num_j)
    )
    with open(
        f"{mypath}/word_counts/{percent}/{counts_j_name}_zscore.csv", "w"
    ) as output:
        for _, row in tmp.iterrows():
            output.write(
                ",".join(
                    ["_".join(row["word"].split()), str(-1.0 * row["log_odds_z_score"])]
                )
                + "\n"
            )


def extract_words(
    # n_samples: int = 1000,
    test_size: int = 0.2,
) -> None:

    target_url = "https://raw.githubusercontent.com/anjisun221/css_codes/main/ay22t1/Lab03_text_analysis/1gram_englishall_count.csv"
    global_counts = read_word_count_file_online(target_url)
    print("The number of unigrams=", len(global_counts))

    stopwords_list = get_stopwords()
    global_counts = {
        k: v
        for k, v in global_counts.items()
        if (k not in stopwords_list) and (len(k) > 2)
    }
    print("After excluding stop words: ", len(global_counts))

    (
        X_train,
        X_test,
        y_train,
        y_test,
        desc_train,
        desc_test,
        zpid_train,
        zpid_test,
        df_words,
    ) = split_data()

    df_words = text_preprocess(df_words)
    df_words["clean_text"] = df_words["description"].apply(clean_text_round2)

    cities = ["CH", "NY", "LA"]
    single_types = [0, 1]
    output_dir = "./dataset/word_counts"

    for a in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        percentages = [a, 1 - 2 * a, a]

        for city in cities:
            for single in single_types:
                city_single_df = df_words[
                    (df_words["city"] == city) & (df_words["single"] == single)
                ]
                process_and_save_word_counts(
                    city_single_df, city, single, percentages, output_dir
                )

        for city in cities:
            for single in single_types:
                counts_i_name = f"{city}_{single}_group_0"  # lower
                counts_i = read_word_count_file(
                    f"./dataset/word_counts/{counts_i_name}_counts.csv"
                )
                counts_i_dict = {
                    k: v for k, v in counts_i.items() if k in global_counts
                }

                counts_j_name = f"{city}_{single}_group_2"  # upper
                counts_j = read_word_count_file(
                    f"./dataset/word_counts/{counts_j_name}_counts.csv"
                )
                counts_j_dict = {
                    k: v for k, v in counts_j.items() if k in global_counts
                }

                top_words_df = calculate_log_odds_idp(
                    global_counts,
                    counts_i_name,
                    counts_i_dict,
                    counts_j_name,
                    counts_j_dict,
                )
                find_discriminative_words(
                    top_words_df,
                    threshold_i=2,
                    threshold_j=2,
                    num_i=50,
                    num_j=50,
                    mypath="./dataset",
                    percent=a,
                )
