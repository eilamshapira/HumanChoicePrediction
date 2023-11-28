import itertools
# import networkx
import pydot
from tqdm import tqdm
from IPython.display import Image, display

N_CONDITIONS = 4
N_ACTIONS = 4
n_situations = 2 ** N_CONDITIONS
n_features = n_situations * N_ACTIONS

split_list = {
    "hotel was chosen in last round": "len(previous_rounds) == 0 or previous_rounds[-1][USER_DECISION] == True",
    "user earn more than bot": "user_score(previous_rounds) >= bot_score(previous_rounds)",
    "hotel score >= 8": "reviews.mean() >= 8",
    "hotel score in the last round >= 8": "len(previous_rounds) == 0 or previous_rounds[-1][REVIEWS].mean() >= 8"}
base_strategy_list = {"max": "reviews.max()",
                      "min": "reviews.min()",
                      "mean": "play_mean(reviews)",
                      "median": "play_median(reviews)"}


def is_action_match_situation(strategy_text, situation):
    splits = list(split_list.keys())
    for i in range(len(splits)):
        strategy_text = strategy_text.replace(splits[i], "True" if situation[i] == "1" else "False")
    for _ in range(2):  # deep
        for a_1 in base_strategy_list.keys():
            for a_2 in base_strategy_list.keys():
                strategy_text = strategy_text.replace(f"Is True? if True, play {a_1}. else, play {a_2}",
                                                      f"play {a_1}")
                strategy_text = strategy_text.replace(f"Is False? if True, play {a_1}. else, play {a_2}",
                                                      f"play {a_2}")
        for char in "[]":
            strategy_text = strategy_text.replace(char, "")
    assert len(strategy_text) < 12
    if "max" in strategy_text:
        return 1
    if "min" in strategy_text:
        return -1
    return 0


features_names = []


def get_vector_for_strategy_text(strategy_text):
    row = []
    for situation in range(n_situations):
        if len(features_names) < n_features:
            features_names.append(f"{situation:04b}")
        assert 4 == N_CONDITIONS, "change situation_bits!"
        situation_bits = f"{situation:04b}"
        cell = is_action_match_situation(strategy_text, situation_bits)
        row.append(cell)
    return tuple(row)


def view_pydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)


def main(most_differance_strategies=3):
    consts = {"REVIEWS": 0,
              "BOT_ACTION": 1,
              "USER_DECISION": 2}
    functions = {"user_score":
                 """def user_score(previous_rounds):
                        return sum([(r[REVIEWS].mean()-8)*r[USER_DECISION] for r in previous_rounds])""",
                 "bot_score":
                     """def bot_score(previous_rounds):
                            return sum([r[USER_DECISION] for r in previous_rounds])""",
                 "play_mean": """def play_mean(reviews):
        tmp = reviews - reviews.mean()
        tmp = abs(tmp)
        return reviews[np.argmin(tmp)]""",
                 "play_median": """def play_median(reviews):
        return sorted(reviews)[3]"""}

    strategies = dict()
    c = 0
    for split_a, split_b1, split_b2 in itertools.permutations(split_list.keys(), 3):
        if most_differance_strategies == 2:
            split_b2 = split_b1
        split_a_code = split_list[split_a]
        split_b1_code = split_list[split_b1]
        split_b2_code = split_list[split_b2]
        for strategy_1, strategy_1_code in base_strategy_list.items():
            for strategy_2, strategy_2_code in base_strategy_list.items():
                for strategy_3, strategy_3_code in base_strategy_list.items():
                    for strategy_4, strategy_4_code in base_strategy_list.items():
                        splits_txt = [split_a, split_b1, split_b2]
                        strategies_txt = [strategy_1, strategy_2, strategy_3, strategy_4]
                        if strategy_1 == strategy_2:
                            split_b1_text = f"play {strategy_1}"
                            split_b1_codeline = [(f"return {strategy_1_code}", 0)]
                        else:
                            split_b1_text = f"Is {split_b1}? if True, play {strategy_1}. else, play {strategy_2}"
                            split_b1_codeline = [(f"if {split_b1_code}:", 0),
                                                 (f"return {strategy_1_code}", 1),
                                                 (f"else:", 0),
                                                 (f"return {strategy_2_code}", 1)]
                        if strategy_3 == strategy_4:
                            split_b2_text = f"play {strategy_3}"
                            split_b2_codeline = [(f"return {strategy_3_code}", 0)]
                        else:
                            split_b2_text = f"Is {split_b2}? if True, play {strategy_3}. else, play {strategy_4}"
                            split_b2_codeline = [(f"if {split_b2_code}:", 0),
                                                 (f"return {strategy_3_code}", 1),
                                                 (f"else:", 0),
                                                 (f"return {strategy_4_code}", 1)]
                        if split_b1_text == split_b2_text:
                            text = split_b1_text
                            code = split_b1_codeline
                        else:
                            text = f"Is {split_a}? if True, [{split_b1_text}]. else, [{split_b2_text}]"
                            code = [(f"if {split_a_code}:", 0),
                                    *[(codeline, il + 1) for codeline, il in split_b1_codeline],
                                    (f"else:", 0),
                                    *[(codeline, il + 1) for codeline, il in split_b2_codeline]]
                        text = f"{text}"
                        vector = get_vector_for_strategy_text(text)
                        if vector not in strategies.keys():
                            strategies[vector] = (text, code, splits_txt, strategies_txt)
                        else:
                            if len(strategies[vector][0]) > len(text):
                                strategies[vector] = (text, code, splits_txt, strategies_txt)

    strategies = sorted([(text, vector, code, splits_txt, strategies_txt) for vector, (text, code, splits_txt, strategies_txt) in strategies.items()], key=lambda x: len(x[0]),
                        reverse=False)
    strategies = [(f"#{i} {text}", vector, code, splits_txt, strategies_txt) for i, (text, vector, code, splits_txt, strategies_txt) in enumerate(strategies)]

    def print_headline(name):
        print()
        print("#" * 32)
        print("#", name.upper())
        print("#" * 32)
        print()


    def print_with_indentation(text, indentation=0):
        print(" " * (indentation * 4 - 1), text)


    def strategies_code(strategies, consts, functions, print_max=10e7):
        print("import numpy as np")
        print_headline("consts")
        for const_name, const_value in consts.items():
            print(f"{const_name} = {const_value}")

        print_headline("functions")
        for func_name, func_code in functions.items():
            print(func_code)
            print()

        print_headline("strategeies codes")
        for strategy, vector, strategy_codelist, _, _ in strategies:
            i = strategy.split(" ")[0][1:]
            print(f"def strategy_{i}(reviews, previous_rounds):")
            print_with_indentation(f"\"\"\"{strategy}", 1)
            print_with_indentation(f"{vector}\"\"\"", 1)
            for line in strategy_codelist:
                print_with_indentation(line[0], line[1] + 1)
            print()
            print_max -= 1
            if print_max == 0:
                break

    def text_to_splits(text):
        text = [l for l in text if l in ["?", "[", "]"]]
        text = "".join(text)
        text = text.replace("[?]", "1")
        text = text.replace("?", "1")
        text = text.replace("[]", "0")
        while len(text) < 3:
            text += "0"
        return text


    def graph_from_text(org_text, C_labels=None, S_labels=None):
        sid = org_text.split(" ")[0][1:]
        splits = text_to_splits(org_text)  # 111 all conds, 000 no conds, 100 just C1, 101 C1+C2B, 110 C1+C2A
        dot_string = """
        digraph my_graph {
        labelloc="t"
        """
        dot_string += f"label=\"Strategy #{sid}\""
        if splits[0] == "0":
            dot_string += f"""S1 [label="play {S_labels[0]}"];"""
        else:
            dot_string += f"C1 [shape=box, label=\"{C_labels[0]}?\"];"
            if splits[1] != "0":
                dot_string += f"""
            C2A [shape=box, label=\"{C_labels[1]}?\"];
            S1 [label="play {S_labels[0]}"];
            S2 [label="play {S_labels[1]}"];
            C1 -> C2A [xlabel="True"];
            C2A -> S1 [xlabel="True"];
            C2A -> S2 [label="False"];
            """
            else:
                dot_string += f"""
            C1 -> S1 [xlabel="True"];
            S1 [label="play {S_labels[0]}"];"""

            if splits[2] != "0":
                dot_string += f"""
            C2B [shape=box, label=\"{C_labels[2]}?\"];
            S3 [label="play {S_labels[2]}"];
            S4 [label="play {S_labels[3]}"];
            C1 -> C2B [label="False"];
            C2B -> S3 [xlabel="True"];
            C2B -> S4 [label="False"];
            """
            else:
                dot_string += f"""
            C1 -> S3 [label="False"];
            S3 [label="play {S_labels[2]}"];
            """
        dot_string += "}"

        dot_string = dot_string.replace("play min", "play worst").\
            replace("play max", "play best").\
            replace("hotel score >= 8?", "Is the current hotel good?").\
            replace("hotel score in the last round >= 8?", "Was the hotel in the\nprevious round good?").\
            replace("hotel was chosen in last round?", "Did the DM choose to go to\nthe hotel in the previous round?").\
            replace("user earn more than bot?", "Has the decision maker earned more points than the\nnumber of times he chose to go to the hotels?")
        graphs = pydot.graph_from_dot_data(dot_string)
        graph = graphs[0]
        graph.write_png(f"strategies_graphs/{sid}.png")
        graph.write_pdf(f"strategies_graphs_pdf/{sid}.pdf")
        # view_pydot(graph)

    strategies_code(strategies, consts, functions)
    # create graphs
    for strategy, _, strategy_codelist, splits_txt, strategies_txt in tqdm(strategies):
         graph_from_text(strategy, C_labels=splits_txt, S_labels=strategies_txt)

    #pd.DataFrame([d[1] for d in strategies]).to_csv("bot_vectors.csv", header=False, index=False)


if __name__ == "__main__":
    main(most_differance_strategies=3)
