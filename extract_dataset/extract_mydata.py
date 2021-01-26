import re
import json
import random
import numpy as np
from ltp_labler import LTP_Labler


sentf = "neural_oie.sent"
triplef = "neural_oie.triple"

label_map = {"A0-B": "B-E", "A0-I": "I-E", "A1-B": "B-A", "A1-I": "I-A", "P-B": "B-P", "P-I": "I-P", "O": "O"}
pos_map = {"PAD": 0}
ner_map = {"PAD": 0}
dp_map = {"PAD": 0}


def clean(text):
    return re.sub("['#`\-]", "", text)


def get_tag(words, postags, arcs, nertags):
    '''
    description: 将文本标签转化为数字标签
    words {str-list} 分词列表
    postags {str-list} 词性列表
    arcs {tuple-list} 依存分析列表 (head, relation)
    nertags {str-list} 实体识别标签列表
    return 词性列表{int-list}, 依存邻接矩阵{2D-int-list}, 实体识别列表{int-list} 0,0为外加root节点
    '''
    poss = []
    ners = []
    dp_mat = np.zeros((len(words)+1, len(words)+1))

    i = 0
    for fetch in zip(words, postags, arcs, nertags):
        word, pos, arc, ner = fetch

        if pos not in pos_map:
            pos_map[pos] = len(pos_map)
        if arc[1] not in dp_map:
            dp_map[arc[1]] = len(dp_map)
        if ner not in ner_map:
            ner_map[ner] = len(ner_map)

        poss.append(pos_map[pos])
        ners.append(ner_map[ner])
        dp_mat[i][arc[0]] = dp_map[arc[1]]
        i += 1

    return poss, dp_mat.tolist(), ners


def convert(sent_file, triple_file):
    sf = open(sent_file, "r")
    tf = open(triple_file, "r")
    labeler = LTP_Labler()
    ore_data, ore_dp_data = [], []
    max_length = 0
    ore_count = 0

    # pos, ner, dp
    for sent, triple in zip(sf, tf):
        sent = clean(sent)
        triple = clean(triple)
        triple = triple.strip()
        sub = re.findall("<arg1>(.*)</arg1>", triple)[0].replace(" ", "")
        pred = re.findall("<rel>(.*)</rel>", triple)[0].replace(" ", "")
        obj = re.findall("<arg2>(.*)</arg2>", triple)[0].replace(" ", "")
        words, postags, arcs, nertags = labeler.get_features(sent)
        poss, dp_mat, ners = get_tag(words, postags, arcs, nertags)

        gold, sub_pos, obj_pos, pred_pos = ["O"] * len(words), [], [], []
        for i, w in enumerate(words):
            if sub.startswith(w):
                sub_pos.append(i)
                sub = sub[len(w):]
                sub = sub.strip()
            elif pred.startswith(w):
                pred_pos.append(i)
                pred = pred[len(w):]
                pred = pred.strip()
            elif obj.startswith(w):
                obj_pos.append(i)
                obj = obj[len(w):]
                obj = obj.strip()

        try:
            gold[sub_pos[0]] = "B-E1"
            gold[obj_pos[0]] = "B-E2"
            gold[pred_pos[0]] = "B-R"
        except Exception:
            continue

        for i in sub_pos[1:]:
            gold[i] = "I-E1"
        for i in obj_pos[1:]:
            gold[i] = "I-E2"
        for i in pred_pos[1:]:
            gold[i] = "I-R"

        ore_data.append({"token": words, "pos": poss, "ner": ners, "gold": gold, "e1": sub_pos, "e2": obj_pos, "pred": pred_pos})
        ore_dp_data.append(dp_mat)
        max_length = len(words) if max_length < len(words) else max_length
        ore_count += 1
        if ore_count >= 50000:
            break

    return ore_data, ore_dp_data, max_length


if __name__ == "__main__":
    ore_data, ore_dp_data, ore_count = convert(sentf, triplef)

    # shuffle
    randnum = random.randint(0, 10)
    random.seed(randnum)
    random.shuffle(ore_data)
    random.seed(randnum)
    random.shuffle(ore_dp_data)

    # split and save
    split1 = len(ore_data) // 10 * 8
    split2 = len(ore_data) // 10 + split1
    with open("oie_train_en.json", 'w', encoding='utf-8') as ief:
        json.dump(ore_data[:split1], ief, ensure_ascii=False)
    with open("oie_dev_en.json", 'w', encoding='utf-8') as ief:
        json.dump(ore_data[split1:split2], ief, ensure_ascii=False)
    with open("oie_test_en.json", 'w', encoding='utf-8') as ief:
        json.dump(ore_data[split2:], ief, ensure_ascii=False)

    ore_dp_data = np.array(ore_dp_data)
    np.save("oie_train_matrixs_en.npy", ore_dp_data[:split1])
    np.save("oie_dev_matrixs_en.npy", ore_dp_data[split1:split2])
    np.save("oie_test_matrixs_en.npy", ore_dp_data[split2:])
    print("max length {}".format(ore_count))

    with open("pos_map.json", "w") as pf:
        json.dump(pos_map, pf)
    with open("ner_map.json", "w") as nf:
        json.dump(ner_map, nf)
    with open("dp_map.json", "w") as df:
        json.dump(dp_map, df)
