from abc import abstractmethod


class Instance:
    def __init__(self, instance_id, weight, inputs=None, output=None):
        self.instance_id = instance_id
        self.weight = weight
        self.input = inputs
        self.output = output
        self.labeled_instance = None
        self.unlabeled_instance = None
        self.prediction = None
        self.is_labeled = True

    def set_instance_id(self, inst_id):
        self.instance_id = inst_id

    def get_instance_id(self):
        return self.instance_id

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight

    def set_labeled(self):
        self.is_labeled = True

    def set_unlabeled(self):
        self.is_labeled = False

    def remove_output(self):
        self.output = None

    # def is_labeled(self):
    #     return self.is_labeled

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def duplicate(self):
        pass

    @abstractmethod
    def removeOutput(self):
        pass

    @abstractmethod
    def removePrediction(self):
        pass

    @abstractmethod
    def get_input(self):
        pass

    @abstractmethod
    def get_output(self):
        pass

    @abstractmethod
    def get_prediction(self):
        pass

    @abstractmethod
    def set_prediction(self, *args):
        pass

    @abstractmethod
    def has_output(self):
        pass

    @abstractmethod
    def has_prediction(self):
        pass

    def get_islabeled(self):
        return self.is_labeled

    def get_labeled_instance(self):
        if self.is_labeled:
            return self

    def set_label_instance(self, inst):
        self.labeled_instance = inst

    def get_unlabeled_instance(self):
        pass

    def set_unlabel_instance(self, inst):
        self.unlabeled_instance = inst


class LinearInstance(Instance):
    def __init__(self, instance_id, weight, inputs, output):
        super().__init__(instance_id, weight, inputs, output)
        self.word_seq = None

    def size(self):
        # print('input:', self.input)
        return len(self.input)

    def duplicate(self):
        dup = LinearInstance(self.instance_id, self.weight, self.input, self.output)
        dup.word_seq = self.word_seq
        # print('dup input:', dup.get_input())
        return dup

    def removeOutput(self):
        self.output = None

    def removePrediction(self):
        self.prediction = None

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output

    def get_prediction(self):
        return self.prediction

    def set_prediction(self, prediction):
        self.prediction = prediction

    def has_output(self):
        return self.output is not None

    def has_prediction(self):
        return self.prediction is not None

    def __str__(self):
        return (
            "input:"
            + str(self.input)
            + "\toutput:"
            + str(self.output)
            + " is_labeled:"
            + str(self.is_labeled)
        )


class TagReader:
    # 0 neu, 1 pos, 2 neg
    label2id_map = {"<START>": 0}

    @classmethod
    def read_inst(cls, file, is_labeled, number, opinion_offset):
        insts = []
        # inputs = []
        # outputs = []
        total_p = 0
        original_p = 0
        f = open(file, "r", encoding="utf-8")

        # read AAAI2020 data
        for line in f:
            line = line.strip()
            line = line.split("####")
            inputs = line[0].split()  # sentence
            # t_output = line[1].split()  # target
            o_output = line[2].split()  # opinion
            raw_pairs = eval(line[3])  # triplets

            # prepare tagging sequence
            output = ["O" for _ in range(len(inputs))]
            for i, t in enumerate(o_output):
                t = t.split("=")[1]
                if t != "O":
                    output[i] = t
            output_o = cls.ot2bieos_o(output)
            output = ["O" for _ in range(len(inputs))]
            for i in range(len(inputs)):
                if output_o[i] != "O":
                    output[i] = output_o[i].split("-")[0]

            # re-format original triplets to jet_o tagging format
            new_raw_pairs = []
            for new_pair in raw_pairs:
                opinion_s = new_pair[1][0]
                opinion_e = new_pair[1][-1]
                target_s = new_pair[0][0]
                target_e = new_pair[0][-1]
                # change sentiment to value --> 0 neu, 1 pos, 2 neg
                if new_pair[2] == "NEG":
                    polarity = 2
                elif new_pair[2] == "POS":
                    polarity = 1
                else:
                    polarity = 0
                # check direction and append
                if target_s < opinion_s:
                    dire = 0
                    new_raw_pairs.append(
                        (
                            [opinion_s, opinion_e],
                            polarity,
                            dire,
                            opinion_s - target_e,
                            opinion_s - target_s,
                        )
                    )
                else:
                    dire = 1
                    new_raw_pairs.append(
                        (
                            [opinion_s, opinion_e],
                            polarity,
                            dire,
                            target_s - opinion_s,
                            target_e - opinion_s,
                        )
                    )

            new_raw_pairs.sort(key=lambda x: x[0][0])
            original_p += len(raw_pairs)

            # remove train data that offset (M) larger than setting and nosiy data during training
            if is_labeled:
                new_pairs = []
                opinion_idxs = []
                remove_idxs = []
                for pair in new_raw_pairs:
                    if opinion_offset > pair[-1] >= pair[-2] > 0:
                        new_pairs.append(pair)
                        opinion_idxs.extend(list(range(pair[0][0], pair[0][1] + 1)))
                    else:
                        remove_idxs.extend(list(range(pair[0][0], pair[0][-1] + 1)))
                for idx in remove_idxs:
                    if idx not in opinion_idxs:
                        output[idx] = "O"
            else:
                # keep all original triplets during eval and test for calculating F1 score
                new_pairs = new_raw_pairs
                output = output

            total_p += len(new_pairs)
            output = (output, new_pairs)

            if len(new_pairs) > 0:
                inst = LinearInstance(len(insts) + 1, 1, inputs, output)
                for label in output[0]:
                    if label not in TagReader.label2id_map and is_labeled:
                        output_id = len(TagReader.label2id_map)
                        TagReader.label2id_map[label] = output_id
                if is_labeled:
                    inst.set_labeled()
                else:
                    inst.set_unlabeled()
                insts.append(inst)
                if len(insts) >= number > 0:
                    break
        print("# of original triplets: ", original_p)
        print("# of triplets for current setup: ", total_p)
        return insts

    @staticmethod
    def ot2bieos_o(ts_tag_sequence):
        """
        ot2bieos function for opinion
        """
        n_tags = len(ts_tag_sequence)
        new_ts_sequence = []
        prev_pos = "$$$"
        for i in range(n_tags):
            cur_ts_tag = ts_tag_sequence[i]
            if cur_ts_tag == "O":
                new_ts_sequence.append("O")
                cur_pos = "O"
            else:
                cur_pos = cur_ts_tag
                # cur_pos is T
                if cur_pos != prev_pos:
                    # prev_pos is O and new_cur_pos can only be B or S
                    if i == n_tags - 1:
                        new_ts_sequence.append("s-o")
                    else:
                        next_ts_tag = ts_tag_sequence[i + 1]
                        if next_ts_tag == "O":
                            new_ts_sequence.append("s-o")
                        else:
                            new_ts_sequence.append("b-o")
                else:
                    # prev_pos is T and new_cur_pos can only be I or E
                    if i == n_tags - 1:
                        new_ts_sequence.append("e-o")
                    else:
                        next_ts_tag = ts_tag_sequence[i + 1]
                        if next_ts_tag == "O":
                            new_ts_sequence.append("e-o")
                        else:
                            new_ts_sequence.append("i-o")
            prev_pos = cur_pos
        return new_ts_sequence


class Span:
    def __init__(self, left, right, type):
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return (
            self.left == other.left
            and self.right == other.right
            and self.type == other.type
        )

    def __hash__(self):
        return hash((self.left, self.right, self.type))


class Score:
    @abstractmethod
    def larger_than(self, obj):
        pass

    @abstractmethod
    def update_score(self, obj):
        pass


class FScore(object):
    def __init__(self, precision, recall, fscore):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore

    def __str__(self):
        return "(Precision={:.2f}%, Recall={:.2f}%, FScore={:.2f}%)".format(
            self.precision * 100, self.recall * 100, self.fscore * 100
        )


class Eval:
    @abstractmethod
    def eval(self, insts) -> Score:
        pass


## the input to the evaluation should already have
## have the predictions which is the label.
## iobest tagging scheme
class nereval(Eval):
    def eval(self, insts):

        pp = 0
        total_entity = 0
        total_predict = 0
        opinion_eval = False
        target_eval = False
        baseline_eval = False
        pair_eval = True
        test_pairs = []

        idx = 0
        if baseline_eval:
            with open("baseline_result.txt", "w") as f:
                for inst in insts:
                    prediction = inst.prediction
                    # print('--------', prediction)
                    gold_pair = inst.output[1]
                    # print(gold_pair)
                    predict_span_ts = []
                    p_start = -1
                    for i in range(len(prediction)):
                        if prediction[i].startswith("B"):
                            p_start = i
                        if prediction[i].startswith("E"):
                            p_end = i
                            predict_span_ts.append(
                                [[p_start, p_end], prediction[i][2:]]
                            )
                        if prediction[i].startswith("S"):
                            predict_span_ts.append([[i], prediction[i][2:]])
                    predict_span_os = []
                    p_start = -1
                    for i in range(len(prediction)):
                        if prediction[i].startswith("b"):
                            p_start = i
                        if prediction[i].startswith("e"):
                            p_end = i
                            predict_span_os.append(
                                [[p_start, p_end], prediction[i][2:]]
                            )
                        if prediction[i].startswith("s"):
                            predict_span_os.append([[i], prediction[i][2:]])
                    pairs = []

                    if len(predict_span_ts) > 0:
                        for target in predict_span_ts:
                            t_pos = target[0][0]
                            min_distance = len(prediction)
                            if len(predict_span_os) > 0:
                                for opinion in predict_span_os:
                                    o_pos = opinion[0][0]
                                    if min_distance > abs(t_pos - o_pos):
                                        min_distance = abs(t_pos - o_pos)
                                        pair = (target[0], opinion[0], target[1])
                                pairs.append(pair)

                    new_pairs = []
                    for p in pairs:
                        opinion_idx = list(range(p[1][0], p[1][-1] + 1))
                        if len(opinion_idx) == 1:
                            opinion_idx.append(opinion_idx[0])
                        if p[-1] == "POS":
                            polarity = 1
                        elif p[-1] == "NEG":
                            polarity = 2
                        elif p[-1] == "NEU":
                            polarity = 0
                        direction = 1
                        if p[1][0] > p[0][0]:
                            direction = 0
                        target_idx = (abs(p[1][0] - p[0][-1]), abs(p[1][0] - p[0][0]))
                        if direction == 1:
                            target_idx = (
                                abs(p[1][0] - p[0][0]),
                                abs(p[1][0] - p[0][-1]),
                            )

                        new_pairs.append(
                            (
                                opinion_idx,
                                polarity,
                                direction,
                                target_idx[0],
                                target_idx[1],
                            )
                        )
                    # print('new pairs', new_pairs)
                    total_entity += len(gold_pair)
                    total_predict += len(new_pairs)
                    for pred in new_pairs:
                        for gold in gold_pair:
                            if pred == gold:
                                pp += 1
                    test_pairs.append(new_pairs)
                    idx += 1
                    f.write(str(inst.get_input()) + "\n")
                    f.write(str(inst.get_output()) + "\n")
                    f.write(str(inst.get_prediction()) + str(new_pairs) + "\n")
                    f.write("\n")
            f.close()
        # print(test_pairs)
        if not baseline_eval:
            for inst in insts:
                output = inst.output[0]
                prediction = inst.prediction
                # print(inst)
                # print('----',output)
                # print('-------', prediction)
                if pair_eval:
                    output = inst.output[1]
                    prediction = inst.prediction[1]
                    total_entity += len(output)
                    total_predict += len(prediction)
                    for pred in prediction:
                        for gold in output:
                            if pred == gold:
                                pp += 1

                # convert to span
                output_spans = set()
                if target_eval:
                    start = -1
                    for i in range(len(output)):
                        if output[i].startswith("B"):
                            start = i
                        if output[i].startswith("E"):
                            end = i
                            output_spans.add(Span(start, end, output[i][2:]))
                        if output[i].startswith("S"):
                            output_spans.add(Span(i, i, output[i][2:]))
                if opinion_eval:
                    start = -1
                    for i in range(len(output)):
                        if output[i].startswith("b"):
                            start = i
                        if output[i].startswith("e"):
                            end = i
                            output_spans.add(Span(start, end, output[i][2:]))
                        if output[i].startswith("s"):
                            output_spans.add(Span(i, i, output[i][2:]))

                predict_spans = set()
                if target_eval:
                    p_start = -1
                    for i in range(len(prediction)):
                        if prediction[i].startswith("B"):
                            p_start = i
                        if prediction[i].startswith("E"):
                            p_end = i
                            predict_spans.add(Span(p_start, p_end, prediction[i][2:]))
                        if prediction[i].startswith("S"):
                            predict_spans.add(Span(i, i, prediction[i][2:]))

                if opinion_eval:
                    p_start = -1
                    for i in range(len(prediction)):
                        if prediction[i].startswith("b"):
                            p_start = i
                        if prediction[i].startswith("e"):
                            p_end = i
                            predict_spans.add(Span(p_start, p_end, prediction[i][2:]))
                        if prediction[i].startswith("s"):
                            predict_spans.add(Span(i, i, prediction[i][2:]))
            # print(output_spans)
            # print(predict_spans)
            if not pair_eval:
                total_entity += len(output_spans)
                total_predict += len(predict_spans)
                pp += len(predict_spans.intersection(output_spans))
        print("toal num of entity: ", total_entity)
        print("total num of prediction: ", total_predict)
        precision = pp * 1.0 / total_predict if total_predict != 0 else 0
        recall = pp * 1.0 / total_entity if total_entity != 0 else 0
        fscore = (
            2.0 * precision * recall / (precision + recall)
            if precision != 0 or recall != 0
            else 0
        )

        # ret = [precision, recall, fscore]
        fscore = FScore(precision, recall, fscore)

        return fscore
