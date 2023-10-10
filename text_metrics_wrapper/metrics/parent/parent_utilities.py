"""Disclaimer:
All the functions in this page are copy-pasted from the original github repo INSERT GITHUB REPO HERE.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import io
import json
import logging
import math
from absl import app
from absl import flags
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf


from bs4 import BeautifulSoup
from copy import deepcopy


def extend_empty_cells(table_list):
    table_list_copy = deepcopy(table_list)
    spanning_row = list(filter(lambda x: x["rowspan"] > 1, table_list_copy))
    for entry in spanning_row:
        span = range(entry["row"] + 1, entry["row"] + entry["rowspan"])
        entry["rowspan"] = 1
        for _r in span:
            new_entry = deepcopy(entry)
            new_entry["row"] = _r
            table_list_copy.append(new_entry)
    spanning_col = list(filter(lambda x: x["colspan"] > 1, table_list_copy))
    for entry in spanning_col:
        span = range(entry["col"] + 1, entry["col"] + entry["colspan"])
        entry["colspan"] = 1
        for _c in span:
            new_entry = deepcopy(entry)
            new_entry["col"] = _c
            table_list_copy.append(new_entry)
    table_list_copy = sorted(table_list_copy, key=lambda x: (x["row"], x["col"]))
    table_row_dic = []
    for _r in range(max([_el["row"] for _el in table_list_copy]) + 1):
        row_entries = list(filter(lambda x: x["row"] == _r, table_list_copy))
        table_row_dic.append(row_entries)
    return table_row_dic


def html_table_to_entries(html_table):
    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_table, "html.parser")

    # Find all the rows in the table
    rows = soup.find_all("tr")

    row_content = []

    for row_idx, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        row_content.append([])

        for col_idx, cell in enumerate(cells):
            cell_text = cell.get_text(strip=True)
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))

            entry = {"text": cell_text, "rowspan": rowspan, "colspan": colspan}

            row_content[-1].append(entry)

    # adjust row/col index according to the colspan
    entries = []
    cols = list(set(list(range(max([len(row) for row in rows])))))
    rows = list(set(list(range(len(rows)))))
    rows.sort()
    cols.sort()

    occupied = []
    for _r, row in enumerate(row_content):
        for _c, _el in enumerate(row):
            position = [_r, _c]
            while position in occupied:
                position[1] += 1
            _el["row"] = position[0]
            _el["col"] = position[1]
            for _r2 in range(_r, _r + _el["rowspan"]):
                for _c2 in range(_c, _c + _el["colspan"]):
                    occupied.append([_r2, _c2])
            entries.append(_el)

    return entries


def _text_reader_reference(raw_text):
    """Yields lines from the text file.

    Performs lowercasing and white-space tokenization on each line before
    returning.

    Args:
        text_file: String filename.
        multiple: Whether multiple references / generations are expected in a line.
    """
    for line in raw_text:
        if isinstance(line, list):
            yield [item.strip().lower().split() for item in line]
        else:
            yield [line.strip().lower().split()]


def _text_reader_candidate(raw_text):
    """Yields lines from the text file.

    Performs lowercasing and white-space tokenization on each line before
    returning.

    Args:
        text_file: String filename.
        multiple: Whether multiple references / generations are expected in a line.
    """
    for line in raw_text:
        yield line.strip().lower().split()


def _table_reader(raw_table):
    """Yields tables from the table file.

    Tables are parsed into a list of tuples with tokenized entries.

    Args:
        table_file: String filename.
    """
    line = raw_table[0]
    table_entries = html_table_to_entries(line)
    complete_table_entries = extend_empty_cells(table_entries)
    headers = [member["text"] for member in complete_table_entries[0][1:]]
    indexes = [member[0]["text"] for member in complete_table_entries[1:]]
    table = []
    for _r, row in enumerate(complete_table_entries[1:]):
        for _c, entry in enumerate(row[1:]):
            table.append(
                [
                    [x for x in indexes[_r].lower().split()],
                    [x for x in headers[_c].lower().split()],
                    [x for x in entry["text"].lower().split()],
                ]
            )
    # table_filtered = []
    # for _entry in table:
    #     if raw_table[1] is None:
    #         continue
    #     for _entity in raw_table[1]:
    #         if _entity.lower() in ' '.join(_entry[0]) or _entity.lower() in ' '.join(_entry[1]):
    #             table_filtered.append(_entry)
    #             break
    # for _sent in raw_table[2]:
    #     table_filtered.append([
    #         _sent.lower().split(),
    #         [], []
    #     ])
    return table


def _table_reader_triplets(raw_table, header=True):
    """Yields tables from the table file.

    Tables are parsed into a list of tuples with tokenized entries.

    Args:
        table_file: String filename.
    """
    for line in raw_table:
        table_entries = html_table_to_entries(line)
        complete_table_entries = extend_empty_cells(table_entries)
        if not header:
            complete_table_entries = complete_table_entries[1:]
        # pylint: disable=g-complex-comprehension
        table = [[member["text"] for member in entry] for entry in complete_table_entries]
        yield table


def cooccur_probability_fn(counts):
    """Returns function for computing entailment probability.

    Args:
      counts: Dict mapping unigrams / bigrams (joined using "|||") to their
        counts.

    Returns:
      Function handle to compute entailment probability.
    """

    def _cooccur_probability(ngram, table):
        """Returns probability of ngram being entailed by the table.

        Uses the co-occurrence counts given along with the lexical
        entailment model described in:

          Glickman, Oren, Ido Dagan, and Moshe Koppel.
          "A lexical alignment model for probabilistic textual entailment."
          Machine Learning Challenges.
          Springer, Berlin, Heidelberg, 2006. 287-298.

        E.g.:
          >>> _cooccur_probability(["michael", "dahlquist"],
                                      [(["name"], ["michael", "dahlquist"])])
          >>> 1.0

        Args:
          ngram: List of tokens.
          table: List of either (attribute, value) pairs or (head, relation, tail)
            triples. Each member of the pair / triple is assumed to already be
            tokenized into a list of strings.

        Returns:
          prob: Float probability of ngram being entailed by the table.
        """
        table_toks = set()
        for item in table:
            if len(item) == 2:
                # attribute, value
                table_toks.add("_".join(item[0]))
                table_toks.update(item[1])
            else:
                # head, relation, tail
                table_toks.update(item[0] + ["_".join(item[1])] + item[2])
        probability = 1.0
        for xtok in ngram:
            if xtok in table_toks:
                continue
            max_p = 0.0
            for btok in table_toks:
                if btok not in counts:
                    continue
                p = float(counts.get(btok + "|||" + xtok, 0.0)) / counts[btok]
                if p > max_p:
                    max_p = p
            probability *= max_p
        return math.pow(probability, 1.0 / len(ngram))

    return _cooccur_probability


def overlap_probability(ngram, table, smoothing=0.0, stopwords=None):
    """Returns the probability that the given n-gram overlaps with the table.

    A simple implementation which checks how many tokens in the n-gram are also
    among the values in the table. For tables with (attribute, value) pairs on the
    `value` field is condidered. For tables with (head, relation, tail) triples a
    concatenation of `head` and `tail` are considered.

    E.g.:
      >>> overlap_probability(["michael", "dahlquist"],
                               [(["name"], ["michael", "dahlquist"])])
      >>> 1.0

    Args:
      ngram: List of tokens.
      table: List of either (attribute, value) pairs or (head, relation, tail)
        triples. Each member of the pair / triple is assumed to already be
        tokenized into a list of strings.
      smoothing: (Optional) Float parameter for laplace smoothing.
      stopwords: (Optional) List of stopwords to ignore (assign P = 1).

    Returns:
      prob: Float probability of ngram being entailed by the table.
    """
    # pylint: disable=g-complex-comprehension
    if len(table[0]) == 2:
        table_values = set([tok for _, value in table for tok in value])
    else:
        table_values = set([tok for head, _, tail in table for tok in head + tail])
    overlap = 0
    for token in ngram:
        if stopwords is not None and token in stopwords:
            overlap += 1
            continue
        if token in table_values:
            overlap += 1
    return float(overlap + smoothing) / float(len(ngram) + smoothing)


def _mention_probability(table_entry, sentence, smoothing=0.0):
    """Returns the probability that the table entry is mentioned in the sentence.

    A simple implementation which checks the longest common subsequence between
    the table entry and the sentence. For tables with (attribute, value) pairs
    only the `value` is considered. For tables with (head, relation, tail) triples
    a concatenation of the `head` and `tail` is considered.

    E.g.:
      >>> _mention_probability((["name"], ["michael", "dahlquist"]),
                               ["michael", "dahlquist", "was", "a", "drummer"])
      >>> 1.0

    Args:
      table_entry: Tuple of either (attribute, value) or (head, relation, tail).
        Each member of the tuple is assumed to already be tokenized into a list of
        strings.
      sentence: List of tokens.
      smoothing: Float parameter for laplace smoothing.

    Returns:
      prob: Float probability of entry being in sentence.
    """
    if len(table_entry) == 2:
        value = table_entry[1]
    else:
        value = table_entry[0] + table_entry[2]
    overlap = _len_lcs(value, sentence)
    return float(overlap + smoothing) / float(len(value) + smoothing)


def _len_lcs(x, y):
    """Returns the length of the Longest Common Subsequence between two seqs.

    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: sequence of words
      y: sequence of words

    Returns
      integer: Length of LCS between x and y
    """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    """Computes the length of the LCS between two seqs.

    The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    Args:
      x: collection of words
      y: collection of words

    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _ngrams(sequence, order):
    """Yields all ngrams of given order in sequence."""
    assert order >= 1
    for n in range(order, len(sequence) + 1):
        yield tuple(sequence[n - order : n])


def _ngram_counts(sequence, order):
    """Returns count of all ngrams of given order in sequence."""
    if len(sequence) < order:
        return collections.Counter()
    return collections.Counter(_ngrams(sequence, order))


def parent(
    predictions,
    references,
    tables,
    lambda_weight=0.5,
    smoothing=0.00001,
    max_order=4,
    entailment_fn=overlap_probability,
    mention_fn=_mention_probability,
):
    """Metric for comparing predictions to references given tables.

    Args:
      predictions: An iterator over tokenized predictions.
        Each prediction is a list.
      references: An iterator over lists of tokenized references.
        Each prediction can have multiple references.
      tables: An iterator over the tables. Each table is a list of tuples, where a
        tuple can either be (attribute, value) pair or (head, relation, tail)
        triple. The members of the tuples are assumed to be themselves tokenized
        lists of strings. E.g.
        `[(["name"], ["michael", "dahlquist"]),
        (["birth", "date"], ["december", "22", "1965"])]`
        is one table in the (attribute, value) format with two entries.
      lambda_weight: Float weight in [0, 1] to multiply table recall.
      smoothing: Float value for replace zero values of precision and recall.
      max_order: Maximum order of the ngrams to use.
      entailment_fn: A python function for computing the probability that an
        ngram is entailed by the table. Its signature should match that of
        `overlap_probability` above.
      mention_fn: A python function for computing the probability that a
        table entry is mentioned in the text. Its signature should
          match that of `_mention_probability` above.

    Returns:
      precision: Average precision of all predictions.
      recall: Average recall of all predictions.
      f1: Average F-scores of all predictions.
      all_f_scores: List of all F-scores for each item.
    """
    precisions, recalls, all_f_scores = [], [], []
    reference_recalls, table_recalls = [], []
    all_lambdas = []
    for prediction, list_of_references, table in zip(predictions, references, tables):
        c_prec, c_rec, c_f = [], [], []
        ref_rec, table_rec = [], []
        for reference in list_of_references:
            # Weighted ngram precisions and recalls for each order.
            ngram_prec, ngram_rec = [], []
            for order in range(1, max_order + 1):
                # Collect n-grams and their entailment probabilities.
                pred_ngram_counts = _ngram_counts(prediction, order)
                pred_ngram_weights = {ngram: entailment_fn(ngram, table) for ngram in pred_ngram_counts}
                ref_ngram_counts = _ngram_counts(reference, order)
                ref_ngram_weights = {ngram: entailment_fn(ngram, table) for ngram in ref_ngram_counts}

                # Precision.
                numerator, denominator = 0.0, 0.0
                for ngram, count in pred_ngram_counts.items():
                    denominator += count
                    prob_ngram_in_ref = min(1.0, float(ref_ngram_counts.get(ngram, 0) / count))
                    numerator += count * (prob_ngram_in_ref + (1.0 - prob_ngram_in_ref) * pred_ngram_weights[ngram])
                if denominator == 0.0:
                    # Set precision to 0.
                    ngram_prec.append(0.0)
                else:
                    ngram_prec.append(numerator / denominator)

                # Recall.
                numerator, denominator = 0.0, 0.0
                for ngram, count in ref_ngram_counts.items():
                    prob_ngram_in_pred = min(1.0, float(pred_ngram_counts.get(ngram, 0) / count))
                    denominator += count * ref_ngram_weights[ngram]
                    numerator += count * ref_ngram_weights[ngram] * prob_ngram_in_pred
                if denominator == 0.0:
                    # Set recall to 1.
                    ngram_rec.append(1.0)
                else:
                    ngram_rec.append(numerator / denominator)

            # Compute recall against table fields.
            table_mention_probs = [mention_fn(entry, prediction) for entry in table]
            table_rec.append(sum(table_mention_probs) / len(table))

            # Smoothing.
            for order in range(1, max_order):
                if ngram_prec[order] == 0.0:
                    ngram_prec[order] = smoothing
                if ngram_rec[order] == 0.0:
                    ngram_rec[order] = smoothing

            # Compute geometric averages of precision and recall for all orders.
            w = 1.0 / max_order
            if any(prec == 0.0 for prec in ngram_prec):
                c_prec.append(0.0)
            else:
                sp = (w * math.log(p_i) for p_i in ngram_prec)
                c_prec.append(math.exp(math.fsum(sp)))
            if any(rec == 0.0 for rec in ngram_rec):
                ref_rec.append(smoothing)
            else:
                sr = [w * math.log(r_i) for r_i in ngram_rec]
                ref_rec.append(math.exp(math.fsum(sr)))

            # Combine reference and table recalls.
            if table_rec[-1] == 0.0:
                table_rec[-1] = smoothing
            if ref_rec[-1] == 0.0 or table_rec[-1] == 0.0:
                c_rec.append(0.0)
            else:
                if lambda_weight is None:
                    lw = sum([mention_fn(entry, reference) for entry in table]) / len(table)
                    lw = 1.0 - lw
                else:
                    lw = lambda_weight
                all_lambdas.append(lw)
                c_rec.append(math.exp((1.0 - lw) * math.log(ref_rec[-1]) + (lw) * math.log(table_rec[-1])))

            # F-score.
            c_f.append((2.0 * c_prec[-1] * c_rec[-1]) / (c_prec[-1] + c_rec[-1] + 1e-8))

        # Get index of best F-score.
        max_i = max(enumerate(c_f), key=lambda x: x[1])[0]
        precisions.append(c_prec[max_i])
        recalls.append(c_rec[max_i])
        all_f_scores.append(c_f[max_i])
        reference_recalls.append(ref_rec[max_i])
        table_recalls.append(table_rec[max_i])

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f_score = sum(all_f_scores) / len(all_f_scores)

    return avg_precision, avg_recall, avg_f_score, all_f_scores
