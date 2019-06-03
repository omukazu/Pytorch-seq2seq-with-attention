import argparse
import glob
import multiprocessing as mp
import os
import time
from typing import List, Tuple

import regex
from pyknp import Juman

WHITE_LIST = regex.compile(r'[\p{Hiragana}\p{Katakana}\p{Han}、「」]+')
HIRAGANA = regex.compile(r'[\p{Hiragana}、「」]+')
# characters that do not represent one mora each / specific symbols
SUTEGANA = r'[ぁぃぅぇぉゃゅょァィゥェォャュョ、「」]'
MORA_PATTERN = {5, 12, 17, 24, 31}


def check_fullmatch(lines: List[str],
                    jobs: int
                    ) -> List[str]:
    chunk_size = len(lines) // jobs + 1
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    with mp.Pool(jobs) as p:
        checked_chunks = p.map(_check_fullmatch, chunks)

    checked = []
    for chunk in checked_chunks:
        checked.extend(chunk)
    return checked


def _check_fullmatch(chunk: List[str]
                     ) -> List[str]:
    return [line for line in chunk if WHITE_LIST.fullmatch(line)]


def analyze(lines: List[str],
            jobs: int
            ) -> List[Tuple]:
    jumanpp = Juman()
    chunk_size = len(lines) // jobs + 1
    arguments = [(lines[i:i + chunk_size], jumanpp) for i in range(0, len(lines), chunk_size)]
    with mp.Pool(jobs) as p:
        analyzed_chunks = p.starmap(_analyze, arguments)

    analyzed = []
    for chunk in analyzed_chunks:
        analyzed.extend(chunk)
    return analyzed


def _analyze(chunk: List[str],
             jumanpp: Juman
             ) -> List[Tuple]:
    analyzed_chunk = []
    for sentence in chunk:
        try:
            analyzed_chunk.append(jumanpp.analysis(sentence))
        except ValueError:
            analyzed_chunk.append(None)
            # print(sentence)

    return [([(mrph.midasi, mrph.yomi, mrph.hinsi, mrph.katuyou2) for mrph in analyzed_sentence.mrph_list()], chunk[i])
            for i, analyzed_sentence in enumerate(analyzed_chunk)]


def cumsum(sub: List[tuple],
           l: int
           ) -> List[int]:
    mora_counts = []
    count = 0
    for i in range(l):
        count += len(sub[i][1]) - len(regex.findall(SUTEGANA, sub[i][1]))
        mora_counts.append(count)
    # return the cumulative sum of mora
    return mora_counts


def rule(tanka: List
         ) -> bool:
    if (tanka[2][-1][2] != '特殊' and '基本形' not in tanka[2][-1][3]) or \
            any([contidition in mora[0][2] for mora in tanka for contidition in {'助詞', '判定詞'}]) or \
            all([contidition not in tanka[0][-1][2] for contidition in {'助詞', '特殊'}]) or \
            HIRAGANA.fullmatch(''.join([m[1] for mora in tanka for m in mora])) is None or \
            '基本形' not in tanka[4][-1][3]:
        return False
    else:
        return True


def extract_tanka(line: List[tuple],
                  index: int,
                  count: List[int]
                  ) -> List:
    inversed = count[::-1]
    n = len(count)
    attention = [0,
                 n - inversed.index(5),
                 n - inversed.index(12),
                 n - inversed.index(17),
                 n - inversed.index(24),
                 n - inversed.index(31)]
    tanka = []
    for i in range(len(attention) - 1):
        tanka.append(line[index + attention[i]:index + attention[i + 1]])
    if rule(tanka):
        return tanka


def extract_tankas(lines: List[tuple],
                   jobs: int
                   ) -> List:
    chunk_size = len(lines) // jobs + 1
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    with mp.Pool(jobs) as p:
        extracted_chunks = p.map(_extract_tankas, chunks)

    extracted = []
    for chunk in extracted_chunks:
        extracted.extend(chunk)
    return [tanka for tanka in extracted if tanka]


def _extract_tankas(chunk: List[tuple]
                    ) -> List:
    tankas = []
    for sentences in chunk:
        analyzed = sentences[0]
        n = len(analyzed)
        mora_counts = [cumsum(analyzed[start:], n - start) for start in range(n)]

        for index, count in enumerate(mora_counts):
            if len(MORA_PATTERN - set(count)) == 0:
                tankas.append((extract_tanka(analyzed, index, count), sentences[1]))
    return [tanka for tanka in tankas if tanka[0]]


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description='extract_5-7-5-7-7_pattern')
    parser.add_argument('INPUT', help='path to input')
    parser.add_argument('--jobs', type=int, default=1, help='p')
    parser.add_argument('OUTPUT', default='None', help='path to output')
    args = parser.parse_args()

    # path = os.path.join(args.INPUT, '**/*.gz')
    path = os.path.join(args.INPUT, '*')
    input_files = glob.glob(path)
    output_files = [os.path.join(args.OUTPUT, str(i)) for i in range(len(input_files))]
    jobs = args.jobs

    for i, file in enumerate(input_files):
        with open(file, "r") as inp:
            lines = [sentence for line in inp for sentence in line.strip().split('。') if sentence]

        print('start processing' + f' file-{str(i)}')
        lines = check_fullmatch(lines, jobs)
        lines = analyze(lines, jobs)
        lines = extract_tankas(lines, jobs)

        elapsed_time = time.time() - start
        print(f'{elapsed_time}seconds spend')

        print('start writing' + f' file-{str(i)}')
        with open(output_files[i], "w") as out:
            for el in lines:
                out.write(''.join([t[0] for tl in el[0] for t in tl]) + '\t' + el[1] + '\n')
    print('done')


if __name__ == '__main__':
    main()
