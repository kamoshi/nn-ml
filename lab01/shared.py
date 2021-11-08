from typing import NewType, Tuple
import random


LabelledPoint = NewType("LabelledPoint", Tuple[list[int], int])


FACTS_AND: list[LabelledPoint] = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

FACTS_OR: list[LabelledPoint] = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

FACTS_XOR = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

FACTS_NAND = [
    ([0, 0], 1),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]


get_noise = lambda x: random.random() / x
get_sign = lambda: [-1, 1][random.randint(0, 1)]


def noisify_point(point: LabelledPoint) -> LabelledPoint:
    (x, y), c = point
    m_x, m_y, s_x, s_y = get_noise(6), get_noise(6), get_sign(), get_sign()
    return ([x + s_x*m_x, y + s_y*m_y], c)


def convert_points_to_bipolar(points: list[LabelledPoint]) -> list[LabelledPoint]:
    return [((x, y), -1 if c == 0 else 1) for (x, y), c in points]


def gen_sequences(length: int, facts): return zip(*[noisify_point(facts[random.randint(0, 3)]) for _ in range(length)])
