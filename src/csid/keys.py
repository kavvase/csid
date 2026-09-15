from __future__ import annotations

from collections.abc import Iterable


def pair_key(a: int, b: int) -> tuple[int, int]:
    a_i, b_i = int(a), int(b)
    return (a_i, b_i) if a_i <= b_i else (b_i, a_i)


def triple_key(a: int, b: int, c: int) -> tuple[int, int, int]:
    x, y, z = sorted((int(a), int(b), int(c)))
    return x, y, z


def as_pair(values: Iterable[int]) -> tuple[int, int]:
    a, b = values
    return int(a), int(b)


def as_triple(values: Iterable[int]) -> tuple[int, int, int]:
    a, b, c = values
    return int(a), int(b), int(c)
