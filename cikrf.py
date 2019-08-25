#!/usr/bin/env python3

from datetime import date as Date
from enum import Enum
from itertools import product
from lxml.etree import XPath  # type:ignore
from lxml.html import document_fromstring  # type: ignore
from typing import Dict, Generator, Iterable, List, Tuple, Type, TypeVar, Optional
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit
from warnings import warn

# __all__ = ["Commission", "Scope", "Object", "Election", "run"]  # FIXME

_E = TypeVar("_E", bound="Election")
_T = TypeVar("_T")
Action = Generator[str, Optional[str], _T]


def _urladjust(
    url: str, params: Dict[str, List[str]] = dict(), **kwargs: List[str]
) -> str:
    parts = urlsplit(url)
    query = parse_qs(parts.query, strict_parsing=True)
    query.update(params, **kwargs)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), "")
    )


def _normalizespace(s: str) -> str:
    return " ".join(s.strip().split())


def _fromdate(date: Date) -> str:
    return date.strftime("%d.%m.%Y")


_NAME = XPath(
    './/*[normalize-space()="Наименование комиссии" or '
    'normalize-space()="Наименование Избирательной комиссии"]/'
    "ancestor-or-self::td[1]/following-sibling::td/text()"
)
_OPTIONS = XPath(".//option")


class Commission(object):
    __slots__ = ("number", "parent", "_tree", "url")

    def __init__(
        self,
        parent: List[Tuple[Optional[int], Optional[str]]],
        number: Optional[int],
        url: str,
    ) -> None:
        self.parent = parent
        self.number = number
        self.url = url
        self._tree = None

    def __str__(self) -> str:
        return "<{}>".format(self.url)

    def __repr__(self) -> str:
        return "{}(parent={!r}, number={!r}, url={!r})".format(
            type(self).__qualname__, self.parent, self.number, self.url
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Election):
            return NotImplemented
        return self.url == other.url

    def load(self) -> Action[None]:
        if self._tree is not None:
            return
        self._tree = document_fromstring((yield self.url))

    @property
    def name(self) -> str:
        if self._tree is None:
            raise AttributeError("name")
        name = _NAME(self._tree)
        if name:
            return _normalizespace(name[0])
        else:
            warn("{}: Unknown commission name".format(self), stacklevel=2)
            return None

    @property
    def path(self) -> List[Tuple[Optional[int], Optional[str]]]:
        return self.parent + [(self.number, self.name)]

    @property
    def children(self) -> Iterable["Commission"]:
        if self._tree is None:
            raise AttributeError("children")
        return [
            Commission(
                parent=self.path,
                number=int(o.text.split()[0]) if o.text.split()[0].isdigit() else None,
                url=urljoin(self.url, o.get("value")),
            )
            for o in _OPTIONS(self._tree)
            if o.get("value")
        ]


class Scope(Enum):
    def __repr__(self) -> str:
        return "{}.{}".format(type(self).__qualname__, self.name)

    COUNTRY = "1"
    PROVINCE = "2"
    COMMUNE = "3"
    MUNICIPALITY = "4"


class Object(Enum):
    def __repr__(self) -> str:
        return "{}.{}".format(type(self).__qualname__, self.name)

    QUESTION = "0"
    OFFICE = "1"
    SEAT = "2"
    SEAT_RECALL = "3"
    OFFICE_RECALL = "4"


_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}


_RESULTS = XPath('.//a[@class="vibLink"]/ancestor::table[1]/tr')
_LABELS = XPath(".//td[1]")
_LINK = XPath(".//a")
_BOLD = XPath(".//b")
_TEXT = XPath("text()")


class Election(Commission):
    __slots__ = ("date", "object", "scope", "title", "url")

    def __init__(
        self, title: str, date: Date, scope: Scope, object: Object, **kwargs
    ) -> None:
        super(Election, self).__init__(parent=[], number=None, **kwargs)
        self.title = title
        self.date = date
        self.scope = scope
        self.object = object

    def __str__(self) -> str:
        return "<{}> ({})".format(self.url, self.title)

    def __repr__(self) -> str:
        return "{}(title={!r}, date={!r}, scope={!r}, object={!r}, url={!r})".format(
            type(self).__qualname__,
            self.title,
            self.date,
            self.scope,
            self.object,
            self.url,
        )

    SEARCH = "http://www.vybory.izbirkom.ru/region/izbirkom?action=search_by_calendar"

    @classmethod
    def _search(
        cls: Type[_E], start: Date, end: Date, scope: Scope, object: Object
    ) -> Action[Iterable[_E]]:
        text = yield _urladjust(
            cls.SEARCH,
            start_date=[_fromdate(start)],
            end_date=[_fromdate(end - Date.resolution)],
            urovproved=[scope.value],
            vidvibref=[object.value],
            # TODO 1 = Основные, 2 = Основные повторные, 3 = Основные отложенные, 4 = Основные отдельные, 5 = Дополнительные, 6 = Дополнительные повторные, 7 = Довыборы, 8 = Повторное голосование, 9 = Основные выборы и повторное голосование
            vibtype=["all"],
            # TODO 1 = мажоритарная, 5 = мажоритарная - по общерегиональному округу и по отдельным избирательным округам, 4 = мажоритарная по общерегиональному округу, 2 = пропорциональная, 3 = смешанная - пропорциональная и мажоритарная, 6 = пропорциональная и мажоритарная по общерегиональному округу и отдельным избирательным округам
            sxemavib=["all"],
            # TODO 0 = whole country
            # region=["0"],
        )
        page = document_fromstring(text)

        results = []
        date = None
        place: List[str] = []
        for row in _RESULTS(page):
            labels, = _LABELS(row)
            if labels.get("colspan") is not None:
                day, month, year, _ = _normalizespace(labels.text).split()
                date = Date(int(year), _MONTHS[month], int(day))
                continue
            assert date is not None

            a, = _LINK(row)
            province, = _BOLD(labels) or (None,)
            if province is not None:
                place = (
                    [_normalizespace(province.text)]
                    if province.text != "Российская Федерация"
                    else []
                )
            commune, = _TEXT(labels) or (None,)
            if commune is not None and commune.strip():
                place = [place[0], _normalizespace(commune)]

            results.append(
                cls(
                    title=_normalizespace(a.text),
                    date=date,
                    scope=scope,
                    object=object,
                    url=_urladjust(
                        urljoin(cls.SEARCH, a.get("href")), sub_region=["99"]
                    ),
                )
            )

        return results

    @classmethod
    def search(
        cls: Type[_E],
        start: Date = Date(1991, 6, 12),
        end: Optional[Date] = None,
        scopes: Iterable[Scope] = Scope.__members__.values(),
        objects: Iterable[Object] = Object.__members__.values(),
    ) -> Action[Iterable[_E]]:
        if end is None:
            end = Date.today() + Date.resolution
        results: List[_E] = []
        for s, o in product(set(scopes), set(objects)):
            results.extend((yield from cls._search(start, end, s, o)))
        results.sort(key=lambda e: e.date)
        return results


# FIXME test code
from gzip import open as gzipopen
from requests import get
from urllib.parse import quote


def run(action: Action[_T]) -> _T:
    try:
        text = None
        while True:
            text = get(action.send(text)).text
    except StopIteration as result:
        return result.value


def run_cached(action: Action[_T]) -> _T:
    try:
        text = None
        url = None
        while True:
            url = action.send(text)
            print(url)
            cache = "cache/" + quote(url, safe="") + ".gz"
            try:
                with gzipopen(cache, "rt") as f:
                    text = f.read()
            except FileNotFoundError:
                text = get(url).text
                with gzipopen(cache, "wt") as f:
                    f.write(text)
    except StopIteration as result:
        return result.value


def visit(c: Commission, level: int) -> None:
    if level <= 0:
        return
    run_cached(c.load())
    print(c.path)
    for cc in c.children:
        visit(cc, level - 1)


es = run_cached(Election.search(end=Date(2019, 8, 20)))
for e in es:
    print("***", e.date, "-", e.title, "***")
    visit(e, 4)
