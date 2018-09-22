from asks            import Session
from asks.errors     import AsksException
from bs4             import BeautifulSoup, Tag
from collections     import OrderedDict, namedtuple
from collections.abc import Iterable
from datetime        import date as Date, datetime as DateTime, timedelta as TimeDelta
from enum            import Enum
from jsonseq         import dump, load
from trio            import Queue, open_nursery, run, sleep
from types           import MethodType
from urllib.parse    import parse_qs, urlencode, urljoin, urlsplit, urlunsplit

# FIXME debug imports
from asks import init as init_asks
from pprint import pprint, pformat
from shutil import get_terminal_size
from sys import stdout, stderr
from textwrap import shorten

PARSER = 'html5lib'

columns = 1000 #columns = get_terminal_size().columns # FIXME
def trace(string):
	print(shorten(string, columns, placeholder='...'), file=stderr, flush=True)

def urladjust(url, params=dict(), **named):
	parts = urlsplit(url)
	query = parse_qs(parts.query, strict_parsing=True)
	query.update(params, **named)
	return urlunsplit((parts.scheme, parts.netloc, parts.path,
	                   urlencode(query, doseq=True), ''))

def normalize(string):
	assert string is not None
	return ' '.join(str(string).split()).replace('\u2013', '-')

def todate(string):
	return DateTime.strptime(normalize(string), '%d.%m.%Y').date()
def fromdate(date):
	return date.strftime('%d.%m.%Y')

def matches(string):
	string = string.casefold()
	def match(s): return s and normalize(s).casefold() == string
	return match

def disj(*matches):
	matches = list(matches)
	def match(s): return any(m(s) for m in matches)
	return match

def nodata(page):
	mess = page.find(string=matches('Нет данных для построения отчета.'))
	return mess is not None

class Scope(Enum):
	COUNTRY  = '1'
	PROVINCE = '2'
	COMMUNE  = '3'
	MUNICPTY = '4'
	# SETTLMNT FIXME

class Hints:
	def __init__(self):
		self.depth = None

Report = namedtuple('Report', ['records', 'results'])
Row    = namedtuple('Row',    ['number', 'name', 'value'])

class Commission:
	__slots__ = ['url', '_ppath', '_hints', '_page', '_children']

	def __init__(self, url, ppath, hints):
		self.url       = url
		self._ppath    = ppath
		self._hints    = hints
		self._page     = dict()
		self._children = None

	def __repr__(self):
		return f'{type(self).__qualname__}(url={self.url!r}, ppath={self._ppath!r}, hints={self._hints!r})'

	async def page(self, session, type):
		if self._page.get(type) is None:
			url = urladjust(self.url, type=[type])
			while True:
				try: res = await session.get(url)
				except AsksException as e:
					print('ERR', url, e, flush=True) # FIXME
				except OSError as e:
					print('ERR', url, e, flush=True) # FIXME
				else:
					if res.status_code // 100 == 2: break
					else: print(res.status_code, url, res.reason_phrase, flush=True) # FIXME
				await sleep(0) # FIXME back off?
			encoding = res.encoding.lower() if 'charset' in res.headers.get('Content-Type','') else None # FIXME unreliable (change in upstream asks?)
			self._page[type] = BeautifulSoup(res.content, PARSER, from_encoding=encoding)
		else:
			await sleep(0)
		return self._page[type]

	async def _result_types(self, session):
		page = await self.page(session, '0')
		capt = page.find(string=matches('Результаты выборов'))
		if capt is None:
			# NB: In general, nodata(page) is false: there _are_
			# (local) elections with no results.
			return dict()

		ancs = (row.find('a')
		        for row in capt.find_parent('tr').next_siblings
		        if isinstance(row, Tag))
		return(OrderedDict
			((parse_qs(urlsplit(a['href']).query)['type'][0],
			  normalize(a.string))
			 for a in ancs))

	@staticmethod
	def _single_result_data(table):
		rows = [tr('td') for tr in table('tr')]
		assert all(len(row) in {2,3} for row in rows)
		seps = [i for i, row in enumerate(rows) if len(row) == 2]
		data = [Row(number=normalize(row[0].string),
		            name=normalize(row[1].string),
		            value=int(row[2].find(string=True).string))
		        for row in rows if len(row) == 3]
		return seps, data

	async def _single_result(self, session, type): # FIXME version
		tabs = (await self.page(session, type))(cellpadding='2')
		assert len(tabs) == 1
		seps, data = self._single_result_data(tabs[0])
		return Report(records=data[:seps[0]], results=data[seps[0]:])

	async def _aggregate_results(self, session, type):
		tabs = (await self.page(session, type))(cellpadding='2')
		assert len(tabs) == 2
		# Left table contains headers
		seps, head = self._single_result_data(tabs[0])
		# Right table contains data per commission
		rows = [tr('td') for tr in tabs[1]('tr')]
		comms = (normalize(td.find('a').string) for td in rows[0])
		datas = ([h._replace(value=int(v.find(string=True).string))
		          for h, v in zip(head, col)]
		         for col in zip(*(row for i,row in enumerate(rows) if i not in seps)))
		return(OrderedDict
			((c, Report(records=d[:seps[1]-1],
			            results=d[seps[1]-1:]))
			 for c, d in zip(comms, datas)))

	async def _directory(self, session):
		# FIXME don't actually need the directory (most of the time)
		return await self.page(session, '0')

	async def name(self, session):
		page = await self._directory(session)
		crumbs = page.find('table', height='80%').find('td')('a')
		if len(crumbs) > 0:
			# NB. When the crumbs are missing from the parent, the
			# text in crumbs[:-1] can _differ_ from self._ppath
			# FIXME Example: http://www.vybory.izbirkom.ru/region/izbirkom?action=show&vrn=485400692143&region=85&prver=0&pronetvd=0
			assert len(crumbs)-1 == len(self._ppath)
			return normalize(crumbs[-1].string)

		page = await self.page(session, '0')
		caption = page.find(string=disj
			(matches('Наименование комиссии'),
			 matches('Наименование избирательной комиссии')))
		if caption is None:
			assert nodata(page)
			return None
		return normalize(capt.find_parent('td')
		                     .find_next_sibling('td')
		                     .string)

	async def path(self, session):
		return self._ppath + [await self.name(session)]

	async def children(self, session):
		if len(self._ppath)+1 == self._hints.depth:
			return []
		elif self._children is None:
			page = await self._directory(session)
			self._children = \
				[Commission(urljoin(self.url, o['value']),
				            await self.path(session),
				            self._hints)
				 for o in page('option')
				 if o.attrs.get('value')]
			if not self._children:
				assert page.find(string=matches
					("нет отчета по навигации или же это "
					 "конечный твд == уик"))
				if self._hints.depth is None:
					self._hints.depth = len(self._ppath)+1
				assert(self._hints.depth is None or
				       self._hints.depth == len(self._ppath)+1)
		else:
			await sleep(0)
		return self._children

class Election(Commission):
	__slots__ = ['title', 'place', 'root']

	def __init__(self, url, title=None, place=None):
		self.title = title
		self.place = place
		super().__init__(url, [], Hints())

	def __repr__(self):
		return f'{type(self).__qualname__}(url={self.url!r}, title={self.title!r}, place={self.place!r})'

	_CONTEXT = 'http://example.org/election.jsonld' # FIXME

	def tojson(self):
		return {'@context': self._CONTEXT, 'url': self.url,
		        'title': self.title, 'place': self.place}

	@classmethod
	def fromjson(cls, data):
		ctx = data.pop('@context', None)
		if ctx and ctx != cls._CONTEXT:
			raise ValueError('Invalid @context for election JSON')
		return cls(**data)

	async def date(self, session):
		page = await self.page(session, '0')
		capt = page.find(string=matches('Дата голосования'))
		if capt is None:
			assert nodata(page)
			return None
		return todate(capt.find_parent('td')
		                  .find_next_sibling('td')
		                  .string)

	@classmethod
	async def search(cls, session, start=Date(1991, 6, 12), end=None, *,
	                 scope=list(Scope)): # FIXME parameter order?

		if end is None:
			end = Date.today() # FIXME reconsider this default?
		else:
			# Python convention prescribes half-open ranges
			end -= Date.resolution
		scope = ([Scope(s).value for s in scope]
			 if isinstance(scope, Iterable)
			 else [Scope(scope).value])

		SEARCH  = 'http://www.vybory.izbirkom.ru/region/izbirkom'
		payload = {
			'action'     : 'search_by_calendar',
			'start_date' : fromdate(start),
			'end_date'   : fromdate(end),
			'urovproved' : scope,
			'vidvibref'  : 'all', # FIXME multiple choice: all; 0 = Референдум; 1 = Выборы на должность; 2 = Выборы депутата; 3 = Отзыв депутата; 4 = Отзыв должностного лица
			'vibtype'    : 'all',
			'sxemavib'   : 'all',
			'region'     : '0', # FIXME whole country
		}
		res = await session.post(SEARCH, data=payload)
		doc = BeautifulSoup(res.text, PARSER)

		place = []
		for a in doc('a', class_='vibLink'):
			cell  = a.find_parent('tr').find('td')
			anode = cell.find('b')
			if anode is not None:
				place = ([normalize(anode.string)]
					 if anode.string != 'Российская Федерация'
					 else [])
			dnode = cell.find(string=True, recursive=False)
			if dnode is not None and dnode.strip():
				assert len(place) >= 1
				place = place[:1] + [dnode.strip()]

			yield cls(url=urladjust(urljoin(SEARCH, a['href']),
			                        sub_region=['99']),
			          title=normalize(a.string),
			          place=list(place))

# FIXME test code

from collections import defaultdict as DefaultDict
from os.path import exists
from random import seed, shuffle
from traceback import print_exc

async def main():
	w = get_terminal_size().columns
	params = {
#		'start': Date(2003,1,1),
#		'end': Date(2004,1,1),
#		'scope': Scope.COUNTRY,
	}
	filename = 'elections.jsonseq'

	async with Session(connections=25) as session:
#		elec = Election('http://www.vybory.izbirkom.ru/region/izbirkom?action=show&global=1&vrn=100100095619&region=0&prver=0&pronetvd=0&sub_region=99')
#		pprint(dict((await elec._single_result(session, '242'))._asdict()), width=w)
#		print()
#		pprint({k: dict(v._asdict()) for k, v in (await elec._aggregate_results(session, '233')).items()}, width=w)
#		return

		if not exists(filename):
			with open(filename, 'w', encoding='utf-8') as fp:
				async for e in Election.search(session, **params):
					dump([e.tojson()], fp, flush=True, ensure_ascii=False, indent=2)

		with open(filename, 'r') as fp:
			els = list(Election.fromjson(obj) for obj in load(fp))

		seed(57)
#		shuffle(els)

#		url = "http://www.vybory.izbirkom.ru/region/izbirkom?action=show&vrn=411401372131&region=11&prver=0&pronetvd=null&sub_region=99"
#		for i, e in enumerate(els):
#			if e.url == url: break
#		els = els[i:]

#		queue = Queue(0)
#		async def visit(comm):
#			await queue.put(await comm._result_types(session))
#			print(comm.title)

#		count = 0
#		types = DefaultDict(set)
#		tsets = set()
#		try:
#			async with open_nursery() as nursery:
#				for e in els:
#					nursery.start_soon(visit, e)
#					await sleep(0)
#				while nursery.child_tasks:
#					ts = await queue.get()
#					for k, v in ts.items(): types[k].add(v)
#					tsets.add(frozenset(ts.keys()))
#					count += 1
#		finally:
#			pprint(dict(types), width=w)
#			pprint([list(s) for s in tsets], width = w)
#			print(f'{count} of {len(els)}')

		for e in els:
			print(e.url, flush=True)
			while True:
				try: print(await e.date(session),
				           e.title,
				           await e.name(session),
				           #pformat(await e._result_types(session), width=w),
				           sep='\n', end='\n\n', flush=True)
				except Exception:
					print_exc()
					if input('Retry? [YN] ').lower().startswith('n'): break
				else:
					break
			#ts = await e._result_types(session)
			#assert (bool(len(ts) == 2*len([t for t in ts.values() if t.startswith("Сводн")])) !=
			#        (bool("еферендум" in e.title or "Опрос" in e.title) and not nodata(await e.page(session, '0'))))

#		root  = els[0]
#		queue = Queue(0)
#		async def walk(comm):
#			print('/'.join(await comm.path(session)),
#			      pformat(await comm._result_types(session), width=w),
#			      sep='\n', flush=True)
#			await queue.put(await comm.children(session))
#		async with open_nursery() as nursery:
#			nursery.start_soon(walk, root)
#			while nursery.child_tasks:
#				for comm in await queue.get():
#					nursery.start_soon(walk, comm)

if __name__ == '__main__':
	init_asks('trio')
	run(main)
