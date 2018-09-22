from asks            import Session
from asks.errors     import AsksException
from bs4             import BeautifulSoup, Tag
from collections     import OrderedDict, namedtuple
from collections.abc import Iterable
from datetime        import date as Date, datetime as DateTime
from enum            import Enum
from jsonseq         import dump, load
from trio            import Queue, open_nursery, run, sleep
from urllib.parse    import parse_qs, urlencode, urljoin, urlsplit, urlunsplit

PARSER = 'html5lib'

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
		pass

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
		return '{}(url={!r}, ppath={!r}, hints={!r})'.format(
			type(self).__qualname__,
			self.url, self._ppath, self._hints)

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
			enc = (res.encoding.lower()
			       if 'charset' in res.headers.get('content-type')
			       else None) # FIXME unreliable (change in asks?)
			self._page[type] = BeautifulSoup(
				res.content, PARSER, from_encoding=enc)
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
		return OrderedDict(
			(parse_qs(urlsplit(a['href']).query)['type'][0],
			 normalize(a.string))
			for a in ancs)

	@staticmethod
	def _single_table(table):
		rows = [tr('td') for tr in table('tr')]
		assert all(len(row) in {2, 3} for row in rows)
		seps = [i for i, row in enumerate(rows) if len(row) == 2]
		data = [Row(number=normalize(row[0].string),
		            name=normalize(row[1].string),
		            value=int(row[2].find(string=True).string))
		        for row in rows if len(row) == 3]
		return seps, data

	@classmethod
	def _single(cls, page):
		tabs = page(cellpadding='2')
		assert len(tabs) == 1
		seps, data = cls._single_table(tabs[0])
		return Report(records=data[:seps[0]], results=data[seps[0]:])

	@classmethod
	def _aggregate(cls, page):
		tabs = page(cellpadding='2')
		assert len(tabs) == 2

		# Left table contains headers
		seps, head = cls._single_table(tabs[0])
		assert seps[0] == 0 # Column titles

		# Right table contains data per commission
		rows = [tr('td') for tr in tabs[1]('tr')]
		data = (row for i, row in enumerate(rows) if i not in seps)
		comms = (normalize(td.find('a').string) for td in rows[0])
		datas = ([h._replace(value=int(v.find(string=True).string))
		          for h, v in zip(head, col)]
		         for col in zip(*data))

		# Extra separator in seps[0] means indices are offset by 1
		return OrderedDict(
			(c, Report(records=d[: seps[1]-1],
			           results=d[seps[1]-1 :]))
			for c, d in zip(comms, datas))

	@property
	def level(self):
		return len(self._ppath)

	async def name(self, session):
		page = await self.page(session, '0') # FIXME Any cached page would work
		crumbs = page.find('table', height='80%').find('td')('a')
		# If the crumbs are absent, or if one of the crumbs is the
		# empty string, we can't be sure we got them right, so use
		# the fallback method.
		if len(crumbs)-1 == self.level:
			# NB. When the crumbs are missing from the parent, the
			# text in crumbs[:-1] _can_ differ from self._ppath.
			return normalize(crumbs[-1].string)

		page = await self.page(session, '0')
		caption = page.find(string=disj(
			matches('Наименование комиссии'),
			matches('Наименование избирательной комиссии')))
		if caption is None:
			assert nodata(page)
			return None
		return normalize(caption.find_parent('td')
		                        .find_next_sibling('td')
		                        .string)

	async def path(self, session):
		return self._ppath + [await self.name(session)]

	async def children(self, session):
		if self._children is None:
			page = await self.page(session, '0') # FIXME Any cached page would work
			self._children = \
				[Commission(urljoin(self.url, o['value']),
				            await self.path(session),
				            self._hints)
				 for o in page('option')
				 if o.attrs.get('value')]
			if not self._children:
				assert page.find(string=matches(
					"нет отчета по навигации или же это "
					"конечный твд == уик"))
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
		return '{}(url={!r}, title={!r}, place={!r})'.format(
			type(self).__qualname__,
			self.url, self.title, self.place)

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

from asks import init as init_asks
from collections import defaultdict as DefaultDict
from os.path import exists
from pprint import pprint, pformat
from random import seed, shuffle
from shutil import get_terminal_size
from sys import stdout, stderr
from textwrap import shorten
from traceback import print_exc
from trio import WouldBlock

def report(done, pending, last):
	global lastrep
	message = (('{done} done, {pending} pending: {last}'
	            if last is not None
	            else '{done} done, {pending} pending')
	           .format(done=done, pending=pending, last=last))
	print('\r\033[K' + shorten(message, width=w, placeholder='...'),
	      end='', file=stderr, flush=True)
def clear():
	print('\r\033[K', end='', file=stderr, flush=True)

async def collect_types(session, els):
	queue = Queue(0)
	count = 0
	types = DefaultDict(set)
	tsets = set()
	last  = None

	async def visit(comm):
		nonlocal last
		types = await comm._result_types(session)
		last = comm.title
		await queue.put(types)

	async def starter(nursery):
		for e in els:
			nursery.start_soon(visit, e)
			await sleep(0)

	try:
		async with open_nursery() as nursery:
			nursery.start_soon(starter, nursery)
			while nursery.child_tasks:
				try:
					ts = queue.get_nowait()
				except WouldBlock:
					pass
				else:
					for k, v in ts.items(): types[k].add(v)
					tsets.add(tuple(ts.keys()))
					count += 1
				await sleep(0)
				report(count, len(nursery.child_tasks), last)
	finally:
		clear()
		pprint(dict(types), width=w)
		pprint(list(map(list, tsets)), width=w)
		print(f'{count} of {len(els)}')

async def traverse(session, root):
	queue = Queue(0)
	done  = 0
	last  = None

	async def visit(comm):
		nonlocal done, last
		try:
			await queue.put(await comm.children(session))
			last = '/'.join(await comm.path(session))
		except Exception as e:
			print(f'Error processing <{comm.url}>: {e}', file=stderr)
			print_exc()
			stderr.flush()
			raise
		done += 1

	async with open_nursery() as nursery:
		nursery.start_soon(visit, root)
		while nursery.child_tasks:
			report(done, len(nursery.child_tasks), last)
			try:
				more = queue.get_nowait()
			except WouldBlock:
				pass
			else:
				for comm in more:
					nursery.start_soon(visit, comm)
			await sleep(0)
		clear()

async def main():
	global w
	w = get_terminal_size().columns
	params = {
#		'start': Date(2003,1,1),
#		'end': Date(2004,1,1),
#		'scope': Scope.COUNTRY,
	}
	filename = 'elections.jsonseq'

	async with Session(connections=25) as session:
		if not exists(filename):
			with open(filename, 'w', encoding='utf-8') as fp:
				async for e in Election.search(session, **params):
					dump([e.tojson()], fp, flush=True, ensure_ascii=False, indent=2)

		with open(filename, 'r') as fp:
			els = list(Election.fromjson(obj) for obj in load(fp))

		seed(57)
		shuffle(els)

		await collect_types(session, els)

#		elec = els[-2]
#		pprint(dict(elec._single(await elec.page(session, '226'))._asdict()), width=w)
#		print()
#		pprint({k: dict(v._asdict()) for k, v in elec._aggregate(await elec.page(session, '227')).items()}, width=w)
#		return

#		url = "http://www.vybory.izbirkom.ru/region/izbirkom?action=show&vrn=411401372131&region=11&prver=0&pronetvd=null&sub_region=99"
#		for i, e in enumerate(els):
#			if e.url == url: break
#		els = els[i:]

		# await traverse(session, els[-1])

if __name__ == '__main__':
	init_asks('trio')
	run(main)
