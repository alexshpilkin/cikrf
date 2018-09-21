from asks            import Session
from asks.errors     import AsksException
from bs4             import BeautifulSoup, Tag
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

def some(*matches):
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
	def __getattr__(self, name):
		assert name != '__dict__'
		self.__dict__[name] = None
		return self.__dict__[name]

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
				else:
					if res.status_code // 100 == 2: break
					else: print(res.status_code, url, res.reason_phrase, flush=True) # FIXME
				await sleep(0) # FIXME back off?
			self._page[type] = BeautifulSoup(res.text, PARSER)
		else:
			await sleep(0)
		return self._page[type]

	async def result_types(self, session):
		page = await self.page(session, '0')
		capt = page.find(string=matches('Результаты выборов'))
		if capt is None:
			# NB: In general, nodata(page) is false: there _are_
			# (local) elections with no results.
			return dict()

		ancs = (row.find('a')
		        for row in capt.find_parent('tr').next_siblings
		        if isinstance(row, Tag))
		return {parse_qs(urlsplit(a['href']).query)['type'][0]:
		        str(a.string) for a in ancs} # FIXME

	async def _directory(self, session):
		# FIXME don't actually need the directory (most of the time)
		return await self.page(session, '0')

	async def name(self, session):
		page = await self._directory(session)
		capt = page.find(string=some(matches('Наименование комиссии'),
		                             matches('Наименование Избирательной комиссии')))
		return normalize(doc.find('table', height='80%').find('td')
		                    .find_all('a')[-1].string)

	async def path(self, session):
		return self._ppath + [await self.name(session)]

	async def children(self, session):
		if self._children is None:
			opts = (await self._directory(session))('option')
			self._children = \
				[Commission(urljoin(self.url, o['value']),
				            await self.path(session),
				            self._hints)
				 for o in opts if o.attrs.get('value')]
		else:
			await sleep(0)
		return self._children

#	async def _collect(self, session, path, url):
#		trace('LIST ' + '/'.join(path))
#		yield Link(url, path)
#		async with session.get(url) as res:
#			soup = BeautifulSoup(await res.text(), PARSER)
#		trace('DOWN ' + '/'.join(path))
#		children = (self._collect(session,
#		                          path + [maybe(normalize, o.string)],
#		                          o['value'])
#		            for o in soup('option') if o.attrs.get('value'))
#		async with merge(*children).stream() as stream:
#			async for link in stream:
#				yield link
#		trace('DONE ' + '/'.join(path))

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

from traceback import print_exc

async def main():
	w = get_terminal_size().columns
	params = {
#		'start': Date(2003,1,1),
#		'end': Date(2004,1,1),
#		'scope': Scope.COUNTRY,
	}

	async with Session(connections=25) as session:
#		els = [e async for e in Election.search(session, **params)]

#		with open('elections.jsonseq', 'w', encoding='utf-8') as fp:
#			async for e in Election.search(session, **params):
#				dump([e.tojson()], fp, flush=True, ensure_ascii=False, indent=2)
#		return

		with open('elections.jsonseq', 'r') as fp:
			els = list(Election.fromjson(obj)
			           for obj in load(fp))

#		pprint(els, width=80)
#		return

		url = "http://www.vybory.izbirkom.ru/region/izbirkom?action=show&vrn=477404472842&region=77&prver=0&pronetvd=null&sub_region=99"
		for i, e in enumerate(els):
			if e.url == url: break
		els = els[i:]

		for e in els:
			print(e.url, flush=True)
			while True:
				try: print(await e.date(session),
				      e.title,
				      pformat(await e.result_types(session), width=w),
				      sep='\n', end='\n\n', flush=True)
				except Exception:
					print_exc()
					input()
				else:
					break
		return

		root  = els[0]
		queue = Queue(0)
		async def walk(comm):
			print('/'.join(await comm.path(session)),
			      pformat(await comm.result_types(session), width=w),
			      sep='\n', flush=True)
			await queue.put(await comm.children(session))
		async with open_nursery() as nursery:
			nursery.start_soon(walk, root)
			while nursery.child_tasks:
				for comm in await queue.get():
					nursery.start_soon(walk, comm)

init_asks('trio')
run(main)
