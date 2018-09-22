from __future__ import unicode_literals

from io import DEFAULT_BUFFER_SIZE
import json

__all__ = ['dump', 'load']

RS = '\x1E'

def dump(iterable, fp, *, flush=False, **named):
	"""Dump iterable to fp as a JSON text sequence.

	Concatenation of JSON text sequences encodes the concatenation
	of the object lists they encode. Thus one can write a JSON text
	sequence file in several batches by calling this function
	several times.

	If flush is set to True, the file is flushed after every object.
	Any remaining keyword arguments are passed to json.dumps.
	"""
	for obj in iterable:
		print(RS + json.dumps(obj, **named), file=fp, flush=flush)

def records(fp, buffer_size=DEFAULT_BUFFER_SIZE):
	"""Iterate over RS-introduced records in fp."""

	buf = fp.read(buffer_size)
	if not buf:
		return # Empty file
	if buf[0] != RS:
		raise IOError('JSON text sequence does not start with RS')
	buf = buf[1:]

	while True:
		new = fp.read(buffer_size)
		if not new:
			yield from buf.split(RS)
			break
		# FIXME quadratic complexity in chunks per object
		recs = (buf + new).split(RS)
		yield from recs[:-1]
		buf = recs[-1]

def load(fp, buffer_size=DEFAULT_BUFFER_SIZE, **named):
	"""Iterate over the objects in JSON text sequence file fp.

	The file is read in chunks of size buffer_size, so it should be
	set to the typical object size when objects are large. Any
	remaining keyword arguments are passed to json.loads.
	"""
	for rec in records(fp, buffer_size):
		yield json.loads(rec, **named)
