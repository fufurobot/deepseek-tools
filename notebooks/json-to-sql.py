#!/usr/bin/env python
# coding: utf-8

# In[1]:


# let's focus on deepseek data scheme first
from pathlib import Path

data_paths = list(Path("data").glob("deepseek*.zip"))


# In[2]:


# open as zip file
import zipfile
import random
# Open and read ZIP file
zip_ref = zipfile.ZipFile(random.choice(data_paths), 'r')
# List all files and folders
for file_info in zip_ref.infolist():
    print(f"{file_info.filename} - {file_info.file_size} bytes - compressed: {file_info.compress_size} bytes")

# Or just get file names
file_names = zip_ref.namelist()
print("Files:", file_names)


# In[3]:


# check our fake test data
test_json_list = list(Path("..").glob("**/*.json"))
test_json_list


# In[4]:


# let's build records recursively as a generator
# let's try user.json first, seems that is simple
# user_json_path = zip_ref.namelist()[0]
# import json
# user_json = json.load(zip_ref.open(user_json_path))
# display(user_json)
import json

user_json_path = Path('../tests/data/test_record_generator/test_foreign_key_with_increment_idgen/source_json/library.json')
user_json = json.load(user_json_path.open(mode="r"))

import typing

def increment_idgen(obj = None, state: int = 0):
    return state+1, state+1

def is_primitive(typ: type):
    if issubclass(typ, dict):
        return False
    elif issubclass(typ, list):
        return False
    else:
        return True

def python_object_id(obj = None, state = None):
    return state, id(obj)

def list_kv_getter(lst):
    return enumerate(lst)

def dict_kv_getter(dic):
    return dic.items()


def _process_primitive_obj_with_types(obj, kv_getter, *, primitive_check=is_primitive):
    primitive_kvs = []
    non_primitive_kvs = []
    type_info = "{obj_type}[{kv_list}]"
    for k,v in kv_getter(obj):
        typ_start = type(obj)
        if primitive_check(type(v)):
            primitive_kvs.append((k,v))
        else:
            non_primitive_kvs.append((k,v))
    kv_template = "{k}={v}"
    kv_list = ','.join(kv_template.format(k=k, v=type(v).__name__) for k,v in primitive_kvs) # FIXME we need to sort the keys to get a unique representation for type. otherwise, same type with different key order will be treated as different type, which is not what we want.
    # we leave foreign key parsing into the main parser.
    return type_info.format(obj_type=type(obj).__name__, kv_list=kv_list), primitive_kvs, non_primitive_kvs


class RecordGenerator:
    def __init__(self, id_generator=python_object_id):
        self.id_gen = id_generator
        self.state,self.current_id = id_generator()

    def next_id(self, obj):
        self.state,self.current_id = self.id_gen(obj, self.state)
        return self.current_id

    def traverse_and_edit_records(self, json_object, *, depth=0, foreign_key=False):
        if is_primitive(type(json_object)):
            pass # we don't want to touch the primitive type
        else:
            typ, primitive_kvs, non_primitive_kvs = _process_primitive_obj_with_types(json_object, dict_kv_getter if isinstance(json_object, dict) else list_kv_getter)
            if foreign_key:
                for k,v in non_primitive_kvs:
                    json_object[k] = {
                        "__type": "FOREIGN_KEY",
                        "__foreign_id": self.next_id(v),
                        "__foreign_id_gen": self.id_gen.__name__,
                        # "__value": v,
                    } # TODO extract editing into a private function.
            else:
                pass # do not touch the object, `edit` function is disabled.
            yield {
                "__id": self.next_id(json_object),
                "__id_gen": self.id_gen.__name__,
                "__type": typ, # TODO we will add foreign key processing later, since it requires a type naming system
                "__depth": depth,
                "__primitive_values": json_object, 
            }
            for _, v in non_primitive_kvs:
                yield from self.traverse_and_edit_records(v, depth=depth+1, foreign_key=foreign_key)

output_stem = user_json_path.stem
output_path = user_json_path.parent.parent / "expected_output_jsonl" /f"{output_stem}.jsonl"
with output_path.open(mode="w", encoding="utf-8") as f:
    for record in RecordGenerator(id_generator=increment_idgen).traverse_and_edit_records(user_json, foreign_key=True):
        line = json.dumps(record)
        print(line)
        print(line, file=f)



# In[16]:


import json
from typing import Optional, Dict, Tuple, Callable, Any, Iterator, Union
from pathlib import Path

# Assume the following external helpers exist (provided elsewhere):
# - is_primitive(type(obj)) -> bool
# - _process_primitive_obj_with_types(obj, kv_getter) -> (typ, primitive_kvs, non_primitive_kvs)
# - dict_kv_getter, list_kv_getter
# - python_object_id, increment_idgen (example ID generators)

class RecordGenerator:
    """
    Generates structured records from a JSON-like object, assigning IDs,
    optionally handling foreign keys, and supporting custom preprocessing.

    The generator traverses the object tree and yields a record for each
    non‑primitive container (dict or list). Each record contains:
      - __id          : unique identifier for the container
      - __id_gen      : name of the ID generator used
      - __type        : type string from _process_primitive_obj_with_types
      - __depth       : nesting depth
      - __primitive_values : the (possibly transformed) container

    If `foreign_key=True` is passed to `traverse_and_edit_records`, any
    non‑primitive values inside a dict are replaced with a foreign‑key
    wrapper (the sub‑object is still traversed separately).

    A preprocessing pipeline can be injected to modify dicts before they are
    processed, enabling type‑specific normalisation or polymorphism handling.
    """

    def __init__(
        self,
        id_generator: Callable[..., Tuple[Any, Any]] = python_object_id,
        preprocessing_type_handler: Optional[
            Dict[
                str,
                Tuple[
                    Callable[[Dict], bool],   # predicate: should this handler apply?
                    Callable[[Dict], Dict]    # transformer: mutates or returns a new dict
                ]
            ]
        ] = None
    ) -> None:
        """
        Initialise the RecordGenerator.

        Parameters
        ----------
        id_generator : callable, optional
            A function that returns a (state, id) tuple. The callable may be
            invoked as `id_generator()` for initialisation, and later as
            `id_generator(obj, state)` to produce the next ID.
            Defaults to `python_object_id`.

        preprocessing_type_handler : dict, optional
            A mapping from a type name (for diagnostics) to a tuple of two
            callables:
              - predicate(obj: dict) -> bool
              - transformer(obj: dict) -> dict
            For each dict encountered during traversal, the predicates are
            evaluated in the dict's key insertion order (i.e., the order in
            which they were defined). The first matching predicate triggers
            its transformer. The transformer may modify the dict in place
            or return a new dict (which will replace the original).
            This is useful for normalising polymorphic structures before
            they are processed further.
        """
        self.id_gen = id_generator
        self.state, self.current_id = id_generator()
        self.preprocessing_handler = preprocessing_type_handler

    def next_id(self, obj: Any) -> Any:
        """
        Advance the ID generator and return the next ID for the given object.

        Parameters
        ----------
        obj : Any
            The object for which an ID is requested (may be used by some generators).

        Returns
        -------
        Any
            The newly generated ID.
        """
        self.state, self.current_id = self.id_gen(obj, self.state)
        return self.current_id

    def _apply_preprocessing(self, obj: Dict) -> Dict:
        """
        Apply the preprocessing pipeline to a dict, if any handler matches.

        Handlers are tested in the order they appear in the handler dictionary.
        The first matching predicate's transformer is invoked. The transformer
        may mutate the dict in place or return a new dict; if it returns a new
        dict, that dict replaces the original.

        Parameters
        ----------
        obj : dict
            The dict to preprocess.

        Returns
        -------
        dict
            The (possibly transformed) dict.
        """
        if self.preprocessing_handler is None:
            return obj

        # Iterate in dict key order (preserved as of Python 3.7)
        for type_name, (predicate, transformer) in self.preprocessing_handler.items():
            if predicate(obj):
                transformed = transformer(obj)
                # If transformer returns a new dict, use it; otherwise assume in‑place mutation
                if transformed is not None and transformed is not obj:
                    obj = transformed
                break  # stop after first match (polymorphism)
        return obj

    def traverse_and_edit_records(
        self,
        json_object: Any,
        *,
        depth: int = 0,
        foreign_key: bool = False
    ) -> Iterator[Dict[str, Any]]:
        """
        Traverse the JSON‑like object and yield record dictionaries.

        Parameters
        ----------
        json_object : Any
            The root object (dict, list, or primitive).

        depth : int, optional
            Current nesting depth (used internally). Default 0.

        foreign_key : bool, optional
            If True, any non‑primitive value inside a dict is replaced with a
            foreign‑key wrapper (`__type`, `__foreign_id`, etc.). The wrapped
            object is still traversed separately. Default False.

        Yields
        ------
        dict
            A record containing metadata and the (possibly transformed) object.

        Notes
        -----
        - Primitive values (int, float, str, bool, None) are left untouched.
        - For dicts, the preprocessing pipeline is applied before any other
          processing (including foreign‑key replacement).
        - The generated record's `__primitive_values` field holds the object
          after preprocessing and after foreign‑key substitution (if any).
        - The record also includes `__id`, `__id_gen`, `__type`, and `__depth`.
        """
        # ---------- Skip primitives ----------
        if is_primitive(type(json_object)):
            return

        # ---------- Apply preprocessing to dicts ----------
        if isinstance(json_object, dict):
            json_object = self._apply_preprocessing(json_object)

        # ---------- Analyse structure ----------
        kv_getter = dict_kv_getter if isinstance(json_object, dict) else list_kv_getter
        typ, primitive_kvs, non_primitive_kvs = _process_primitive_obj_with_types(
            json_object, kv_getter
        )

        # ---------- Foreign‑key substitution (only for dicts) ----------
        if foreign_key and isinstance(json_object, dict):
            for k, v in non_primitive_kvs:
                # Replace the non‑primitive value with a foreign‑key reference
                json_object[k] = {
                    "__type": "FOREIGN_KEY",
                    "__foreign_id": self.next_id(v),
                    "__foreign_id_gen": self.id_gen.__name__,
                    # "__value": v,  # commented out to reduce size
                }
            # Re‑extract non‑primitive kvs? No, we keep the original list for traversal,
            # but we already mutated the dict; the traversal below will still process the
            # original `v` objects, not the wrapper. This is fine because we want to
            # recurse into the original non‑primitive values.

        # ---------- Yield record for the current object ----------
        yield {
            "__id": self.next_id(json_object),
            "__id_gen": self.id_gen.__name__,
            "__type": typ,
            "__depth": depth,
            "__primitive_values": json_object,
        }

        # ---------- Recurse into non‑primitive children ----------
        for _, v in non_primitive_kvs:
            yield from self.traverse_and_edit_records(
                v,
                depth=depth + 1,
                foreign_key=foreign_key
            )


output_stem = user_json_path.stem
output_path = user_json_path.parent.parent / "expected_output_jsonl" /f"{output_stem}.jsonl"
with output_path.open(mode="w", encoding="utf-8") as f:
    for record in RecordGenerator(id_generator=increment_idgen).traverse_and_edit_records(user_json, foreign_key=True):
        line = json.dumps(record)
        print(line)
        print(line, file=f)


# In[5]:


is_primitive(type([1,2,3]))


# In[6]:


str(type({}))


# In[7]:


type({}).__name__


# In[8]:


# test conversation
def conversation_json():
    return json.load(zip_ref.open(zip_ref.namelist()[1]))

import time
for record in RecordGenerator().traverse_and_edit_records(conversation_json(), foreign_key=True):
    print("--- BEGIN RECORD ---")
    print(record)
    print("--- END RECORD ---")
    break


# In[9]:


get_ipython().run_line_magic('pinfo', 'set.')


# In[10]:


# get unique dict types
import time
type_set = set()
import tqdm
for record in RecordGenerator().traverse_and_edit_records(conversation_json(), foreign_key=True):
    record_typ = record['__type']
    if not record_typ.startswith("list") and record_typ not in type_set:
        type_set.add(record_typ)
        print(f"--- BEGIN RECORD WITH NEW TYPE {record_typ}---")
        print(record)
        print(f"--- END RECORD WITH NEW TYPE {record_typ}---")
type_set


# In[11]:


# next step: refactor the `__type` field as a object oriented design: with good internal structure, can be serialized and deserialized from and to its string representation. 

# we need a `SchemaCollector` to work on RecordGenerator to parse and deduce actual types.

# a `SchemaCollector` will parse primitives from the record, check non-primitive foreign keys and add as a directed edge (adjancent table for schema object). 

# a schema object contains all primitive types and all links to their foreign keys. while serializing, it considers their link recursively.

# next step: add foreign key into type name to distinguish more.

# next step: hash the primitive value with foreign key IDs in a determistic way to get ID for the data, like GIT did. we can use layered hash: first crc32 in zlib, then sha256 in hashlib.


# In[12]:


# very first step: merge similar dict types (with same keys, we allow NoneType to merge with other type (treat it as not exists), and allow hierarchy of types (we check the frontier set that not be contained by other sets))
# here, we just collect all keys naively, no matter their value are primitive or not. we don't even need full collector for now, just a collector that find frontier set of keys that can be different types.
# we don't consider list for now. later, we have list homogenious detection and merge. if a list is not homogenious, we treat it as a namedtuple(column_0...column_n), or a dict.
# for each primitive typed homogenious list, we use one table for each primitive type and use foreign key to keep the reference of the list.

def find_frontier_sorted(sets):
    sorted_sets = sorted(sets, key=len, reverse=True)
    frontier = []
    for s in sorted_sets:
        if not any(s.issubset(f) for f in frontier):
            frontier.append(s)
    return frontier
# we use a naive schema collector that only collect frozenset as key set of the __value of the record. 
# we use a set to collect and deduplicate that frozenset. after the set is built, we `find_frontier_sorted` sets to output the sorted unique keys as schema keys.


# In[13]:


def collect_schema_key_frozenset(record_stream):
    collected_sets = set()
    for record in record_stream:
        value = record["__primitive_values"]
        if isinstance(value, dict):
            keys = frozenset(value.keys())
            if keys not in collected_sets:
                yield keys
                collected_sets.add(keys)


# In[14]:


schemas = list(collect_schema_key_frozenset(RecordGenerator().traverse_and_edit_records(conversation_json(), foreign_key=True)))
schemas


# In[15]:


# seems that wee need to merge numbered function into a judger. we need a frozen dict (which is just a frozenset of tuples)
schema_template = """
{{
 "type": {class_name},
 "optional": {is_optional},
 "keys": {list_of_keys},
 "judge": {python_expr_returns_a_callable_to_judge_whether_parameter_belongs_to_the_type}
}}
"""

def collect_schema_key_frozenset(record_stream):
    collected_sets = set()
    for record in record_stream:
        value = record["__primitive_values"]
        if isinstance(value, dict):
            keys = frozenset(value.keys())
            if keys not in collected_sets:
                yield keys
                collected_sets.add(keys)

def always_false(x):
    return False

def basic_parse(schemas, record_stream):
    schemas = list(schemas)
    schema_dicts = [eval(schema_template.format(
        class_name = None,
        is_optional = False,
        keys = schema
        python_expr_returns_a_callable_to_judge_whether_parameter_belongs_to_the_type = always_false
    )) for schema in schemas]
    for record in record_stream:
        for schema_index, schema in enumerate(schemas):
            schema_dicts[schema_index] = tet
            # try to match schema from the record; if matched




# In[ ]:


# it is convenient that the schema collector keeps all the original records in memory, but for now, we only keeps a value range for each key: the key may get different value range from different types. 
# for the numbers, we check min and max. for strings we check min and max len.
# for foreign keys, it can be tricky. it can only be computed after our frontier type names are generated.
# we know how many kinds of frontier schema keys is referenced under that name. 


# In[ ]:


import inspect
inspect.signature(_process_primitive_obj_with_types)


# In[ ]:


test_p = list(Path("..").glob("**/*.py"))[0]


# In[ ]:


get_ipython().run_line_magic('pinfo', 'test_p.open')


# In[ ]:




