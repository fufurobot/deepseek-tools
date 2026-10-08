import datetime
import pathlib
from typing import Any, Dict, Callable, Optional

import base64
from typing import Any, Callable, Dict, Optional, Type


def object_to_json(
    obj: Any,
    append_pyid: bool = True,
    append_pytype: bool = True,
    append_pymodulepath: bool = False,
    resolve_ref: bool = True,
    use_autoincrement: bool = False,
    preprocessing_type_handler: Optional[Dict[Type, Callable[[Any], Any]]] = None,
    primitive_judge: Optional[Callable[[Any], bool]] = None,
    use_base64: bool = True,
    **universal_keys: Callable[[Any], Any]
) -> Any:
    """
    Convert an arbitrary Python object into a JSON‑serializable structure.

    Parameters
    ----------
    obj : Any
        The object to convert.
    append_pyid : bool, optional
        If True, add a ``"pyid"`` field containing ``id(obj)`` (or an auto‑increment
        id if ``use_autoincrement`` is True). Default True.
    append_pytype : bool, optional
        If True, add a ``"pytype"`` field with the class name. Default True.
    append_pymodulepath : bool, optional
        If True, add a ``"pymodulepath"`` field with the module of the class.
        Default False.
    resolve_ref : bool, optional
        If True and ``append_pyid`` is also True, repeated objects are represented
        as ``{"ref": <id>}`` to break cycles. Default True.
    use_autoincrement : bool, optional
        If True and ``append_pyid`` is True, assign sequential integers instead of
        raw memory addresses. Default False.
    preprocessing_type_handler : dict, optional
        Maps a type to a callable. If an object's type matches, the callable is
        invoked first and the returned object is processed instead.
    primitive_judge : callable, optional
        A function that takes an object and returns True if it should be treated
        as a primitive. Defaults to isinstance check for int, float, str, bytes,
        bool, and None.
    use_base64 : bool, optional
        If True, bytes are encoded as base64 strings. Default True.
    **universal_keys : callable
        For each keyword argument, the callable is invoked with the original
        object and its return value is added to the output dictionary under the
        given key. Only applied to non‑primitive objects.

    Returns
    -------
    Any
        A JSON‑serializable structure (dict, list, str, int, float, bool, None).
    """

    # Default primitive judge
    if primitive_judge is None:
        def primitive_judge(x: Any) -> bool:
            return isinstance(x, (int, float, str, bytes, bool)) or x is None

    # Preprocessing handler
    if preprocessing_type_handler is None:
        preprocessing_type_handler = {}

    # State for cycle handling and auto‑increment ids
    visited: Dict[int, int] = {}      # raw id -> assigned id (if use_autoincrement)
    autoincrement_counter = 0

    def get_assigned_id(x: Any) -> int:
        """Return the id to use for object x, assigning a new one if needed."""
        nonlocal autoincrement_counter
        raw_id = id(x)
        if use_autoincrement:
            if raw_id not in visited:
                visited[raw_id] = autoincrement_counter
                autoincrement_counter += 1
            return visited[raw_id]
        else:
            # using raw id; still store it so we can detect cycles
            if raw_id not in visited:
                visited[raw_id] = raw_id
            return raw_id

    def convert(x: Any) -> Any:
        # 1. Apply preprocessing if applicable
        handler = preprocessing_type_handler.get(type(x))
        if handler:
            x = handler(x)

        # 2. Primitive handling
        if primitive_judge(x):
            if isinstance(x, bytes) and use_base64:
                return base64.b64encode(x).decode('ascii')
            return x

        # 3. Non‑primitive: check for already visited (cycles)
        if resolve_ref and append_pyid:
            raw_id = id(x)
            if raw_id in visited:
                # Return a reference marker
                return {"ref": visited[raw_id]}
            # First time: assign an id (will be stored in visited)
            assigned_id = get_assigned_id(x)
        else:
            # Even if not resolving refs, we may still need ids for metadata
            if append_pyid:
                assigned_id = get_assigned_id(x)
            else:
                # No id needed, but we still need to register the object
                # if we want to avoid cycles later? We'll ignore if resolve_ref=False.
                pass

        # 4. Build the output dictionary
        out: Dict[str, Any] = {}
        if append_pyid:
            out["pyid"] = assigned_id
        if append_pytype:
            out["pytype"] = type(x).__name__
        if append_pymodulepath:
            out["pymodulepath"] = type(x).__module__

        # Add universal keys
        for key, func in universal_keys.items():
            out[key] = func(x)

        # 5. Determine the actual data representation
        if isinstance(x, dict):
            data = {}
            for k, v in x.items():
                key_str = str(k) if not isinstance(k, str) else k
                data[key_str] = convert(v)
            out["value"] = data
        elif isinstance(x, (list, tuple)):
            out["value"] = [convert(item) for item in x]
        elif isinstance(x, (set, frozenset)):
            # Sort for deterministic output
            sorted_items = sorted(x, key=lambda item: str(item))
            out["value"] = [convert(item) for item in sorted_items]
        else:
            # Custom object: use __dict__ if available
            if hasattr(x, "__dict__"):
                data = {}
                for k, v in x.__dict__.items():
                    data[k] = convert(v)
                out["value"] = data
            else:
                # Fallback: string representation
                out["value"] = str(x)

        # 6. If no metadata was added and resolve_ref is inactive, we can
        #    return the plain data without the wrapper.
        if not (append_pyid or append_pytype or append_pymodulepath or universal_keys):
            if not (resolve_ref and append_pyid):
                # No wrapper needed
                return out["value"]

        return out

    return convert(obj)

def object_to_dict(
    obj: Any,
    primitive: bool = True,
    public: bool = True,
    try_call: bool = False,
    **keys: Callable[[Any], Any]
) -> Dict[str, Any]:
    """
    Convert any object to a dictionary.
    
    Args:
        obj: The object to convert
        primitive: If True, ignore non-primitive values
        public: If True, ignore attributes starting with '_'
        try_call: If True and primitive is True, try to call callables with self
        **keys: Custom key-value pairs where value is a function taking the object
    
    Returns:
        Dictionary representation of the object
    """
    # Define primitive types
    primitive_types = (
        bool, int, complex, str, float, bytes,
        datetime.datetime, pathlib.Path
    )
    
    result = {}
    
    # Get all attributes using dir
    for attr_name in dir(obj):
        # Skip special attributes if public is True
        if public and attr_name.startswith('_'):
            continue
        
        # Skip if this is a method/attribute that starts with '__' (already handled by public)
        if public and attr_name.startswith('__') and attr_name.endswith('__'):
            continue
        
        try:
            value = getattr(obj, attr_name)
        except Exception:
            # Skip attributes that can't be accessed
            continue
        
        # Handle primitive filtering
        if primitive:
            # Check if value is a primitive type
            is_primitive = isinstance(value, primitive_types)
            
            # If not primitive, check if it's a callable and try_call is True
            if not is_primitive and try_call and callable(value):
                try:
                    # Try to call the method with self
                    called_value = value()
                    # Check if the result is primitive
                    if isinstance(called_value, primitive_types):
                        value = called_value
                        is_primitive = True
                except Exception:
                    # Skip if calling fails
                    continue
            
            # Skip if still not primitive
            if not is_primitive:
                continue
        
        # Add to result
        result[attr_name] = value
    
    # Apply custom key transformations
    for key, func in keys.items():
        try:
            result[key] = func(obj)
        except Exception:
            # Skip if function fails
            pass
    
    return result


import datetime
import pathlib
from typing import Any, Dict, Callable, Optional

def object_to_dict(
    obj: Any,
    primitive: bool = True,
    public: bool = True,
    try_call: bool = False,
    flatten_dict_sep: Optional[str] = None,
    flatten_list_sep: Optional[str] = None,
    **keys: Callable[[Any], Any]
) -> Dict[str, Any]:
    """
    Convert any object to a dictionary.
    
    Args:
        obj: The object to convert
        primitive: If True, ignore non-primitive values
        public: If True, ignore attributes starting with '_'
        try_call: If True and primitive is True, try to call callables with self
        flatten_dict_sep: If not None and primitive is True, flatten dict attributes
                         using this separator between attribute name and dict key.
                         Only primitive dict values are included.
        flatten_list_sep: If not None and primitive is True, flatten list/tuple
                         attributes using this separator between attribute name
                         and zero‑based index. Only primitive elements are included.
        **keys: Custom key-value pairs where value is a function taking the object
    
    Returns:
        Dictionary representation of the object
    """
    # Define primitive types
    primitive_types = (
        bool, int, complex, str, float, bytes,
        datetime.datetime, pathlib.Path
    )
    
    def is_primitive(val: Any) -> bool:
        return isinstance(val, primitive_types)
    
    result = {}
    
    # Get all attributes using dir
    for attr_name in dir(obj):
        # Skip special attributes if public is True
        if public and attr_name.startswith('_'):
            continue
        
        # Skip if this is a method/attribute that starts with '__' (already handled by public)
        if public and attr_name.startswith('__') and attr_name.endswith('__'):
            continue
        
        try:
            value = getattr(obj, attr_name)
        except Exception:
            # Skip attributes that can't be accessed
            continue
        
        # Handle try_call for callables
        if primitive and try_call and callable(value):
            try:
                # Try to call the method with self
                called_value = value()
                # Check if the result is primitive
                if is_primitive(called_value):
                    value = called_value
                else:
                    # If not primitive, keep original? Actually we might want to try flattening
                    # on the called value if it's a container? But the instruction doesn't
                    # specify that; we keep the called value as is and continue normal processing.
                    # However, if called_value is not primitive, we might still flatten it if
                    # it's a dict/list. So we set value = called_value (even if not primitive)
                    # and let the flattening logic handle it if applicable.
                    value = called_value
            except Exception:
                # Skip if calling fails
                continue
        
        # --- Flattening logic ---
        # Only process flattening when primitive is True and separators are provided.
        if primitive:
            # Check for dict-like objects (have .items())
            if flatten_dict_sep is not None and hasattr(value, 'items') and callable(value.items):
                # Flatten dict: iterate over items, keep only primitive values
                try:
                    for sub_key, sub_value in value.items():
                        if is_primitive(sub_value):
                            new_key = f"{attr_name}{flatten_dict_sep}{sub_key}"
                            result[new_key] = sub_value
                    # Skip adding the original attribute
                    continue
                except Exception:
                    # If iteration fails, treat as non-flattenable and fall through
                    pass
            
            # Check for list/tuple (enumerable)
            if flatten_list_sep is not None and isinstance(value, (list, tuple)):
                # Flatten list/tuple: iterate by index, keep only primitive values
                for idx, sub_value in enumerate(value):
                    if is_primitive(sub_value):
                        new_key = f"{attr_name}{flatten_list_sep}{idx}"
                        result[new_key] = sub_value
                # Skip adding the original attribute
                continue
        
        # --- Normal primitive filtering ---
        if primitive:
            if not is_primitive(value):
                continue
        
        # Add to result
        result[attr_name] = value
    
    # Apply custom key transformations
    for key, func in keys.items():
        try:
            result[key] = func(obj)
        except Exception:
            # Skip if function fails
            pass
    
    return result


import datetime
import pathlib
from typing import Any, Dict, Callable, Optional, Union, Iterator, Tuple


# ---------- helpers ----------
def _is_primitive(val: Any) -> bool:
    """Return True if val is a primitive type (immutable and simple)."""
    primitive_types = (
        bool, int, complex, str, float, bytes,
        datetime.datetime, pathlib.Path
    )
    return isinstance(val, primitive_types)


def _flatten_container(
    value: Any,
    prefix: str,
    dict_sep: Optional[str],
    list_sep: Optional[str],
    remaining_depth: int
) -> Iterator[Tuple[str, Any]]:
    """
    Recursively traverse a dict/list container and yield (key, value) for every
    primitive leaf found. The traversal stops when remaining_depth reaches 0.
    """
    if isinstance(value, dict) and dict_sep is not None:
        for k, v in value.items():
            new_prefix = f"{prefix}{dict_sep}{k}"
            if _is_primitive(v):
                yield new_prefix, v
            elif remaining_depth > 0 and (isinstance(v, dict) or isinstance(v, (list, tuple))):
                yield from _flatten_container(v, new_prefix, dict_sep, list_sep, remaining_depth - 1)
            # else: skip non‑primitive nested containers when depth exhausted

    elif isinstance(value, (list, tuple)) and list_sep is not None:
        for idx, v in enumerate(value):
            new_prefix = f"{prefix}{list_sep}{idx}"
            if _is_primitive(v):
                yield new_prefix, v
            elif remaining_depth > 0 and (isinstance(v, dict) or isinstance(v, (list, tuple))):
                yield from _flatten_container(v, new_prefix, dict_sep, list_sep, remaining_depth - 1)
            # else: skip


# ---------- main function ----------
def object_to_dict(
    obj: Any,
    primitive: bool = True,
    public: bool = True,
    try_call: bool = False,
    flatten_dict_sep: Optional[str] = None,
    flatten_list_sep: Optional[str] = None,
    max_depth: int = 0,                         # <-- new parameter
    **keys: Callable[[Any], Any]
) -> Dict[str, Any]:
    """
    Convert any object to a dictionary.

    Args:
        obj: The object to convert
        primitive: If True, ignore non-primitive values
        public: If True, ignore attributes starting with '_'
        try_call: If True and primitive is True, try to call callables with self
        flatten_dict_sep: If not None and primitive is True, flatten dict attributes
                         using this separator between attribute name and dict key.
                         Only primitive dict values are included.
        flatten_list_sep: If not None and primitive is True, flatten list/tuple
                         attributes using this separator between attribute name
                         and zero‑based index. Only primitive elements are included.
        max_depth: Maximum nesting level to flatten (0 = only immediate children).
                   Only effective when both primitive and the appropriate separator
                   are set.
        **keys: Custom key-value pairs where value is a function taking the object

    Returns:
        Dictionary representation of the object
    """
    result = {}

    # Iterate over all attributes
    for attr_name in dir(obj):
        # Skip special / private attributes if requested
        if public and (attr_name.startswith('_') or
                       (attr_name.startswith('__') and attr_name.endswith('__'))):
            continue

        try:
            value = getattr(obj, attr_name)
        except Exception:
            continue  # attribute not accessible

        # Try calling callables if requested
        if primitive and try_call and callable(value):
            try:
                called = value()
                # Keep the called value even if not primitive; it will be processed further
                value = called
            except Exception:
                continue  # skip if method call fails

        # ---------- flattening logic ----------
        # If primitive is True and we have a container with a matching separator,
        # flatten it (recursively) and skip adding the container itself.
        if primitive:
            is_dict = isinstance(value, dict)
            is_list = isinstance(value, (list, tuple))
            if (is_dict and flatten_dict_sep is not None) or (is_list and flatten_list_sep is not None):
                # Flatten the container; yields (flat_key, primitive_value)
                for flat_key, flat_val in _flatten_container(
                    value,
                    attr_name,
                    flatten_dict_sep,
                    flatten_list_sep,
                    max_depth
                ):
                    result[flat_key] = flat_val
                continue   # do not add the raw attribute

            # If not a flattable container, check primitive constraint
            if not _is_primitive(value):
                continue

        # Add the attribute (or its processed value) as is
        result[attr_name] = value

    # Apply custom transformations
    for key, func in keys.items():
        try:
            result[key] = func(obj)
        except Exception:
            pass

    return result

import pandas as pd
import numpy as np
from functools import wraps

def apply_robust(self, func, errors='raise', **kwargs):
    """
    Apply a function to each element of the Series with robust error handling.
    
    Parameters
    ----------
    func : function
        Function to apply to each element
    errors : str, default 'raise'
        Error handling strategy:
        - 'raise': raise the exception (default pandas behavior)
        - 'nan': fill with NaN when errors occur
        - 'original' or 'ignore': fill with original value when errors occur
    **kwargs : additional arguments
        Passed to the original apply function
        
    Returns
    -------
    pd.Series
        Series with applied function or handled errors
    """
    
    if errors == 'raise':
        # Default behavior - use original apply
        return self.apply(func, **kwargs)
    
    elif errors in ['nan', 'original', 'ignore']:
        # Create a result array with the same index
        result = []
        
        for idx, val in self.items():
            try:
                # Try to apply the function
                result.append(func(val, **kwargs))
            except Exception:
                if errors == 'nan':
                    result.append(np.nan)
                elif errors in ['original', 'ignore']:
                    result.append(val)
        
        # Return as Series with original index
        return pd.Series(result, index=self.index, name=self.name)
    
    else:
        raise ValueError(f"errors must be 'raise', 'nan', 'original', or 'ignore'. Got '{errors}'")

# Monkey patch pd.Series
pd.Series.apply_robust = apply_robust

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

def download_favicon(url, save_path=None):
    """
    Download favicon from a website using requests and BeautifulSoup.
    
    Args:
        url (str): The website URL to get favicon from
        save_path (str, optional): Path to save the favicon file. 
                                   If None, returns bytes.
    
    Returns:
        bytes or None: If save_path is None, returns the favicon bytes.
                       If save_path is provided, returns None after saving.
                       Returns None if favicon cannot be found.
    """
    try:
        # Ensure URL has scheme
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Get the HTML content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML to find favicon
        soup = BeautifulSoup(response.text, 'html.parser')
        favicon_url = None
        
        # Look for favicon in link tags
        favicon_selectors = [
            "link[rel='icon']",
            "link[rel='shortcut icon']",
            "link[rel='apple-touch-icon']",
            "link[rel='apple-touch-icon-precomposed']"
        ]
        
        for selector in favicon_selectors:
            link = soup.select_one(selector)
            if link and link.get('href'):
                favicon_url = link.get('href')
                break
        
        # If no favicon found in HTML, try standard locations
        if not favicon_url:
            # Try common favicon locations
            base_url = url.rstrip('/')
            possible_paths = [
                '/favicon.ico',
                '/favicon.png',
                '/favicon.jpg',
                '/favicon.svg'
            ]
            
            for path in possible_paths:
                test_url = base_url + path
                try:
                    test_response = requests.head(test_url, headers=headers, timeout=5)
                    if test_response.status_code == 200:
                        favicon_url = test_url
                        break
                except:
                    continue
        
        # If still no favicon, try root domain
        if not favicon_url:
            parsed_url = urlparse(url)
            root_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            test_url = root_url + '/favicon.ico'
            try:
                test_response = requests.head(test_url, headers=headers, timeout=5)
                if test_response.status_code == 200:
                    favicon_url = test_url
            except:
                pass
        
        if not favicon_url:
            print(f"No favicon found for {url}")
            return None
        
        # Make URL absolute if relative
        if not favicon_url.startswith(('http://', 'https://')):
            favicon_url = urljoin(url, favicon_url)
        
        # Download the favicon
        favicon_response = requests.get(favicon_url, headers=headers, timeout=10)
        favicon_response.raise_for_status()
        
        favicon_data = favicon_response.content
        
        # Save or return
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            with open(save_path, 'wb') as f:
                f.write(favicon_data)
            print(f"Favicon saved to: {save_path}")
            return None
        else:
            return favicon_data
            
    except requests.RequestException as e:
        print(f"Error downloading favicon: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


# # Example usage:
# if __name__ == "__main__":
#     # Get favicon as bytes
#     favicon_bytes = download_favicon("https://www.google.com")
#     if favicon_bytes:
#         print(f"Downloaded {len(favicon_bytes)} bytes")
    
#     # Save favicon to file
#     download_favicon("https://www.github.com", "github_favicon.ico")
#     download_favicon("https://www.python.org", "python_favicon.ico")

from PIL import Image
import pathlib
import io
from typing import Union, Optional

def convert_image(
    input_data: Union[str, pathlib.Path, bytes],
    output_format: str,
    output_path: Optional[Union[str, pathlib.Path]] = None,
    **kwargs
) -> Optional[bytes]:
    """
    Convert an image between formats.
    
    Args:
        input_data: Image source - can be a string path, pathlib.Path, or bytes
        output_format: Target format (e.g., 'PNG', 'JPEG', 'WEBP', 'BMP', 'TIFF')
        output_path: Optional path to save the converted image. If not provided,
                    returns bytes of the converted image.
        **kwargs: Additional arguments passed to Image.save() (e.g., quality, optimize)
    
    Returns:
        bytes if output_path is None, else None
    
    Raises:
        ValueError: If input format is invalid or unsupported
        FileNotFoundError: If input file doesn't exist
        PIL.UnidentifiedImageError: If image data is corrupted or invalid
    
    Examples:
        # Convert from file path to bytes
        result_bytes = convert_image('input.jpg', 'PNG')
        
        # Convert from bytes to file
        with open('input.jpg', 'rb') as f:
            image_bytes = f.read()
        convert_image(image_bytes, 'WEBP', 'output.webp')
        
        # Convert between paths
        convert_image('input.png', 'JPEG', 'output.jpg', quality=85)
        
        # Convert from pathlib.Path
        from pathlib import Path
        convert_image(Path('input.bmp'), 'TIFF', 'output.tiff')
    """
    
    # Normalize output_format
    output_format = output_format.upper()
    
    # Load the image
    try:
        if isinstance(input_data, (str, pathlib.Path)):
            # Handle string or Path input
            path = pathlib.Path(input_data)
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")
            image = Image.open(path)
        elif isinstance(input_data, bytes):
            # Handle bytes input
            image = Image.open(io.BytesIO(input_data))
        else:
            raise ValueError(
                f"Input must be str, pathlib.Path, or bytes, got {type(input_data).__name__}"
            )
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")
    
    # Convert to RGB for JPEG if needed (JPEG doesn't support transparency)
    if output_format == 'JPEG' and image.mode in ('RGBA', 'P', 'LA'):
        # Create a white background for transparency
        if image.mode == 'P':
            image = image.convert('RGBA')
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        else:
            image = image.convert('RGB')
    
    # Handle output
    if output_path is None:
        # Return bytes
        buffer = io.BytesIO()
        image.save(buffer, format=output_format, **kwargs)
        return buffer.getvalue()
    else:
        # Save to file
        output_path = pathlib.Path(output_path)
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with format specified
        image.save(output_path, format=output_format, **kwargs)
        return None


# Convenience functions for common conversions
def convert_to_jpeg(input_data, output_path=None, quality=85, **kwargs):
    """Convert image to JPEG format."""
    return convert_image(input_data, 'JPEG', output_path, quality=quality, **kwargs)


def convert_to_png(input_data, output_path=None, optimize=True, **kwargs):
    """Convert image to PNG format."""
    return convert_image(input_data, 'PNG', output_path, optimize=optimize, **kwargs)


def convert_to_webp(input_data, output_path=None, quality=80, **kwargs):
    """Convert image to WEBP format."""
    return convert_image(input_data, 'WEBP', output_path, quality=quality, **kwargs)


# # Example usage and testing
# if __name__ == "__main__":
#     # Example 1: String path to bytes
#     try:
#         result = convert_image('example.jpg', 'PNG')
#         print(f"Conversion to bytes successful. Size: {len(result)} bytes")
#     except FileNotFoundError:
#         print("Example file not found, skipping...")
    
#     # Example 2: Path to file
#     from pathlib import Path
#     try:
#         convert_image(Path('example.png'), 'JPEG', 'output.jpg', quality=90)
#         print("Successfully saved to output.jpg")
#     except FileNotFoundError:
#         print("Example file not found, skipping...")
    
#     # Example 3: Bytes to bytes
#     try:
#         with open('example.jpg', 'rb') as f:
#             image_bytes = f.read()
#         result = convert_image(image_bytes, 'WEBP')
#         print(f"Bytes to bytes conversion successful. Size: {len(result)} bytes")
#     except FileNotFoundError:
#         print("Example file not found, skipping...")

import bs4
from typing import Iterator, Union

def navigable_texts(
    root: Union[bs4.element.PageElement, bs4.element.NavigableString],
    non_empty: bool = True,
) -> Iterator[bs4.element.NavigableString]:
    """Recursively yield all `NavigableString` nodes beneath `root`.

    This function walks the BeautifulSoup parse tree rooted at `root` in
    document order and yields every `bs4.element.NavigableString` it
    encounters. It works uniformly whether `root` is itself a string node
    or any tag-like element that exposes a ``children`` iterator.

    Args:
        root: The node to start traversal from. This may be a
            `bs4.element.NavigableString` (in which case it is yielded
            directly, subject to `non_empty`) or any BeautifulSoup element
            that provides a ``children`` attribute (e.g. `Tag`,
            `BeautifulSoup`).
        non_empty: If ``True`` (the default), string nodes whose content is
            empty or consists solely of whitespace are skipped. If
            ``False``, every `NavigableString` encountered is yielded,
            including whitespace-only nodes.

    Yields:
        Each `bs4.element.NavigableString` found in the subtree, in the
        order they appear in the document.

    Examples:
        >>> from bs4 import BeautifulSoup
        >>> soup = BeautifulSoup("<p>Hello <b>world</b>!</p>", "html.parser")
        >>> [str(s) for s in navigable_texts(soup)]
        ['Hello ', 'world', '!']
        >>> [str(s) for s in navigable_texts(soup, non_empty=False)]
        ['Hello ', 'world', '!']
    """
    if isinstance(root, bs4.element.NavigableString):
        if non_empty and str(root).strip() != "":
            yield root
    elif hasattr(root, "children"):
        for child in root.children:
            yield from navigable_texts(child, non_empty=non_empty)

import json
from typing import Optional, Dict, Tuple, Callable, Any, Iterator, Union
from pathlib import Path

# Assume the following external helpers exist (provided elsewhere):
# - is_primitive(type(obj)) -> bool
# - _process_primitive_obj_with_types(obj, kv_getter) -> (typ, primitive_kvs, non_primitive_kvs)
# - dict_kv_getter, list_kv_getter
# - python_object_id, increment_idgen (example ID generators)

"""
record_generator.py
===================

Structured record generation from JSON-like Python objects.

The module traverses a nested ``dict`` / ``list`` structure and yields one
record per non-primitive container.  Each record carries:

    __id               unique identifier for the container
    __id_gen           name of the ID generator that produced __id
    __type             stable type name derived from a type signature
    __depth            nesting depth (root == 0)
    __primitive_values the (possibly transformed) container

Type inference is signature-based: a container's *type* is uniquely defined
by the ``frozenset`` of ``(field_name, primitive_type_name)`` tuples for its
primitive-valued fields.  FOREIGN_KEY wrappers (emitted when
``foreign_key=True``) are treated as a first-class primitive type.  The
inference strategy is pluggable via the ``type_inferrer`` constructor
argument.

Optionally, a preprocessing pipeline can be injected to rewrite dicts
before they are typed — useful for normalising polymorphic structures.

The module is self-contained: it only depends on the standard library.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)


# =====================================================================
# 1. Primitive detection
# =====================================================================
# A value is *primitive* if its Python type is a leaf type, OR if it is a
# FOREIGN_KEY wrapper produced by RecordGenerator.  Treating the wrapper
# as a primitive lets a parent dict's type signature record "an object
# used to live here" without exploding the signature with recursive data.

_PRIMITIVE_TYPES: Tuple[type, ...] = (bool, int, float, str, type(None))


def is_primitive(t: type) -> bool:
    """Type-level primitive test (used at the top of ``traverse``)."""
    return t in _PRIMITIVE_TYPES


def is_foreign_key_value(v: Any) -> bool:
    """Recognise the wrapper emitted when ``foreign_key=True``."""
    return (
        isinstance(v, dict)
        and v.get("__type") == "FOREIGN_KEY"
        and "__foreign_id" in v
    )


def is_primitive_value(v: Any) -> bool:
    """Value-level primitive test — FOREIGN_KEY counts as primitive."""
    return is_primitive(type(v)) or is_foreign_key_value(v)


# =====================================================================
# 2. Key/value getters
# =====================================================================

def dict_kv_getter(d: Dict[Any, Any]) -> Iterator[Tuple[Any, Any]]:
    return iter(d.items())


def list_kv_getter(l: List[Any]) -> Iterator[Tuple[Any, Any]]:
    return iter(enumerate(l))


# =====================================================================
# 3. Primitive type naming and signature hashing
# =====================================================================

def _primitive_type_name(v: Any) -> str:
    """
    Return the *type name string* used inside a type signature.

    Order matters: ``bool`` must be tested before ``int`` because ``bool``
    is a subclass of ``int``.  FOREIGN_KEY is checked first because its
    wrapper is a ``dict`` and would otherwise look non-primitive.
    """
    if is_foreign_key_value(v):
        return "FOREIGN_KEY"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if v is None:
        return "null"
    return type(v).__name__          # extensible catch-all


# A module-level registry makes type names *stable* across a run:
#   same signature  -> same name
#   different signature -> different name
_SIGNATURE_REGISTRY: Dict[FrozenSet[Tuple[str, str]], str] = {}
_SIGNATURE_COUNTER = itertools.count(1)


def _name_for_signature(signature: FrozenSet[Tuple[str, str]], kind: str) -> str:
    if signature not in _SIGNATURE_REGISTRY:
        _SIGNATURE_REGISTRY[signature] = f"{kind}_T{next(_SIGNATURE_COUNTER)}"
    return _SIGNATURE_REGISTRY[signature]


def _dict_signature(d: Dict[Any, Any]) -> FrozenSet[Tuple[str, str]]:
    """
    Every dict type is uniquely defined by the frozenset of
    ``(field_name, primitive_type_name)`` for its primitive-valued fields.

    Non-primitive fields do not participate — they are typed recursively,
    and including them would explode the signature.
    """
    return frozenset(
        (str(k), _primitive_type_name(v))
        for k, v in d.items()
        if is_primitive_value(v)
    )


def _list_signature(l: List[Any]) -> FrozenSet[Tuple[str, str]]:
    """
    A list has no named fields, so we use ``(str(index), typename)`` for
    its primitive elements.  ``[1, 2, 3]`` and ``[4, 5, 6]`` therefore
    share a type; ``["a", "b"]`` is a different type.

    Override this if you need e.g. order-independent (multiset) semantics.
    """
    return frozenset(
        (str(i), _primitive_type_name(v))
        for i, v in enumerate(l)
        if is_primitive_value(v)
    )


# =====================================================================
# 4. Pluggable type inference
# =====================================================================
# Contract:
#     (obj, kv_getter) -> (type_name,
#                          primitive_kvs,       # [(k, v), ...]
#                          non_primitive_kvs)   # [(k, v), ...]
#
# Swap in a custom implementation via ``RecordGenerator(type_inferrer=...)``
# to support schema-based typing, regex-based field grouping, etc.

def _process_primitive_obj_with_types(
    obj: Union[dict, list],
    kv_getter: Callable[[Any], Iterator[Tuple[Any, Any]]],
) -> Tuple[str, List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
    """
    Default type-inference strategy.

    Partitions ``obj`` into primitive and non-primitive key/value pairs
    and derives a stable type name from the frozenset of primitive
    ``(field, type)`` tuples.  FOREIGN_KEY wrappers count as primitive.
    """
    primitive_kvs:     List[Tuple[Any, Any]] = []
    non_primitive_kvs: List[Tuple[Any, Any]] = []

    for k, v in kv_getter(obj):
        (primitive_kvs if is_primitive_value(v) else non_primitive_kvs).append((k, v))

    if isinstance(obj, dict):
        signature = _dict_signature(obj)
        kind = "Dict"
    elif isinstance(obj, list):
        signature = _list_signature(obj)
        kind = "List"
    else:
        raise TypeError(f"Unsupported container type: {type(obj)!r}")

    return _name_for_signature(signature, kind), primitive_kvs, non_primitive_kvs


# =====================================================================
# 5. ID generators
# =====================================================================
# Contract:
#     id_gen()            -> (state, initial_id)
#     id_gen(obj, state)  -> (state, next_id)

def python_object_id(obj: Any = None, state: Any = None) -> Tuple[Any, int]:
    """ID = CPython's ``id()``.  State is unused but kept for API symmetry."""
    return (state, id(obj))


def increment_idgen(obj: Any = None, state: Any = None) -> Tuple[Any, int]:
    """Monotonic integer IDs starting at 1."""
    if state is None:
        state = itertools.count(1)
    return (state, next(state))


# =====================================================================
# 6. RecordGenerator
# =====================================================================

class RecordGenerator:
    """
    Generates structured records from a JSON-like object, assigning IDs,
    optionally handling foreign keys, and supporting custom preprocessing
    and type inference.
    """

    def __init__(
        self,
        id_generator: Callable[..., Tuple[Any, Any]] = python_object_id,
        preprocessing_type_handler: Optional[
            Dict[
                str,
                Tuple[
                    Callable[[Dict], bool],   # predicate
                    Callable[[Dict], Dict],   # transformer
                ],
            ]
        ] = None,
        type_inferrer: Callable[
            [Union[dict, list], Callable[[Any], Iterator[Tuple[Any, Any]]]],
            Tuple[str, List[Tuple[Any, Any]], List[Tuple[Any, Any]]],
        ] = _process_primitive_obj_with_types,
    ) -> None:
        """
        Parameters
        ----------
        id_generator : callable, optional
            Returns ``(state, id)``.  Called as ``id_generator()`` for
            initialisation, and ``id_generator(obj, state)`` afterwards.
            Defaults to :func:`python_object_id`.

        preprocessing_type_handler : dict, optional
            Mapping ``type_name -> (predicate, transformer)``.  For each
            dict encountered, predicates are evaluated in dict key order;
            the first match triggers its transformer.  The transformer may
            mutate in place or return a new dict.

        type_inferrer : callable, optional
            Custom strategy for type inference.  Defaults to
            :func:`_process_primitive_obj_with_types`.
        """
        self.id_gen = id_generator
        self.state, self.current_id = id_generator()
        self.preprocessing_handler = preprocessing_type_handler
        self.type_inferrer = type_inferrer

    # -----------------------------------------------------------------

    def next_id(self, obj: Any) -> Any:
        """Advance the ID generator and return the next ID."""
        self.state, self.current_id = self.id_gen(obj, self.state)
        return self.current_id

    # -----------------------------------------------------------------

    def _apply_preprocessing(self, obj: Dict) -> Dict:
        """
        Apply the preprocessing pipeline to a dict, if any handler matches.

        Handlers are tested in insertion order.  The first matching
        predicate's transformer is invoked; if it returns a new dict, that
        dict replaces the original.
        """
        if self.preprocessing_handler is None:
            return obj

        for _type_name, (predicate, transformer) in self.preprocessing_handler.items():
            if predicate(obj):
                transformed = transformer(obj)
                if transformed is not None and transformed is not obj:
                    obj = transformed
                break
        return obj

    # -----------------------------------------------------------------

    def traverse_and_edit_records(
        self,
        json_object: Any,
        *,
        depth: int = 0,
        foreign_key: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """
        Traverse the JSON-like object and yield record dictionaries.

        Parameters
        ----------
        json_object : Any
            The root object (dict, list, or primitive).

        depth : int, optional
            Current nesting depth (used internally).  Default 0.

        foreign_key : bool, optional
            If True, any non-primitive value inside a dict is replaced with
            a foreign-key wrapper (``__type``, ``__foreign_id``,
            ``__foreign_id_gen``).  The wrapped object is still traversed
            separately.  Default False.

        Yields
        ------
        dict
            A record containing metadata and the (possibly transformed)
            object.
        """
        # ---------- Skip primitives ----------
        if is_primitive(type(json_object)):
            return

        # ---------- Apply preprocessing to dicts ----------
        if isinstance(json_object, dict):
            json_object = self._apply_preprocessing(json_object)

        # ---------- Analyse structure & infer type ----------
        kv_getter = dict_kv_getter if isinstance(json_object, dict) else list_kv_getter
        typ, _primitive_kvs, non_primitive_kvs = self.type_inferrer(
            json_object, kv_getter
        )

        # ---------- Foreign-key substitution (only for dicts) ----------
        if foreign_key and isinstance(json_object, dict):
            for k, v in non_primitive_kvs:
                json_object[k] = {
                    "__type": "FOREIGN_KEY",
                    "__foreign_id": self.next_id(v),
                    "__foreign_id_gen": self.id_gen.__name__,
                    # "__value": v,  # commented out to reduce size
                }

        # ---------- Yield record for the current object ----------
        yield {
            "__id": self.next_id(json_object),
            "__id_gen": self.id_gen.__name__,
            "__type": typ,
            "__depth": depth,
            "__primitive_values": json_object,
        }

        # ---------- Recurse into non-primitive children ----------
        for _, v in non_primitive_kvs:
            yield from self.traverse_and_edit_records(
                v,
                depth=depth + 1,
                foreign_key=foreign_key,
            )


# =====================================================================
# 7. Example: custom type inferrer
# =====================================================================

def schema_based_inferrer(
    obj: Union[dict, list],
    kv_getter: Callable[[Any], Iterator[Tuple[Any, Any]]],
) -> Tuple[str, List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
    """
    Prefer an explicit ``__schema__`` field if present, otherwise fall back
    to the default signature-based inference.
    """
    if isinstance(obj, dict) and isinstance(obj.get("__schema__"), str):
        primitive_kvs:     List[Tuple[Any, Any]] = []
        non_primitive_kvs: List[Tuple[Any, Any]] = []
        for k, v in kv_getter(obj):
            (primitive_kvs if is_primitive_value(v) else non_primitive_kvs).append((k, v))
        return f"Schema::{obj['__schema__']}", primitive_kvs, non_primitive_kvs
    return _process_primitive_obj_with_types(obj, kv_getter)


# =====================================================================
# 8. Demo
# =====================================================================

# if __name__ == "__main__":
#     # --- preprocessing handlers -----------------------------------
#     def is_person(d: dict) -> bool:
#         return d.get("kind") == "person"

#     def transform_person(d: dict) -> dict:
#         d["full_name"] = f"{d['first']} {d['last']}"
#         return d

#     handlers = {
#         "person": (is_person, transform_person),
#     }

#     user_json = {
#         "kind": "person",
#         "first": "Alice",
#         "last": "Smith",
#         "age": 30,
#         "address": {
#             "city": "Wonderland",
#             "zip": 12345,
#             "geo": {"lat": 1.5, "lon": -2.5},
#         },
#         "tags": ["admin", "editor"],
#     }

#     gen = RecordGenerator(
#         id_generator=increment_idgen,
#         preprocessing_type_handler=handlers,
#         # type_inferrer=schema_based_inferrer,   # swap in if desired
#     )

#     output_path = Path("output.jsonl")
#     with output_path.open("w", encoding="utf-8") as f:
#         for record in gen.traverse_and_edit_records(user_json, foreign_key=True):
#             line = json.dumps(record)
#             print(line)
#             print(line, file=f)
