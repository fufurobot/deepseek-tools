#!/usr/bin/env python3
"""
concat.py - Concatenate files with specified extensions into a single output file.

Usage:
    concat.py [OPTIONS] [EXTENSIONS...]

Examples:
    concat.py txt md py
    concat.py -e txt md py -o output.txt
    concat.py --extensions py js html --no-header
    concat.py -e txt -d ./src -o combined.txt
    concat.py -e txt md --exclude-dir node_modules .git
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Set, Optional
import fnmatch
from datetime import datetime
import json


class FileConcatenator:
    def __init__(
        self,
        extensions: List[str],
        output_file: str = "combined.txt",
        root_dir: str = ".",
        exclude_dirs: Optional[List[str]] = None,
        exclude_files: Optional[List[str]] = None,
        include_hidden: bool = False,
        show_header: bool = True,
        show_file_separator: bool = True,
        separator_format: str = "--- BEGIN `{file}` ---\n{content}\n--- END `{file}` ---\n",
        max_file_size: Optional[int] = None,
        recursive: bool = True,
        verbose: bool = False,
        generate_summary: bool = False,
        output_format: str = "text",  # text, json, markdown
    ):
        self.extensions = [ext.lstrip('.') for ext in extensions]
        self.output_file = output_file
        self.root_dir = Path(root_dir)
        self.exclude_dirs = exclude_dirs or []
        self.exclude_files = exclude_files or []
        self.include_hidden = include_hidden
        self.show_header = show_header
        self.show_file_separator = show_file_separator
        self.separator_format = separator_format
        self.max_file_size = max_file_size
        self.recursive = recursive
        self.verbose = verbose
        self.generate_summary = generate_summary
        self.output_format = output_format
        
        # Statistics
        self.total_files = 0
        self.total_size = 0
        self.skipped_files = 0
        self.error_files = 0
        self.processed_files = []

    def should_exclude_dir(self, dir_path: Path) -> bool:
        """Check if directory should be excluded."""
        dir_name = dir_path.name
        if not self.include_hidden and dir_name.startswith('.'):
            return True
        for pattern in self.exclude_dirs:
            if fnmatch.fnmatch(dir_name, pattern):
                return True
            if fnmatch.fnmatch(str(dir_path), pattern):
                return True
        return False

    def should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded."""
        file_name = file_path.name
        if not self.include_hidden and file_name.startswith('.'):
            return True
        for pattern in self.exclude_files:
            if fnmatch.fnmatch(file_name, pattern):
                return True
            if fnmatch.fnmatch(str(file_path), pattern):
                return True
        return False

    def should_include_file(self, file_path: Path) -> bool:
        """Check if file should be included based on extension and other criteria."""
        # Check extension
        if file_path.suffix.lstrip('.') not in self.extensions:
            return False
        
        # Check file size
        if self.max_file_size is not None:
            try:
                if file_path.stat().st_size > self.max_file_size:
                    if self.verbose:
                        print(f"Skipping {file_path}: file too large ({file_path.stat().st_size} > {self.max_file_size})")
                    self.skipped_files += 1
                    return False
            except OSError:
                pass
        
        return True

    def get_files(self) -> List[Path]:
        """Get all files matching the criteria."""
        files = []
        
        if self.recursive:
            for root, dirs, filenames in os.walk(self.root_dir):
                root_path = Path(root)
                
                # Remove excluded directories from walk
                dirs[:] = [d for d in dirs if not self.should_exclude_dir(root_path / d)]
                
                for filename in filenames:
                    file_path = root_path / filename
                    if self.should_exclude_file(file_path):
                        continue
                    if self.should_include_file(file_path):
                        files.append(file_path)
        else:
            for item in self.root_dir.iterdir():
                if item.is_file():
                    if self.should_exclude_file(item):
                        continue
                    if self.should_include_file(item):
                        files.append(item)
        
        # Sort files for consistent output
        return sorted(files)

    def read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.total_size += len(content.encode('utf-8'))
                return content
        except UnicodeDecodeError:
            # Try with different encoding or skip binary files
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    self.total_size += len(content.encode('latin-1'))
                    return content
            except Exception as e:
                if self.verbose:
                    print(f"Error reading {file_path}: {e}", file=sys.stderr)
                self.error_files += 1
                return None
        except Exception as e:
            if self.verbose:
                print(f"Error reading {file_path}: {e}", file=sys.stderr)
            self.error_files += 1
            return None

    def format_file_content(self, file_path: Path, content: str) -> str:
        """Format file content with header and separator."""
        if self.show_file_separator:
            return self.separator_format.format(file=file_path, content=content)
        else:
            return content

    def generate_header(self) -> str:
        """Generate header for the output."""
        if not self.show_header:
            return ""
        
        header = "=" * 80 + "\n"
        header += f"CONCATENATED FILES\n"
        header += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Root directory: {self.root_dir.absolute()}\n"
        header += f"Extensions: {', '.join(self.extensions)}\n"
        header += f"Total files: {self.total_files}\n"
        header += f"Total size: {self.total_size:,} bytes\n"
        if self.exclude_dirs:
            header += f"Excluded directories: {', '.join(self.exclude_dirs)}\n"
        if self.exclude_files:
            header += f"Excluded files: {', '.join(self.exclude_files)}\n"
        header += "=" * 80 + "\n\n"
        return header

    def generate_summary_data(self) -> dict:
        """Generate summary data for JSON output."""
        return {
            "timestamp": datetime.now().isoformat(),
            "root_directory": str(self.root_dir.absolute()),
            "extensions": self.extensions,
            "total_files": self.total_files,
            "total_size_bytes": self.total_size,
            "skipped_files": self.skipped_files,
            "error_files": self.error_files,
            "files": self.processed_files,
            "excluded_dirs": self.exclude_dirs,
            "excluded_files": self.exclude_files,
        }

    def concatenate(self) -> bool:
        """Main concatenation process."""
        files = self.get_files()
        self.total_files = len(files)
        
        if self.total_files == 0:
            print(f"No files found with extensions: {', '.join(self.extensions)}", file=sys.stderr)
            return False
        
        if self.verbose:
            print(f"Found {self.total_files} files to process")
        
        try:
            if self.output_format == "json":
                # Collect all content for JSON output
                all_content = {}
                for file_path in files:
                    content = self.read_file_content(file_path)
                    if content is not None:
                        rel_path = str(file_path.relative_to(self.root_dir))
                        all_content[rel_path] = content
                        self.processed_files.append(rel_path)
                
                summary = self.generate_summary_data()
                summary["files_content"] = all_content
                
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                
                if self.verbose:
                    print(f"JSON output written to {self.output_file}")
                return True
            
            elif self.output_format == "markdown":
                # Markdown format
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    if self.show_header:
                        f.write(f"# Concatenated Files\n\n")
                        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
                        f.write(f"**Root directory:** {self.root_dir.absolute()}  \n")
                        f.write(f"**Extensions:** {', '.join(self.extensions)}  \n")
                        f.write(f"**Total files:** {self.total_files}  \n\n")
                        f.write("---\n\n")
                    
                    for file_path in files:
                        content = self.read_file_content(file_path)
                        if content is not None:
                            rel_path = str(file_path.relative_to(self.root_dir))
                            self.processed_files.append(rel_path)
                            f.write(f"## `{rel_path}`\n\n")
                            f.write("```" + file_path.suffix.lstrip('.') + "\n")
                            f.write(content)
                            if not content.endswith('\n'):
                                f.write('\n')
                            f.write("```\n\n")
                
                if self.verbose:
                    print(f"Markdown output written to {self.output_file}")
                return True
            
            else:  # text format (default)
                # Standard text format
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    # Write header
                    header = self.generate_header()
                    f.write(header)
                    
                    # Write file contents
                    for file_path in files:
                        content = self.read_file_content(file_path)
                        if content is not None:
                            rel_path = str(file_path.relative_to(self.root_dir))
                            self.processed_files.append(rel_path)
                            
                            if self.show_file_separator:
                                f.write(self.separator_format.format(file=rel_path, content=content))
                            else:
                                f.write(content)
                            
                            if not content.endswith('\n') and self.show_file_separator:
                                f.write('\n')
                            
                            if self.show_file_separator:
                                f.write('\n')
                
                if self.verbose:
                    print(f"Text output written to {self.output_file}")
                
                # Generate summary if requested
                if self.generate_summary:
                    summary_file = self.output_file + ".summary.json"
                    summary = self.generate_summary_data()
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        json.dump(summary, f, indent=2)
                    if self.verbose:
                        print(f"Summary written to {summary_file}")
                
                return True
                
        except IOError as e:
            print(f"Error writing to {self.output_file}: {e}", file=sys.stderr)
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate files with specified extensions into a single output file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s txt md py                    # Concatenate all .txt, .md, .py files
  %(prog)s -e txt md -o output.txt      # Specify output file
  %(prog)s -e py --no-header            # Don't include header
  %(prog)s -e txt -d ./src              # Process only src directory
  %(prog)s -e txt --exclude-dir tests   # Exclude tests directory
  %(prog)s -e txt --max-size 1MB        # Skip files larger than 1MB
  %(prog)s -e txt --format markdown     # Generate Markdown output
  %(prog)s -e txt --format json         # Generate JSON output
        """
    )
    
    # Positional arguments for extensions
    parser.add_argument(
        'extensions',
        nargs='*',
        help='File extensions to concatenate (e.g., txt md py)'
    )
    
    # Optional arguments
    parser.add_argument(
        '-e', '--extensions',
        nargs='+',
        help='File extensions to concatenate (alternative to positional)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='combined.txt',
        help='Output file name (default: combined.txt)'
    )
    
    parser.add_argument(
        '-d', '--directory',
        default='.',
        help='Root directory to search (default: current directory)'
    )
    
    parser.add_argument(
        '--exclude-dir',
        action='append',
        help='Directory patterns to exclude (can be used multiple times)'
    )
    
    parser.add_argument(
        '--exclude-file',
        action='append',
        help='File patterns to exclude (can be used multiple times)'
    )
    
    parser.add_argument(
        '--include-hidden',
        action='store_true',
        help='Include hidden files and directories'
    )
    
    parser.add_argument(
        '--no-header',
        action='store_true',
        help='Do not include header in output'
    )
    
    parser.add_argument(
        '--no-separator',
        action='store_true',
        help='Do not include file separators'
    )
    
    parser.add_argument(
        '--separator-format',
        default='--- BEGIN `{file}` ---\n{content}\n--- END `{file}` ---\n',
        help='Format for file separators (default: "--- BEGIN `{file}` ---\\n{content}\\n--- END `{file}` ---\\n")'
    )
    
    parser.add_argument(
        '--max-size',
        help='Maximum file size to include (e.g., 1MB, 512KB, 1024B)'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search recursively'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Generate summary JSON file (for text format only)'
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json', 'markdown'],
        default='text',
        help='Output format (default: text)'
    )
    
    args = parser.parse_args()
    
    # Combine extensions from positional and -e flag
    extensions = list(args.extensions) if args.extensions else []
    if args.extensions is not None:
        extensions.extend(args.extensions)
    
    if not extensions:
        parser.error("No extensions specified. Please provide at least one file extension.")
    
    # Parse max file size
    max_size = None
    if args.max_size:
        size_str = args.max_size.upper()
        multipliers = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        for unit, multiplier in multipliers.items():
            if size_str.endswith(unit):
                try:
                    number = float(size_str[:-len(unit)])
                    max_size = int(number * multiplier)
                    break
                except ValueError:
                    pass
        if max_size is None:
            try:
                max_size = int(size_str)
            except ValueError:
                parser.error(f"Invalid size format: {args.max_size}. Use e.g., 1MB, 512KB, 1024B")
    
    # Create concatenator instance
    concat = FileConcatenator(
        extensions=extensions,
        output_file=args.output,
        root_dir=args.directory,
        exclude_dirs=args.exclude_dir,
        exclude_files=args.exclude_file,
        include_hidden=args.include_hidden,
        show_header=not args.no_header,
        show_file_separator=not args.no_separator,
        separator_format=args.separator_format,
        max_file_size=max_size,
        recursive=not args.no_recursive,
        verbose=args.verbose,
        generate_summary=args.summary,
        output_format=args.format,
    )
    
    # Run concatenation
    success = concat.concatenate()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()