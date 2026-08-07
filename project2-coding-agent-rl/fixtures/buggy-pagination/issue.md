`paginate` documents and validates pages as 1-indexed, but currently returns
the wrong slice for valid page numbers. Correct the offset calculation while
preserving argument validation and partial final pages. Do not modify tests.
