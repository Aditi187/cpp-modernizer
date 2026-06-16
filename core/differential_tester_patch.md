BUG: Misplaced docstring after early return in compile_cpp_source
The docstring appears AFTER the early-return block, making it a dead string literal, not a real docstring.
This is valid Python but looks like a bug and confuses linters/IDEs.
