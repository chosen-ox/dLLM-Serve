# Contributing to dLLM-Serve

Thank you for your interest in contributing to dLLM-Serve! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/dllm-serve.git
   cd dllm-serve
   ```
3. **Set up your development environment** - See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed instructions

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected vs. actual behavior
- Your environment (Python version, CUDA version, GPU model, etc.)
- Relevant code snippets or error messages

### Suggesting Features

Feature suggestions are welcome! Please create an issue with:
- A clear description of the proposed feature
- Use cases and motivation
- Potential implementation approach (if you have ideas)

### Contributing Code

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, readable code following the project's style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   # Run unit tests
   pytest tests/unit/

   # Run integration tests
   pytest tests/integration/

   # Format code
   black dllmserve/ server/ tests/ examples/
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

   Use conventional commit messages:
   - `feat:` - New features
   - `fix:` - Bug fixes
   - `docs:` - Documentation changes
   - `test:` - Test additions or changes
   - `refactor:` - Code refactoring
   - `perf:` - Performance improvements
   - `chore:` - Maintenance tasks

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## Code Style Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use `black` for code formatting (line length: 100)
- Add type hints for function signatures where appropriate
- Write docstrings for public APIs (Google style)

Example:

```python
def generate_text(
    prompts: List[str],
    sampling_params: SamplingParams,
    sparse_config: Optional[SparseConfig] = None
) -> List[Dict[str, Any]]:
    """Generate text completions for the given prompts.

    Args:
        prompts: List of input prompts to generate from.
        sampling_params: Sampling configuration for generation.
        sparse_config: Optional sparse attention configuration.

    Returns:
        List of dictionaries containing generated text and metadata.
    """
    # Implementation
    pass
```

### Code Organization

- Keep functions focused and single-purpose
- Avoid deep nesting (max 3-4 levels)
- Use meaningful variable and function names
- Add comments for complex logic

## Testing Guidelines

### Unit Tests

- Write unit tests for new functions and classes
- Tests should be isolated and not depend on external state
- Use descriptive test names: `test_<what>_<condition>_<expected>`

Example:

```python
def test_sparse_attention_reduces_memory():
    """Test that sparse attention uses less memory than dense."""
    # Test implementation
    pass
```

### Integration Tests

- Test interactions between components
- Verify end-to-end workflows
- Include tests for different model types

### Performance Tests

- Add benchmarks for performance-critical code
- Ensure changes don't regress performance
- Document expected performance characteristics

## Documentation

### Code Documentation

- Add docstrings to all public functions, classes, and methods
- Include type hints in function signatures
- Document expected inputs, outputs, and side effects

### User Documentation

- Update relevant documentation in `docs/` when adding features
- Add examples to `examples/` for new functionality
- Update README.md if adding major features

## Pull Request Process

1. **Ensure all tests pass** before submitting
2. **Update documentation** to reflect your changes
3. **Keep PRs focused** - one feature or fix per PR
4. **Provide a clear description** of what your PR does and why
5. **Link related issues** using "Fixes #123" or "Closes #123"
6. **Be responsive** to review feedback

### PR Checklist

- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Code formatted with `black`
- [ ] No new warnings or errors
- [ ] Commit messages follow conventional commit format
- [ ] PR description clearly explains the changes

## Review Process

- All PRs require at least one review before merging
- Maintainers will review your PR and may request changes
- Address feedback by pushing new commits to your branch
- Once approved, a maintainer will merge your PR

## Development Workflow

### Setting Up Development Environment

See [DEVELOPMENT.md](DEVELOPMENT.md) for complete setup instructions.

### Running Tests Locally

```bash
# All tests
pytest tests/

# Specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# With coverage
pytest --cov=dllmserve tests/
```

### Code Formatting

```bash
# Format all code
black dllmserve/ server/ tests/ examples/ benchmarks/

# Check formatting without modifying
black --check dllmserve/ server/ tests/ examples/
```

## Community Guidelines

- Be respectful and constructive in all interactions
- Help newcomers and answer questions when you can
- Follow GitHub's [Community Guidelines](https://docs.github.com/en/site-policy/github-terms/github-community-guidelines)

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Ask in existing issues or PRs
- Reach out to maintainers

Thank you for contributing to dLLM-Serve!
