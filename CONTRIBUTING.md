# Contributing to viva_math

Thanks for your interest in contributing to viva_math!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/mrootx/viva_math.git
cd viva_math

# Install dependencies
gleam deps download

# Build
gleam build

# Run tests
gleam test

# Format code
gleam format
```

## Guidelines

### Code Style

- Run `gleam format` before committing
- Follow existing patterns in the codebase
- Add doc comments (`////`) for public functions
- Include examples in doc comments when helpful

### Testing

- Add tests for new functions in `test/viva_math_test.gleam`
- Tests should be self-contained and descriptive
- Use `is_close` helper for floating point comparisons

### Documentation

- Update `README.md` for new modules
- Add entries to `CHANGELOG.md` under `[Unreleased]`
- Include academic references when implementing algorithms

### Commit Messages

Use conventional commits:

```
feat: add new function to entropy module
fix: correct bistability threshold in cusp
docs: update README with new examples
test: add tests for vector operations
refactor: simplify trigonometric_roots
```

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes
4. Run `gleam test` and `gleam format`
5. Commit with descriptive message
6. Push and open a PR

## Questions?

Open an issue or reach out to @mrootx.

## License

By contributing, you agree that your contributions will be licensed under MIT.
