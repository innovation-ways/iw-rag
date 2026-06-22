# Contributing to Innovation Ways RAG

Thank you for your interest in contributing. This project is maintained by Innovation Ways and welcomes community contributions.

## Quick Start

1. [Open an issue](https://github.com/innovation-ways/Innovation Ways RAG/issues/new/choose) to discuss significant changes before coding.
2. Fork the repository and create a feature branch.
3. Make your changes with tests where applicable.
4. Ensure all checks pass locally (`pre-commit run --all-files`).
5. Sign off each commit (DCO — see below).
6. Open a Pull Request against `main`.

## Reporting Issues

- **Bugs**: Use the `Bug report` issue template.
- **Feature requests**: Use the `Feature request` issue template.
- **Security issues**: Do NOT file public issues. See [SECURITY.md](SECURITY.md).
- **Code of Conduct concerns**: Email info@innovation-ways.com.

## Development Setup

<!-- TODO: Fill in project-specific setup steps. -->

```bash
git clone https://github.com/innovation-ways/Innovation Ways RAG.git
cd Innovation Ways RAG
# ... install dependencies, run tests
```

## Developer Certificate of Origin (DCO)

**All commits must be signed off** per the [Developer Certificate of Origin](https://developercertificate.org/).

Add a `Signed-off-by` trailer to each commit automatically by using `-s` on `git commit`:

```bash
git commit -s -m "feat: add new capability"
```

This produces:

```
feat: add new capability

Signed-off-by: Your Name <your.email@example.com>
```

By signing off, you certify that you wrote the code (or have the right to submit it) under this project's license. See the [DCO text](https://developercertificate.org/) for the full certification.

If you forget to sign off a commit, you can amend:

```bash
git commit --amend -s --no-edit
```

PRs with unsigned commits will fail the DCO status check and cannot be merged.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]

Signed-off-by: ...
```

Common types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`, `perf`, `build`.

Breaking changes: add `!` after the type (`feat!: drop Python 3.9 support`) or include `BREAKING CHANGE:` in the footer.

## Pull Request Checklist

- [ ] All commits are signed off (DCO)
- [ ] Commit messages follow Conventional Commits
- [ ] Tests added/updated for the change
- [ ] Documentation updated if user-facing behavior changed
- [ ] `pre-commit run --all-files` passes locally
- [ ] No secrets, internal references, or PII in changes

## Review Process

Pull requests require at least one approving review from a project maintainer. Maintainers may request changes or ask clarifying questions; address feedback by pushing additional commits (do not force-push unless asked).

## License

By contributing, you agree that your contributions will be licensed under the [](LICENSE) license covering this project.

## Questions

If you have questions about the contribution process, open a [Discussion](https://github.com/innovation-ways/Innovation Ways RAG/discussions) or email info@innovation-ways.com.